"""
run_pipeline.py
---------------
End-of-day orchestrator for the Nifty 500 trading model.

Usage
-----
  python src/run_pipeline.py                      # today's date
  python src/run_pipeline.py --date 2024-01-15
  python src/run_pipeline.py --date 2024-01-15 --retrain
  python src/run_pipeline.py --backtest
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date as Date
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

# ── Project root on sys.path so 'src.*' imports work when run directly ──────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_ingestion   import fetch_nifty500_data, load_all_data
from src.preprocessing    import preprocess_one, load_processed
from src.pattern_detection import detect_all_patterns, tag_dataframe
from src.feature_engineering import engineer_features, get_feature_columns
from src.model             import train_model, predict_single, OUTPUTS_DIR
from src.signals           import generate_signal, generate_daily_report
from src.backtest          import run_backtest, load_all_signals
from src.symbols           import NIFTY500

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/pipeline.log", encoding="utf-8"),
    ],
)
# Suppress noisy sub-library loggers
for _noisy in ("yfinance", "urllib3", "lightgbm", "xgboost", "shap"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger("run_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
FEATURES_DIR  = Path("data/features")
REPORTS_DIR   = Path("outputs/reports")


# ─────────────────────────────────────────────────────────────────────────────
# Step helpers
# ─────────────────────────────────────────────────────────────────────────────

def _step(n: int, title: str) -> None:
    logger.info("")
    logger.info("-" * 58)
    logger.info("  STEP %d -- %s", n, title)
    logger.info("-" * 58)


def _load_artefacts() -> tuple:
    """Load pre-trained model artefacts. Returns (xgb, lgb, le, prep, feat_cols)."""
    try:
        xgb_model = joblib.load(OUTPUTS_DIR / "xgb_model.pkl")
        lgb_model = joblib.load(OUTPUTS_DIR / "lgb_model.pkl")
        le        = joblib.load(OUTPUTS_DIR / "label_encoder.pkl")
        prep      = joblib.load(OUTPUTS_DIR / "preprocessor.pkl")
        with open(OUTPUTS_DIR / "feature_columns.json") as fh:
            feat_cols = json.load(fh)
        logger.info("Model artefacts loaded from %s", OUTPUTS_DIR)
        return xgb_model, lgb_model, le, prep, feat_cols
    except FileNotFoundError as exc:
        logger.error("Model artefacts not found: %s", exc)
        logger.error("Run with --retrain first to train the models.")
        sys.exit(1)


def _ensemble_proba(xgb_m, lgb_m, prep, feat_cols: List[str], X_row: pd.DataFrame) -> dict:
    """Return averaged probability dict for a single feature row."""
    import numpy as np

    for c in feat_cols:
        if c not in X_row.columns:
            X_row[c] = 0.0

    X = prep.transform(X_row[feat_cols])
    xgb_p = xgb_m.predict_proba(X)[0]
    lgb_p = lgb_m.predict_proba(X)[0]
    ens   = (xgb_p + lgb_p) / 2.0
    return {
        "bullish_prob": float(ens[2]) if len(ens) == 3 else float(ens[-1]),
        "neutral_prob": float(ens[1]) if len(ens) >= 2 else 0.33,
        "bearish_prob": float(ens[0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_eod_pipeline(
    run_date:    str,
    retrain:     bool = False,
    symbols:     Optional[List[str]] = None,
    skip_fetch:  bool = False,
) -> dict:
    """
    Execute the full end-of-day pipeline for `run_date`.

    Returns a summary dict with counts for the console report.
    """
    t0 = time.time()
    target_date = pd.Timestamp(run_date)
    syms = symbols or NIFTY500
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── STEP 1: Fetch data ──────────────────────────────────────────────────
    _step(1, "Fetch / update price data")
    if not skip_fetch:
        fetch_nifty500_data(
            symbols  = syms,
            period   = "2y",
            interval = "1d",
            raw_dir  = RAW_DIR,
        )
    else:
        logger.info("Skipping fetch (--skip-fetch flag set).")

    raw_data = load_all_data(raw_dir=RAW_DIR, min_trading_days=1)
    logger.info("Raw data loaded: %d stocks", len(raw_data))

    # ── STEP 2: Preprocess ──────────────────────────────────────────────────
    _step(2, "Preprocess raw OHLCV data")
    processed: Dict[str, pd.DataFrame] = {}
    for sym, df in raw_data.items():
        clean = preprocess_one(df, symbol=sym, processed_dir=PROCESSED_DIR, save=True)
        if clean is not None:
            processed[sym] = clean
    logger.info("Preprocessing done: %d / %d stocks passed", len(processed), len(raw_data))

    # ── STEP 3 & 4: Pattern detection + feature engineering (latest row) ──
    _step(3, "Pattern detection + feature engineering (latest candle)")
    feature_rows: Dict[str, pd.DataFrame] = {}
    pattern_count = 0

    for sym, df in processed.items():
        try:
            # Tag patterns on full history (needed for rolling context)
            tagged = tag_dataframe(df)

            # Feature-engineer
            feat_df = engineer_features(
                tagged, symbol=sym,
                features_dir=FEATURES_DIR, save=False,
            )
            if feat_df is None or feat_df.empty:
                continue

            # Count patterns on the last 5 candles
            recent_events = detect_all_patterns(df.tail(5))
            if recent_events:
                pattern_count += len(recent_events)

            # Keep only the most recent row for signal generation
            latest = feat_df.sort_index().tail(1).copy()
            latest.insert(0, "symbol", sym)
            feature_rows[sym] = latest

        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] Feature step failed: %s", sym, exc)

    logger.info(
        "Features computed: %d stocks | Patterns (last 5 candles): %d",
        len(feature_rows), pattern_count,
    )

    # ── STEP 5: Load (or train) models ─────────────────────────────────────
    _step(5, "Load models" + (" (+ retrain)" if retrain else ""))
    if retrain:
        combined_path = FEATURES_DIR / "combined_features.csv"
        if combined_path.exists():
            feat_all = pd.read_csv(combined_path, index_col=0, parse_dates=True)
        else:
            # Build combined from what we have
            frames = [df.assign(symbol=s) for s, df in feature_rows.items()]
            feat_all = pd.concat(frames) if frames else pd.DataFrame()

        if feat_all.empty:
            logger.error("No data available for retraining.")
        else:
            train_model(feat_all, outputs_dir=OUTPUTS_DIR)

    xgb_m, lgb_m, le, prep, feat_cols = _load_artefacts()

    # ── STEP 6 & 7: Predictions + signal generation ────────────────────────
    _step(6, "Generate predictions and signals")
    all_signals: List[dict] = []
    skipped = 0

    for sym, latest_row in feature_rows.items():
        try:
            probs = _ensemble_proba(xgb_m, lgb_m, prep, feat_cols, latest_row)

            # Build a row dict with OHLCV and pattern meta restored
            proc_df = processed.get(sym)
            if proc_df is not None and not proc_df.empty:
                last_price_row = proc_df.sort_index().iloc[-1].to_dict()
            else:
                last_price_row = {}

            feat_dict = latest_row.iloc[0].to_dict()
            feat_dict.update({k: v for k, v in last_price_row.items()
                              if k in ("Open", "High", "Low", "Close", "Volume")})

            sig = generate_signal(
                symbol       = sym,
                date         = run_date,
                features_row = feat_dict,
                model_probs  = probs,
            )
            all_signals.append(sig)

        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] Signal generation failed: %s", sym, exc)
            skipped += 1

    # ── STEP 8: Generate daily report ──────────────────────────────────────
    _step(8, "Apply quality filters and save daily report")
    valid_signals = generate_daily_report(
        date        = run_date,
        all_signals = all_signals,
        top_n       = 20,
        reports_dir = REPORTS_DIR,
    )

    # ── Compute summary counts ──────────────────────────────────────────────
    rejected      = [s for s in all_signals if s.get("reject_signal")]
    non_hold      = [s for s in all_signals if not s.get("reject_signal")
                     and s.get("signal") != "HOLD"]
    buy_sigs      = [s for s in non_hold if s.get("signal") == "BUY"]
    sell_sigs     = [s for s in non_hold if s.get("signal") == "SELL"]
    hold_sigs     = [s for s in all_signals if not s.get("reject_signal")
                     and s.get("signal") == "HOLD"]
    high_conf     = [s for s in valid_signals if s.get("confidence_score") == "HIGH"]
    med_conf      = [s for s in valid_signals if s.get("confidence_score") == "MEDIUM"]
    low_conf      = [s for s in valid_signals if s.get("confidence_score") == "LOW"]

    elapsed = time.time() - t0

    summary = {
        "date"             : run_date,
        "stocks_analysed"  : len(feature_rows),
        "patterns_detected": pattern_count,
        "signals_generated": len(all_signals),
        "signals_rejected" : len(rejected),
        "valid_signals"    : len(valid_signals),
        "high_confidence"  : len(high_conf),
        "medium_confidence": len(med_conf),
        "low_confidence"   : len(low_conf),
        "buy_signals"      : len(buy_sigs),
        "sell_signals"     : len(sell_sigs),
        "hold_signals"     : len(hold_sigs),
        "elapsed_seconds"  : round(elapsed, 1),
    }

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Console report printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_eod_report(summary: dict) -> None:
    date      = summary["date"]
    csv_path  = REPORTS_DIR / f"{date}_signals.csv"
    txt_path  = REPORTS_DIR / f"{date}_signals.txt"

    print()
    print("=" * 54)
    print("  NIFTY 500 TRADING MODEL -- EOD REPORT")
    print(f"  Date: {date}")
    print("=" * 54)
    print(f"  Stocks analysed     : {summary['stocks_analysed']}")
    print(f"  Patterns detected   : {summary['patterns_detected']}")
    print(f"  Signals generated   : {summary['signals_generated']}")
    print(f"  Signals rejected    : {summary['signals_rejected']}")
    print(f"  Valid signals       : {summary['valid_signals']}")
    print()
    print(f"  HIGH confidence     : {summary['high_confidence']}")
    print(f"  MEDIUM confidence   : {summary['medium_confidence']}")
    print(f"  LOW confidence      : {summary['low_confidence']}")
    print()
    print(f"  BUY signals         : {summary['buy_signals']}")
    print(f"  SELL signals        : {summary['sell_signals']}")
    print(f"  HOLD (neutral)      : {summary['hold_signals']}")
    print()
    print(f"  Report saved to     : {csv_path}")
    if txt_path.exists():
        print(f"  Text report         : {txt_path}")
    print(f"  Pipeline time       : {summary['elapsed_seconds']}s")
    print("=" * 54)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Nifty 500 EOD trading pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/run_pipeline.py
  python src/run_pipeline.py --date 2024-01-15
  python src/run_pipeline.py --date 2024-01-15 --retrain
  python src/run_pipeline.py --backtest
  python src/run_pipeline.py --date 2024-01-15 --symbols RELIANCE.NS TCS.NS
        """,
    )
    p.add_argument("--date",       default=str(Date.today()),
                   help="Signal date YYYY-MM-DD (default: today)")
    p.add_argument("--retrain",    action="store_true",
                   help="Retrain models before generating signals")
    p.add_argument("--backtest",   action="store_true",
                   help="Run backtest on all saved signal reports instead of EOD pipeline")
    p.add_argument("--symbols",    nargs="*", metavar="SYM",
                   help="Restrict to specific symbols (default: full Nifty 500)")
    p.add_argument("--skip-fetch", action="store_true",
                   help="Skip yfinance download, use existing data/raw CSVs")
    p.add_argument("--capital",    type=float, default=1_000_000.0,
                   help="Initial capital for backtest in INR (default: 1000000)")
    p.add_argument("--raw-dir",    default="data/raw")
    p.add_argument("--reports-dir",default="outputs/reports")
    return p


if __name__ == "__main__":
    # Ensure outputs dir exists before FileHandler is created
    Path("outputs").mkdir(parents=True, exist_ok=True)

    args = _build_parser().parse_args()

    # ── Backtest mode ────────────────────────────────────────────────────────
    if args.backtest:
        logger.info("=== BACKTEST MODE ===")
        price_data = load_all_data(
            raw_dir=Path(args.raw_dir), min_trading_days=1
        )
        sig_df = load_all_signals(Path(args.reports_dir))
        if sig_df.empty:
            logger.error("No signal reports found in %s. Run EOD pipeline first.",
                         args.reports_dir)
            sys.exit(1)
        run_backtest(
            signals_path    = sig_df,
            price_data_dict = price_data,
            initial_capital = args.capital,
            outputs_dir     = OUTPUTS_DIR,
        )
        sys.exit(0)

    # ── EOD pipeline mode ────────────────────────────────────────────────────
    logger.info("=== EOD PIPELINE: date=%s | retrain=%s ===", args.date, args.retrain)
    summary = run_eod_pipeline(
        run_date   = args.date,
        retrain    = args.retrain,
        symbols    = args.symbols,
        skip_fetch = args.skip_fetch,
    )
    _print_eod_report(summary)
