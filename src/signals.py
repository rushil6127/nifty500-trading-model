"""
signals.py
----------
Probabilistic trade signal generation engine for the Nifty 500 trading system.

Philosophy: Never "BUY because pattern formed."
Output a full probabilistic assessment combining model output,
context, quality flags, and rejection logic.

Public API
----------
generate_signal(symbol, date, features_row, model_probs) -> dict
generate_daily_report(date, all_signals)                 -> list[dict]
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("outputs/reports")

# ─────────────────────────────────────────────────────────────────────────────
# Threshold constants
# ─────────────────────────────────────────────────────────────────────────────

CANDLE_TOO_SMALL_PCT = 0.01   # total_range < 1% of close
CANDLE_TOO_LARGE_PCT = 0.10   # total_range > 10% of close
SIDEWAYS_TREND_STR   = 0.33   # |trend_strength| < this → sideways
STRONG_TREND_STR     = 0.67   # |trend_strength| >= this → strong trend
VOLUME_SPIKE_RATIO   = 1.5    # volume_ratio threshold for "high_volume"
NEAR_SR_DISTANCE     = 0.02   # within 2% of rolling high/low = near S/R
RSI_OVERSOLD         = 35
RSI_OVERBOUGHT       = 65
MIN_CONFIDENCE       = 0.50   # reject below this
HIGH_CONFIDENCE      = 0.70
MED_CONFIDENCE       = 0.55


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe feature getter
# ─────────────────────────────────────────────────────────────────────────────

def _f(row: dict, key: str, default=0.0):
    """Safely retrieve a feature value from the row dict."""
    val = row.get(key, default)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Context extraction
# ─────────────────────────────────────────────────────────────────────────────

def _build_context(row: dict) -> dict:
    trend_str   = float(_f(row, "trend_strength", 0.0))
    rsi         = float(_f(row, "rsi_14", 50.0))
    vol_ratio   = float(_f(row, "volume_ratio", 1.0))
    atr_ratio   = float(_f(row, "atr_ratio", 0.0))
    vol_exp     = int(_f(row, "volatility_expansion", 0))
    vol_con     = int(_f(row, "volatility_contraction", 0))
    gap_up      = int(_f(row, "gap_up", 0))
    gap_down    = int(_f(row, "gap_down", 0))
    dist_res    = float(_f(row, "distance_from_resistance", 1.0))
    dist_sup    = float(_f(row, "distance_from_support", 1.0))
    e20_above_50  = float(_f(row, "ema_20", 0)) > float(_f(row, "ema_50", 0))
    e50_above_200 = float(_f(row, "ema_50", 0)) > float(_f(row, "ema_200", 0))

    # Prior trend
    if trend_str >= SIDEWAYS_TREND_STR:
        prior_trend = "UPTREND"
    elif trend_str <= -SIDEWAYS_TREND_STR:
        prior_trend = "DOWNTREND"
    else:
        prior_trend = "SIDEWAYS"

    # RSI zone
    if rsi <= RSI_OVERSOLD:
        rsi_zone = "OVERSOLD"
    elif rsi >= RSI_OVERBOUGHT:
        rsi_zone = "OVERBOUGHT"
    else:
        rsi_zone = "NEUTRAL"

    # Volatility state
    if vol_exp:
        vol_state = "EXPANDING"
    elif vol_con:
        vol_state = "CONTRACTING"
    else:
        vol_state = "NEUTRAL"

    # EMA alignment
    if e20_above_50 and e50_above_200:
        ema_align = "BULLISH"
    elif not e20_above_50 and not e50_above_200:
        ema_align = "BEARISH"
    else:
        ema_align = "MIXED"

    return {
        "prior_trend":           prior_trend,
        "rsi_zone":              rsi_zone,
        "volume_confirmation":   vol_ratio >= VOLUME_SPIKE_RATIO,
        "volatility_state":      vol_state,
        "near_support":          abs(dist_sup) <= NEAR_SR_DISTANCE,
        "near_resistance":       abs(dist_res) <= NEAR_SR_DISTANCE,
        "gap_present":           bool(gap_up or gap_down),
        "ema_alignment":         ema_align,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quality flags
# ─────────────────────────────────────────────────────────────────────────────

def _build_quality_flags(row: dict, context: dict) -> dict:
    close       = float(_f(row, "Close", 1.0))
    total_range = float(_f(row, "total_range", 0.0))
    vol_ratio   = float(_f(row, "volume_ratio", 1.0))
    trend_str   = abs(float(_f(row, "trend_strength", 0.0)))
    mom_score   = float(_f(row, "momentum_score", 0.0))
    macd_cross  = int(_f(row, "macd_crossover", 0))
    near_sr     = context["near_support"] or context["near_resistance"]
    sideways    = context["prior_trend"] == "SIDEWAYS"
    vol_spike   = int(_f(row, "volume_spike", 0))

    range_pct = total_range / close if close > 0 else 0

    return {
        "high_volume":                   vol_ratio >= VOLUME_SPIKE_RATIO,
        "strong_trend":                  trend_str >= STRONG_TREND_STR,
        "momentum_confirmation":         abs(mom_score) >= 0.33 or macd_cross != 0,
        "support_resistance_reaction":   near_sr,
        "pattern_in_sideways_market":    sideways,
        "candle_too_small":              range_pct < CANDLE_TOO_SMALL_PCT,
        "candle_too_large":              range_pct > CANDLE_TOO_LARGE_PCT,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Derived probabilities
# ─────────────────────────────────────────────────────────────────────────────

def _derived_probs(row: dict, context: dict) -> tuple[float, float]:
    """
    continuation_probability = trend alignment strength × normalised momentum
    reversal_probability     = RSI extreme × pattern_strength × volume_spike
    Both clamped to [0, 1].
    """
    trend_str    = abs(float(_f(row, "trend_strength", 0.0)))
    mom_score    = float(_f(row, "momentum_score", 0.0))       # -1..+1
    rsi          = float(_f(row, "rsi_14", 50.0))
    pat_strength = float(_f(row, "pattern_strength", 0.0))
    vol_spike    = int(_f(row, "volume_spike", 0))

    # Continuation: trend strength × |momentum|
    continuation = min(trend_str * (abs(mom_score) + 0.5), 1.0)

    # RSI extremity (0 at 50, 1 at 0 or 100)
    rsi_extremity = abs(rsi - 50) / 50.0
    reversal = min(rsi_extremity * pat_strength * (1.0 + vol_spike * 0.5), 1.0)

    return round(continuation, 4), round(reversal, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score_confidence(
    base: float,
    flags: dict,
) -> tuple[float, str]:
    """
    Apply quality flag adjustments to the base confidence score.
    Returns (confidence_value, confidence_label).
    """
    score = base

    # Positive boosts (+0.05 each)
    boost_flags = ["high_volume", "strong_trend",
                   "momentum_confirmation", "support_resistance_reaction"]
    for f in boost_flags:
        if flags.get(f):
            score += 0.05

    # Penalties
    if flags.get("pattern_in_sideways_market"):
        score -= 0.10
    if flags.get("candle_too_small") or flags.get("candle_too_large"):
        score -= 0.15

    score = round(min(max(score, 0.0), 1.0), 4)

    if score >= HIGH_CONFIDENCE:
        label = "HIGH"
    elif score >= MED_CONFIDENCE:
        label = "MEDIUM"
    else:
        label = "LOW"

    return score, label


# ─────────────────────────────────────────────────────────────────────────────
# Signal + strength classification
# ─────────────────────────────────────────────────────────────────────────────

def _classify_signal(
    bull_p: float,
    bear_p: float,
    conf_val: float,
    pattern_dir: str,
) -> tuple[str, str]:
    """
    Determine BUY / SELL / HOLD and STRONG / MODERATE / WEAK.
    Model probability is primary; pattern direction acts as a confirming tiebreaker.
    """
    margin = bull_p - bear_p   # positive → lean bullish

    if bull_p >= 0.45 and (margin > 0.05 or pattern_dir == "bullish"):
        signal = "BUY"
    elif bear_p >= 0.45 and (margin < -0.05 or pattern_dir == "bearish"):
        signal = "SELL"
    else:
        signal = "HOLD"

    if conf_val >= HIGH_CONFIDENCE:
        strength = "STRONG"
    elif conf_val >= MED_CONFIDENCE:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    return signal, strength


# ─────────────────────────────────────────────────────────────────────────────
# Stop-loss calculation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stoploss(row: dict, signal: str) -> tuple[float, float, str]:
    """
    Compute stoploss price, risk percentage, and stoploss method.
    """
    entry_price = float(_f(row, "Close", 0.0))
    atr_14 = float(_f(row, "atr_14", entry_price * 0.01))
    lower_wick = float(_f(row, "lower_wick", 0.0))
    upper_wick = float(_f(row, "upper_wick", 0.0))
    pattern_candle_count = int(_f(row, "pattern_candle_count", 0))

    if signal not in ("BUY", "SELL"):
        return round(entry_price, 2), 0.0, "none"

    method = "pattern"

    if pattern_candle_count >= 2:
        # Multi-candle pattern
        # The stoploss must span the full pattern range.
        if "pattern_low" in row and "pattern_high" in row:
            pattern_low = float(_f(row, "pattern_low", 0.0))
            pattern_high = float(_f(row, "pattern_high", 0.0))
            if signal == "BUY":
                stoploss = pattern_low - (0.1 * atr_14)
            else:
                stoploss = pattern_high + (0.1 * atr_14)
        else:
            # Fall back to ATR stoploss
            if signal == "BUY":
                stoploss = entry_price - (1.5 * atr_14)
            else:
                stoploss = entry_price + (1.5 * atr_14)
            method = "atr_fallback"
    else:
        # Single candle or no pattern
        if signal == "BUY":
            pattern_stoploss = entry_price - lower_wick - (0.1 * atr_14)
            atr_stoploss     = entry_price - (1.5 * atr_14)
            if pattern_stoploss > 0 and abs(entry_price - pattern_stoploss) < (2 * atr_14):
                stoploss = pattern_stoploss
            else:
                stoploss = atr_stoploss
                method = "atr_fallback"
        else: # SELL
            pattern_stoploss = entry_price + upper_wick + (0.1 * atr_14)
            atr_stoploss     = entry_price + (1.5 * atr_14)
            if pattern_stoploss > 0 and abs(pattern_stoploss - entry_price) < (2 * atr_14):
                stoploss = pattern_stoploss
            else:
                stoploss = atr_stoploss
                method = "atr_fallback"

    risk_pct = abs(entry_price - stoploss) / entry_price if entry_price > 0 else 0.0
    return round(stoploss, 2), round(risk_pct * 100, 3), method


# ─────────────────────────────────────────────────────────────────────────────
# Rejection logic
# ─────────────────────────────────────────────────────────────────────────────

def _check_rejection(flags: dict, conf_val: float, context: dict) -> tuple[bool, str]:
    reasons = []

    if flags.get("candle_too_small"):
        reasons.append("candle range < 1% of close (too small)")
    if flags.get("candle_too_large"):
        reasons.append("candle range > 10% of close (likely news / circuit)")
    if conf_val < MIN_CONFIDENCE:
        reasons.append(f"confidence {conf_val:.2f} < minimum {MIN_CONFIDENCE}")
    if flags.get("pattern_in_sideways_market") and not flags.get("high_volume"):
        reasons.append("pattern in sideways market without volume confirmation")

    return (bool(reasons), "; ".join(reasons))


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def generate_signal(
    symbol:       str,
    date:         str,
    features_row: dict,
    model_probs:  Dict[str, float],
) -> dict:
    """
    Generate a full probabilistic trade signal for one (symbol, date) pair.

    Parameters
    ----------
    symbol       : NSE ticker, e.g. "TCS.NS"
    date         : Signal date string, e.g. "2026-05-23"
    features_row : Full feature dict for this row (from feature_engineering output).
    model_probs  : {"bullish_prob": float, "bearish_prob": float, "neutral_prob": float}
                   These come directly from model.predict_single() ensemble probabilities.

    Returns
    -------
    Signal dict with all assessment fields (see module docstring).
    """
    row = features_row if isinstance(features_row, dict) else features_row.to_dict()

    # ── Probabilities from model
    bull_p = float(model_probs.get("bullish_prob",  0.333))
    bear_p = float(model_probs.get("bearish_prob",  0.333))
    neut_p = float(model_probs.get("neutral_prob",  0.333))
    # Normalise in case they don't sum to 1
    total  = bull_p + bear_p + neut_p or 1.0
    bull_p, bear_p, neut_p = bull_p/total, bear_p/total, neut_p/total

    # ── Pattern information from feature columns
    pat_name   = str(row.get("pattern_name", "none") or "none")
    pat_dir    = str(row.get("pattern_direction", "neutral") or "neutral")
    pat_str    = float(_f(row, "pattern_strength", 0.0))
    # Numeric direction stored as pattern_direction_num in features
    pat_dir_num = int(_f(row, "pattern_direction_num", 0))
    if pat_dir == "neutral" and pat_dir_num:
        pat_dir = {1: "bullish", -1: "bearish"}.get(pat_dir_num, "neutral")

    # ── Context
    context = _build_context(row)

    # ── Quality flags
    flags = _build_quality_flags(row, context)

    # ── Derived probabilities
    cont_prob, rev_prob = _derived_probs(row, context)

    # ── Base confidence = max model directional probability
    base_conf = max(bull_p, bear_p)
    conf_val, conf_label = _score_confidence(base_conf, flags)

    # ── Signal classification
    signal, strength = _classify_signal(bull_p, bear_p, conf_val, pat_dir)

    # ── Entry + stop-loss
    close     = float(_f(row, "Close", 0.0))
    sl, risk, sl_method  = _compute_stoploss(row, signal)

    # ── Rejection
    rejected, reject_reason = _check_rejection(flags, conf_val, context)

    return {
        "symbol":   symbol,
        "date":     str(date),

        "pattern_detected":  pat_name,
        "pattern_direction": pat_dir,
        "pattern_strength":  round(pat_str, 4),

        "bullish_probability": round(bull_p, 4),
        "bearish_probability": round(bear_p, 4),
        "neutral_probability": round(neut_p, 4),

        "continuation_probability": cont_prob,
        "reversal_probability":     rev_prob,

        "confidence_score": conf_label,
        "confidence_value": conf_val,

        "signal":          signal,
        "signal_strength": strength,

        "entry_price": round(close, 2),
        "stoploss":    sl,
        "risk_pct":    risk,
        "stoploss_method": sl_method,

        "context":       context,
        "quality_flags": flags,

        "reject_signal": rejected,
        "reject_reason": reject_reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Daily report
# ─────────────────────────────────────────────────────────────────────────────

_REPORT_COLS = [
    "symbol", "pattern_detected", "pattern_direction",
    "bullish_probability", "bearish_probability",
    "confidence_score", "confidence_value",
    "signal", "signal_strength",
    "entry_price", "stoploss", "risk_pct",
]

_TXT_HEADER = (
    f"{'Symbol':<14} {'Pattern':<22} {'Dir':<9} "
    f"{'Bull%':>6} {'Bear%':>6} {'Conf%':>6} "
    f"{'Conf':>6} {'Signal':<6} {'Str':<9} "
    f"{'Entry':>9} {'SL':>9} {'Risk%':>6}"
)
_TXT_SEP = "-" * len(_TXT_HEADER)


def _format_signal_row(s: dict) -> str:
    return (
        f"{s['symbol']:<14} "
        f"{s['pattern_detected']:<22} "
        f"{s['pattern_direction']:<9} "
        f"{s['bullish_probability']*100:>5.1f}% "
        f"{s['bearish_probability']*100:>5.1f}% "
        f"{s['confidence_value']*100:>5.1f}% "
        f"{s['confidence_score']:>6} "
        f"{s['signal']:<6} "
        f"{s['signal_strength']:<9} "
        f"{s['entry_price']:>9.2f} "
        f"{s['stoploss']:>9.2f} "
        f"{s['risk_pct']:>5.2f}%"
    )


def generate_daily_report(
    date:        str,
    all_signals: List[dict],
    top_n:       int = 20,
    reports_dir: Path = REPORTS_DIR,
) -> List[dict]:
    """
    Filter, rank, and persist today's trade signals.

    Parameters
    ----------
    date        : Date string for the report, e.g. "2026-05-23"
    all_signals : List of signal dicts from generate_signal() across the universe.
    top_n       : Maximum signals to include in the report (default 20).
    reports_dir : Directory in which CSV and TXT reports are saved.

    Returns
    -------
    Filtered, ranked list of up to `top_n` signal dicts.
    """
    # Filter rejected
    valid = [s for s in all_signals if not s.get("reject_signal", True)]

    # Sort by confidence descending
    valid.sort(key=lambda x: x.get("confidence_value", 0.0), reverse=True)
    top = valid[:top_n]

    logger.info(
        "generate_daily_report [%s]: %d total | %d valid | %d in report",
        date, len(all_signals), len(valid), len(top),
    )

    if not top:
        logger.warning("No valid signals for %s", date)
        return []

    # ── Save CSV
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in top:
        flat = {c: s.get(c) for c in _REPORT_COLS}
        # Add context fields
        ctx = s.get("context", {})
        flat.update({
            "prior_trend":         ctx.get("prior_trend"),
            "rsi_zone":            ctx.get("rsi_zone"),
            "volume_confirmation": ctx.get("volume_confirmation"),
            "ema_alignment":       ctx.get("ema_alignment"),
        })
        rows.append(flat)

    df_report = pd.DataFrame(rows)
    csv_path  = reports_dir / f"{date}_signals.csv"
    df_report.to_csv(csv_path, index=False)
    logger.info("CSV report -> %s", csv_path)

    # ── Save TXT
    txt_path = reports_dir / f"{date}_signals.txt"
    lines = [
        f"NIFTY 500 TRADE SIGNAL REPORT — {date}",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total signals scanned: {len(all_signals)} | Valid: {len(valid)} | Shown: {len(top)}",
        "",
        _TXT_HEADER,
        _TXT_SEP,
    ]
    for s in top:
        lines.append(_format_signal_row(s))

    lines += [
        _TXT_SEP,
        "",
        "LEGEND:",
        "  Bull%/Bear% = Model probability  |  Conf% = Confidence score",
        "  Signal: BUY / SELL / HOLD        |  Str: STRONG / MODERATE / WEAK",
        "  SL = Pattern-based stop-loss     |  Risk% = (Entry-SL)/Entry",
        "",
        "DISCLAIMER: This is a probabilistic model output for research purposes only.",
        "Not financial advice. Past patterns do not guarantee future returns.",
    ]

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("TXT report -> %s", txt_path)

    return top


# ─────────────────────────────────────────────────────────────────────────────
# Batch helper: generate signals for an entire feature DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals_batch(
    features_df:  pd.DataFrame,
    model_output: pd.DataFrame,
    date:         Optional[str] = None,
    reports_dir:  Path = REPORTS_DIR,
) -> List[dict]:
    """
    Vectorised signal generation over a combined features DataFrame.

    Parameters
    ----------
    features_df  : Combined feature DataFrame with 'symbol' column.
    model_output : DataFrame indexed like features_df with columns
                   prob_bullish, prob_neutral, prob_bearish (from predict_single loop).
    date         : If provided, filter to this date; otherwise use latest date per symbol.
    reports_dir  : Output directory.

    Returns
    -------
    List of signal dicts after daily report is generated and saved.
    """
    if date:
        ts = pd.Timestamp(date)
        features_df  = features_df[features_df.index == ts]
        model_output = model_output[model_output.index == ts] if not model_output.empty else model_output

    if features_df.empty:
        logger.warning("No rows for date=%s", date)
        return []

    signals: List[dict] = []
    for idx, feat_row in features_df.iterrows():
        sym = str(feat_row.get("symbol", "UNKNOWN"))
        row_date = str(idx.date()) if hasattr(idx, "date") else str(idx)

        # Look up model probs for this row
        if not model_output.empty and idx in model_output.index:
            mo = model_output.loc[idx]
            probs = {
                "bullish_prob": float(mo.get("prob_bullish", 1/3)),
                "bearish_prob": float(mo.get("prob_bearish", 1/3)),
                "neutral_prob": float(mo.get("prob_neutral", 1/3)),
            }
        else:
            probs = {"bullish_prob": 1/3, "bearish_prob": 1/3, "neutral_prob": 1/3}

        sig = generate_signal(sym, row_date, feat_row.to_dict(), probs)
        signals.append(sig)

    report_date = date or str(features_df.index.max().date())
    top = generate_daily_report(report_date, signals, reports_dir=reports_dir)
    return top


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Generate Nifty 500 trade signals.")
    parser.add_argument("--features-csv",  default="data/features/combined_features.csv")
    parser.add_argument("--model-dir",     default="outputs")
    parser.add_argument("--date",          default=None, help="YYYY-MM-DD (default: latest)")
    parser.add_argument("--reports-dir",   default="outputs/reports")
    parser.add_argument("--top-n",         type=int, default=20)
    args = parser.parse_args()

    from src.model import predict_single  # noqa: E402

    logger.info("Loading features from %s ...", args.features_csv)
    feat_df = pd.read_csv(args.features_csv, index_col=0, parse_dates=True)

    target_date = args.date or str(feat_df.index.max().date())
    ts = pd.Timestamp(target_date)
    day_df = feat_df[feat_df.index == ts]

    if day_df.empty:
        logger.error("No data found for date %s. Available range: %s to %s",
                     target_date, feat_df.index.min().date(), feat_df.index.max().date())
        sys.exit(1)

    logger.info("Generating signals for %s (%d rows) ...", target_date, len(day_df))

    signals = []
    for idx, row in day_df.iterrows():
        sym = str(row.get("symbol", "UNKNOWN"))
        pred = predict_single(sym, target_date, feat_df, Path(args.model_dir))
        probs = {
            "bullish_prob": pred.get("prob_bullish", 1/3),
            "bearish_prob": pred.get("prob_bearish", 1/3),
            "neutral_prob": pred.get("prob_neutral", 1/3),
        }
        sig = generate_signal(sym, target_date, row.to_dict(), probs)
        signals.append(sig)

    top = generate_daily_report(
        target_date, signals,
        top_n=args.top_n,
        reports_dir=Path(args.reports_dir),
    )

    print(f"\nTop {len(top)} signals for {target_date}:")
    for s in top:
        print(f"  {s['symbol']:<14} {s['signal']:<5} {s['confidence_score']:<7} "
              f"Bull:{s['bullish_probability']:.2f} Bear:{s['bearish_probability']:.2f} "
              f"Entry:{s['entry_price']:.2f} SL:{s['stoploss']:.2f} Risk:{s['risk_pct']:.2f}%")
