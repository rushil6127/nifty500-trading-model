"""
backtest.py
-----------
Historical backtesting engine for Nifty 500 trade signals.

Public API
----------
run_backtest(signals_path, price_data_dict) -> dict
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")

# ─────────────────────────────────────────────────────────────────────────────
# Indian market cost constants
# ─────────────────────────────────────────────────────────────────────────────
BROKERAGE_PCT  = 0.001   # 0.1% per side
STT_PCT        = 0.001   # 0.1% on delivery buy + sell
GST_ON_BROK    = 0.18    # 18% GST on brokerage amount
CIRCUIT_LIMIT  = 0.10    # ±10% daily circuit (conservative)

# ─────────────────────────────────────────────────────────────────────────────
# NSE Holiday calendar 2023-2026 (fixed national + exchange holidays)
# ─────────────────────────────────────────────────────────────────────────────

NSE_HOLIDAYS = set(pd.to_datetime([
    # 2023
    "2023-01-26","2023-03-07","2023-03-30","2023-04-04","2023-04-07",
    "2023-04-14","2023-05-01","2023-06-28","2023-08-15","2023-09-19",
    "2023-10-02","2023-10-24","2023-11-14","2023-11-27","2023-12-25",
    # 2024
    "2024-01-22","2024-01-26","2024-03-25","2024-03-29","2024-04-11",
    "2024-04-14","2024-04-17","2024-04-21","2024-05-01","2024-05-23",
    "2024-06-17","2024-07-17","2024-08-15","2024-10-02","2024-10-12",
    "2024-11-01","2024-11-15","2024-12-25",
    # 2025
    "2025-02-26","2025-03-14","2025-03-31","2025-04-10","2025-04-14",
    "2025-04-18","2025-05-01","2025-08-15","2025-08-27","2025-10-02",
    "2025-10-02","2025-10-21","2025-10-22","2025-11-05","2025-12-25",
    # 2026 (approximate)
    "2026-01-26","2026-03-03","2026-03-20","2026-04-02","2026-04-06",
    "2026-04-14","2026-05-01","2026-08-15","2026-10-02","2026-11-11","2026-12-25",
]))


def _is_trading_day(dt: pd.Timestamp) -> bool:
    return dt.weekday() < 5 and dt not in NSE_HOLIDAYS


def _next_trading_day(dt: pd.Timestamp) -> pd.Timestamp:
    nxt = dt + pd.Timedelta(days=1)
    while not _is_trading_day(nxt):
        nxt += pd.Timedelta(days=1)
    return nxt


# ─────────────────────────────────────────────────────────────────────────────
# Transaction cost calculator
# ─────────────────────────────────────────────────────────────────────────────

def _transaction_cost(value: float) -> float:
    """Total round-trip cost for a trade of given notional value (INR)."""
    brok      = value * BROKERAGE_PCT * 2         # both sides
    stt       = value * STT_PCT  * 2              # buy + sell
    gst       = brok * GST_ON_BROK
    return brok + stt + gst


# ─────────────────────────────────────────────────────────────────────────────
# Position sizer
# ─────────────────────────────────────────────────────────────────────────────

def _position_size(portfolio_val: float, entry: float, stoploss: float) -> int:
    """Shares to buy so that 1% of portfolio is at risk. Returns integer shares."""
    risk_per_share = abs(entry - stoploss)
    if risk_per_share <= 0:
        return 0
    capital_at_risk = portfolio_val * 0.01
    return max(1, int(capital_at_risk / risk_per_share))


# ─────────────────────────────────────────────────────────────────────────────
# Circuit-limit guard
# ─────────────────────────────────────────────────────────────────────────────

def _hit_circuit(price_df: pd.DataFrame, date: pd.Timestamp) -> bool:
    """True if the stock's daily range on `date` exceeds circuit limit."""
    if date not in price_df.index:
        return False
    row = price_df.loc[date]
    prev_dates = price_df.index[price_df.index < date]
    if len(prev_dates) == 0:
        return False
    prev_close = float(price_df.loc[prev_dates[-1], "Close"])
    if prev_close <= 0:
        return False
    high_move = (float(row["High"]) - prev_close) / prev_close
    low_move  = (prev_close - float(row["Low"]))  / prev_close
    return max(high_move, low_move) >= CIRCUIT_LIMIT


# ─────────────────────────────────────────────────────────────────────────────
# Single trade simulator
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_trade(
    signal:     dict,
    price_df:   pd.DataFrame,
    hold_days:  int = 5,
) -> Optional[dict]:
    """
    Simulate one trade from signal entry to exit.

    Returns a trade-result dict or None if trade cannot be entered.
    """
    sig_date = pd.Timestamp(signal["date"])
    entry_date = _next_trading_day(sig_date)

    if entry_date not in price_df.index:
        return None
    if _hit_circuit(price_df, entry_date):
        return None

    entry_row   = price_df.loc[entry_date]
    entry_price = float(entry_row["Open"])
    stoploss    = float(signal["stoploss"])
    direction   = signal["signal"]   # BUY or SELL
    target      = entry_price + 2 * abs(entry_price - stoploss) * (1 if direction == "BUY" else -1)

    if entry_price <= 0 or abs(entry_price - stoploss) < 0.01:
        return None

    # Walk forward up to hold_days trading days
    future_dates = [
        d for d in price_df.index if entry_date < d
        and _is_trading_day(d)
    ][:hold_days]

    exit_price  = entry_price
    exit_date   = entry_date
    exit_reason = "HOLD_EXPIRED"

    for d in future_dates:
        row = price_df.loc[d]
        h, l, c = float(row["High"]), float(row["Low"]), float(row["Close"])

        if direction == "BUY":
            if l <= stoploss:
                exit_price  = stoploss
                exit_date   = d
                exit_reason = "STOPLOSS"
                break
            if h >= target:
                exit_price  = target
                exit_date   = d
                exit_reason = "TARGET"
                break
        else:  # SELL
            if h >= stoploss:
                exit_price  = stoploss
                exit_date   = d
                exit_reason = "STOPLOSS"
                break
            if l <= target:
                exit_price  = target
                exit_date   = d
                exit_reason = "TARGET"
                break

        exit_price = c
        exit_date  = d

    # PnL
    if direction == "BUY":
        raw_return = (exit_price - entry_price) / entry_price
    else:
        raw_return = (entry_price - exit_price) / entry_price

    # Transaction costs as % of entry value
    cost_pct = _transaction_cost(entry_price) / entry_price
    net_return = raw_return - cost_pct

    return {
        "symbol":           signal["symbol"],
        "signal_date":      str(sig_date.date()),
        "entry_date":       str(entry_date.date()),
        "exit_date":        str(exit_date.date()),
        "direction":        direction,
        "entry_price":      round(entry_price, 2),
        "exit_price":       round(exit_price, 2),
        "stoploss":         round(stoploss, 2),
        "target":           round(target, 2),
        "exit_reason":      exit_reason,
        "raw_return_pct":   round(raw_return * 100, 3),
        "net_return_pct":   round(net_return * 100, 3),
        "win":              net_return > 0,
        "pattern_detected": signal.get("pattern_detected", "none"),
        "confidence_score": signal.get("confidence_score", "LOW"),
        "confidence_value": signal.get("confidence_value", 0.0),
        "pattern_strength": signal.get("pattern_strength", 0.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    trades: List[dict],
    equity_curve: List[float],
) -> dict:
    if not trades:
        return {"error": "No trades executed"}

    returns     = [t["net_return_pct"] / 100 for t in trades]
    wins        = [r for r in returns if r > 0]
    losses      = [r for r in returns if r <= 0]

    win_rate    = len(wins) / len(returns) * 100
    avg_win     = np.mean(wins)  * 100 if wins   else 0.0
    avg_loss    = np.mean(losses)* 100 if losses else 0.0
    gross_wins  = sum(wins)
    gross_losses= abs(sum(losses))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float("inf")

    # Max drawdown from equity curve
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min()) * 100

    # Sharpe (annualised, daily returns proxy)
    r_arr = np.array(returns)
    sharpe = (r_arr.mean() / (r_arr.std() + 1e-9)) * math.sqrt(252) if len(r_arr) > 1 else 0.0

    # CAGR
    if len(equity_curve) > 1:
        years = len(equity_curve) / 252
        cagr  = ((equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    else:
        cagr = 0.0

    # Pattern-wise win rates
    pat_results: Dict[str, List[bool]] = {}
    for t in trades:
        p = t.get("pattern_detected", "none")
        pat_results.setdefault(p, []).append(t["win"])
    pattern_winrates = {
        p: round(sum(v) / len(v) * 100, 1)
        for p, v in pat_results.items() if len(v) >= 2
    }
    pattern_winrates = dict(sorted(pattern_winrates.items(), key=lambda x: -x[1]))

    # Confidence-tier win rates
    conf_results: Dict[str, List[bool]] = {}
    for t in trades:
        c = t.get("confidence_score", "LOW")
        conf_results.setdefault(c, []).append(t["win"])
    conf_winrates = {
        c: round(sum(v) / len(v) * 100, 1)
        for c, v in conf_results.items()
    }

    return {
        "total_trades":     len(trades),
        "wins":             len(wins),
        "losses":           len(losses),
        "win_rate_pct":     round(win_rate, 2),
        "avg_win_pct":      round(avg_win, 3),
        "avg_loss_pct":     round(avg_loss, 3),
        "profit_factor":    round(profit_factor, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio":     round(sharpe, 3),
        "cagr_pct":         round(cagr, 2),
        "final_equity":     round(equity_curve[-1], 2),
        "total_return_pct": round((equity_curve[-1] / equity_curve[0] - 1) * 100, 2),
        "pattern_winrates": pattern_winrates,
        "confidence_winrates": conf_winrates,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_equity_curve(equity: List[float], dates: List[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates, equity, linewidth=1.5, color="#1976D2")
    ax.fill_between(range(len(equity)), equity[0], equity,
                    where=[e >= equity[0] for e in equity],
                    alpha=0.15, color="green")
    ax.fill_between(range(len(equity)), equity[0], equity,
                    where=[e < equity[0] for e in equity],
                    alpha=0.15, color="red")
    ax.axhline(equity[0], color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Portfolio Equity Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Portfolio Value (INR)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Equity curve saved -> %s", out_path)


def _plot_pattern_winrates(pattern_winrates: dict, out_path: Path) -> None:
    if not pattern_winrates:
        return
    labels = list(pattern_winrates.keys())
    values = list(pattern_winrates.values())
    colors = ["#43A047" if v >= 50 else "#E53935" for v in values]

    fig, ax = plt.subplots(figsize=(11, max(4, len(labels) * 0.45)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white")
    ax.axvline(50, color="black", linestyle="--", linewidth=0.9, label="50% break-even")
    ax.set_xlabel("Win Rate (%)")
    ax.set_title("Pattern Win Rates", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 105)
    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Pattern win-rate chart saved -> %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest runner
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    signals_path:    Path,
    price_data_dict: Dict[str, pd.DataFrame],
    initial_capital: float = 1_000_000.0,   # 10 lakh INR
    max_positions:   int   = 10,
    hold_days:       int   = 5,
    outputs_dir:     Path  = OUTPUTS_DIR,
) -> dict:
    """
    Execute the full backtesting simulation.

    Parameters
    ----------
    signals_path     : Path to a signals CSV (columns: symbol, date, signal,
                       stoploss, confidence_score, confidence_value,
                       pattern_detected, pattern_strength).
                       Typically outputs/reports/{date}_signals.csv or a
                       concatenation of multiple daily reports.
    price_data_dict  : {symbol: OHLCV DataFrame} from data_ingestion.load_all_data().
    initial_capital  : Starting portfolio value in INR (default 10 lakh).
    max_positions    : Maximum concurrent open positions (default 10).
    hold_days        : Maximum days to hold a position (default 5).
    outputs_dir      : Directory for output files.

    Returns
    -------
    Metrics summary dict (also saved to outputs/backtest_summary.json).
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load signals
    if isinstance(signals_path, (str, Path)):
        sig_df = pd.read_csv(signals_path, parse_dates=["date"])
    else:
        sig_df = signals_path  # accept DataFrame directly

    # Filter only actionable signals
    sig_df = sig_df[sig_df["signal"].isin(["BUY", "SELL"])].copy()
    sig_df = sig_df.sort_values("date").reset_index(drop=True)
    logger.info("Loaded %d actionable signals from %s", len(sig_df), signals_path)

    if sig_df.empty:
        logger.warning("No BUY/SELL signals found. Nothing to backtest.")
        return {}

    # ── 2. Portfolio state
    portfolio_value = initial_capital
    open_positions: Dict[str, dict] = {}   # symbol -> trade info
    equity_curve: List[float] = [portfolio_value]
    equity_dates: List[str]   = ["start"]
    all_trades:   List[dict]  = []

    # ── 3. Iterate signals chronologically
    for _, row in sig_df.iterrows():
        sym    = str(row["symbol"])
        signal = row.to_dict()

        # Ensure required keys exist with defaults
        signal.setdefault("stoploss",         float(row.get("entry_price", 0)) * 0.98)
        signal.setdefault("confidence_score", "LOW")
        signal.setdefault("confidence_value", 0.0)
        signal.setdefault("pattern_detected", "none")
        signal.setdefault("pattern_strength", 0.0)

        # Skip if already holding this stock
        if sym in open_positions:
            continue

        # Skip if max positions reached
        if len(open_positions) >= max_positions:
            continue

        # Get price data for this symbol
        price_df = price_data_dict.get(sym)
        if price_df is None:
            continue

        # Position sizing
        entry_date_t = _next_trading_day(pd.Timestamp(signal["date"]))
        if entry_date_t not in price_df.index:
            continue
        entry_price = float(price_df.loc[entry_date_t, "Open"])
        stoploss    = float(signal["stoploss"])

        shares = _position_size(portfolio_value, entry_price, stoploss)
        if shares < 1:
            continue

        notional = shares * entry_price
        if notional > portfolio_value * 0.30:   # max 30% in one trade
            shares = max(1, int(portfolio_value * 0.30 / entry_price))
            notional = shares * entry_price

        # Reserve capital
        portfolio_value -= notional
        open_positions[sym] = {
            "signal":   signal,
            "shares":   shares,
            "notional": notional,
        }

        # Simulate trade
        result = _simulate_trade(signal, price_df, hold_days)
        if result is None:
            portfolio_value += notional   # return reserved capital
            del open_positions[sym]
            continue

        # Apply trade result
        net_ret      = result["net_return_pct"] / 100
        pnl          = notional * net_ret
        portfolio_value += notional + pnl

        result["shares"]        = shares
        result["notional"]      = round(notional, 2)
        result["pnl"]           = round(pnl, 2)
        result["portfolio_after"] = round(portfolio_value, 2)

        all_trades.append(result)
        equity_curve.append(portfolio_value)
        equity_dates.append(result["exit_date"])

        del open_positions[sym]

    # Add remaining open capital back (cash portion)
    logger.info("Simulation complete: %d trades executed", len(all_trades))

    # ── 4. Compute metrics
    metrics = _compute_metrics(all_trades, equity_curve)
    metrics["initial_capital"]  = initial_capital
    metrics["signals_evaluated"] = len(sig_df)

    # ── 5. Save trade log
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_csv = outputs_dir / "backtest_trades.csv"
        trades_df.to_csv(trades_csv, index=False)
        logger.info("Trade log saved -> %s (%d rows)", trades_csv, len(trades_df))

    # ── 6. Save summary JSON
    summary_path = outputs_dir / "backtest_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    logger.info("Summary saved -> %s", summary_path)

    # ── 7. Plots
    _plot_equity_curve(equity_curve, equity_dates, outputs_dir / "equity_curve.png")
    _plot_pattern_winrates(metrics.get("pattern_winrates", {}),
                           outputs_dir / "pattern_winrates.png")

    # ── 8. Print summary
    _print_summary(metrics)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(m: dict) -> None:
    print("\n" + "=" * 58)
    print("  BACKTEST SUMMARY")
    print("=" * 58)
    print(f"  Initial capital      : INR {m.get('initial_capital', 0):>12,.0f}")
    print(f"  Final equity         : INR {m.get('final_equity', 0):>12,.2f}")
    print(f"  Total return         : {m.get('total_return_pct', 0):>+.2f}%")
    print(f"  CAGR                 : {m.get('cagr_pct', 0):>+.2f}%")
    print(f"  Sharpe ratio         : {m.get('sharpe_ratio', 0):>.3f}")
    print(f"  Max drawdown         : {m.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Profit factor        : {m.get('profit_factor', 0):.3f}")
    print("-" * 58)
    print(f"  Total trades         : {m.get('total_trades', 0)}")
    print(f"  Win rate             : {m.get('win_rate_pct', 0):.1f}%  "
          f"({m.get('wins',0)}W / {m.get('losses',0)}L)")
    print(f"  Avg win              : +{m.get('avg_win_pct', 0):.3f}%")
    print(f"  Avg loss             : {m.get('avg_loss_pct', 0):.3f}%")
    print("-" * 58)
    print("  Pattern win rates:")
    for pat, wr in list(m.get("pattern_winrates", {}).items())[:10]:
        bar = "#" * int(wr / 5)
        print(f"    {pat:<26} {wr:>5.1f}%  {bar}")
    print("-" * 58)
    print("  Confidence-tier win rates:")
    for tier, wr in m.get("confidence_winrates", {}).items():
        print(f"    {tier:<10} {wr:.1f}%")
    print("=" * 58 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for loading concatenated signal files
# ─────────────────────────────────────────────────────────────────────────────

def load_all_signals(reports_dir: Path = Path("outputs/reports")) -> pd.DataFrame:
    """
    Concatenate all per-day *_signals.csv files in reports_dir
    into a single DataFrame for backtesting.
    """
    frames = []
    for f in sorted(reports_dir.glob("*_signals.csv")):
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception as exc:
            logger.warning("Could not read %s: %s", f.name, exc)
    if not frames:
        logger.warning("No signal CSV files found in %s", reports_dir)
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"])
    logger.info("Loaded %d signal rows from %d files", len(combined), len(frames))
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Run Nifty 500 backtest.")
    parser.add_argument("--signals",      default=None,
                        help="Path to signals CSV (default: auto-load from outputs/reports/)")
    parser.add_argument("--raw-dir",      default="data/raw")
    parser.add_argument("--outputs-dir",  default="outputs")
    parser.add_argument("--capital",      type=float, default=1_000_000.0,
                        help="Initial capital in INR (default: 1000000)")
    parser.add_argument("--max-pos",      type=int,   default=10)
    parser.add_argument("--hold-days",    type=int,   default=5)
    args = parser.parse_args()

    from src.data_ingestion import load_all_data  # noqa: E402

    price_data = load_all_data(
        raw_dir=Path(args.raw_dir),
        min_trading_days=1,
    )

    if args.signals:
        sig_source = Path(args.signals)
    else:
        sig_source = load_all_signals(Path("outputs/reports"))
        if isinstance(sig_source, pd.DataFrame) and sig_source.empty:
            logger.error("No signals found. Run src/signals.py first.")
            sys.exit(1)

    run_backtest(
        signals_path    = sig_source,
        price_data_dict = price_data,
        initial_capital = args.capital,
        max_positions   = args.max_pos,
        hold_days       = args.hold_days,
        outputs_dir     = Path(args.outputs_dir),
    )
