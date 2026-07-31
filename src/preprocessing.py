"""
preprocessing.py
----------------
Clean and adjust raw OHLCV DataFrames for the Nifty 500 universe.

Pipeline (applied in order)
----------------------------
1. Corporate action adjustment  – ratio-based backward price adjustment
2. Missing day handling         – NSE calendar comparison, ffill / drop
3. Outlier clipping             – flag & clip daily returns beyond ±20%
4. Candle validity filter       – drop structurally invalid candles
5. Data type enforcement        – float64 prices, int64 volume, DatetimeIndex
6. Output                       – save to data/processed/{SYMBOL}.csv

Public API
----------
preprocess_one(df, symbol, processed_dir) -> pd.DataFrame | None
preprocess_all(raw_data_dict, processed_dir, max_workers) -> dict
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
MAX_RETURN_CLIP = 0.20          # ±20% daily return threshold
MIN_RANGE_PCT   = 0.0001        # 0.01% of close = minimum candle range
MAX_GAP_FILL    = 2             # forward-fill gaps ≤ 2 consecutive days
MAX_GAP_DROP    = 3             # drop gaps > 3 consecutive days
SPLIT_RATIO_TOL = 0.01          # tolerance for detecting corporate action


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Corporate Action Adjustment
# ─────────────────────────────────────────────────────────────────────────────

def _adjust_corporate_actions(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Backward-adjust Open/High/Low/Close using the Adj_Close / Close ratio.

    yfinance returns both 'Close' (unadjusted) and an adjusted series when
    auto_adjust=False.  When auto_adjust=True the columns are already adjusted
    and the ratio is always 1 – the function is a no-op in that case.

    Detection logic
    ---------------
    If  |Adj_Close / Close - 1|  >  SPLIT_RATIO_TOL  for any row, a corporate
    action (split, dividend, rights issue) is assumed and the adjustment factor
    is applied backward across all OHLC columns.
    """
    df = df.copy()

    # Normalise column names to title-case
    df.columns = [c.strip().title().replace(" ", "_") for c in df.columns]

    # Build Adj_Close from available columns
    adj_col = None
    for candidate in ["Adj_Close", "Adj Close", "Adj_close"]:
        if candidate in df.columns:
            adj_col = candidate
            break

    if adj_col is None:
        # auto_adjust=True: Close is already adjusted; create Adj_Close alias
        df["Adj_Close"] = df["Close"]
        logger.debug("[%s] No separate Adj_Close column – treating Close as adjusted.", symbol)
        return df

    if adj_col != "Adj_Close":
        df.rename(columns={adj_col: "Adj_Close"}, inplace=True)

    # Compute adjustment ratio
    ratio = df["Adj_Close"] / df["Close"].replace(0, np.nan)

    # Check whether any meaningful adjustment is needed
    needs_adjustment = (ratio - 1).abs() > SPLIT_RATIO_TOL

    if needs_adjustment.any():
        n_events = int(needs_adjustment.sum())
        logger.info(
            "[%s] Corporate action detected on %d rows – applying backward adjustment.",
            symbol, n_events,
        )
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col] * ratio

    # Replace unadjusted Close with Adj_Close so downstream sees one price series
    df["Close"] = df["Adj_Close"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Missing Day Handling
# ─────────────────────────────────────────────────────────────────────────────

def _nse_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Return an approximate NSE trading calendar (Mon–Fri, excluding India's
    national public holidays).  A more accurate calendar can be swapped in
    via `exchange_calendars` or `trading_calendars` if installed.
    """
    # All weekdays in the range
    bdays = pd.bdate_range(start=start, end=end, freq="B")

    # Fixed national holidays (month-day tuples) that fall on weekdays
    # This is a representative list; extend as required.
    nse_holidays_md: List[Tuple[int, int]] = [
        (1, 26),   # Republic Day
        (3, 25),   # Holi (approximate – floats; here as a placeholder)
        (4, 14),   # Ambedkar Jayanti / Good Friday (approximate)
        (5, 1),    # Maharashtra Day / Labour Day
        (8, 15),   # Independence Day
        (10, 2),   # Gandhi Jayanti
        (10, 24),  # Dussehra (approximate)
        (11, 1),   # Diwali Laxmi Puja (approximate)
        (11, 15),  # Gurunanak Jayanti (approximate)
        (12, 25),  # Christmas
    ]

    holiday_dates = set()
    for yr in range(start.year, end.year + 1):
        for month, day in nse_holidays_md:
            try:
                holiday_dates.add(pd.Timestamp(yr, month, day))
            except ValueError:
                pass  # invalid date (e.g., Feb 30) – skip

    trading_days = bdays[~bdays.isin(holiday_dates)]
    return trading_days


def _handle_missing_days(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Compare the DataFrame's date index against the NSE trading calendar.

    Rules
    -----
    - Gaps of 1–2 days  → forward-fill (public holidays / data glitch).
    - Gaps > 3 days     → rows in that gap are dropped entirely (data error).
    - All gaps are logged at WARNING level.
    """
    df = df.copy().sort_index()

    if df.empty:
        return df

    expected = _nse_trading_days(df.index.min(), df.index.max())
    actual   = df.index

    missing = expected.difference(actual)

    if missing.empty:
        logger.debug("[%s] No missing trading days.", symbol)
        return df

    # Identify consecutive run lengths in the missing dates
    missing_series = pd.Series(missing.sort_values())
    gaps: List[List[pd.Timestamp]] = []
    current_run: List[pd.Timestamp] = [missing_series.iloc[0]]

    for ts in missing_series.iloc[1:]:
        if (ts - current_run[-1]).days <= 3:   # within 3 cal days → same gap
            current_run.append(ts)
        else:
            gaps.append(current_run)
            current_run = [ts]
    gaps.append(current_run)

    rows_to_drop: List[pd.Timestamp] = []

    for gap in gaps:
        n = len(gap)
        if n <= MAX_GAP_FILL:
            logger.warning(
                "[%s] Gap of %d trading day(s) around %s – will forward-fill.",
                symbol, n, gap[0].date(),
            )
        else:
            logger.warning(
                "[%s] Large gap of %d trading days starting %s – marking for drop.",
                symbol, n, gap[0].date(),
            )
            rows_to_drop.extend(gap)

    # Reindex to full calendar, forward-fill short gaps, then drop large-gap rows
    df = df.reindex(expected)
    df = df.ffill(limit=MAX_GAP_FILL)

    # Drop rows that correspond to large-gap dates (they will be NaN after reindex)
    df = df.dropna(subset=["Close"])

    # Also remove any explicitly flagged large-gap rows
    drop_idx = [d for d in rows_to_drop if d in df.index]
    if drop_idx:
        df = df.drop(index=drop_idx)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Outlier Clipping
# ─────────────────────────────────────────────────────────────────────────────

def _clip_outliers(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Detect and clip extreme single-day price moves.

    Any row where  |daily_return|  >  MAX_RETURN_CLIP (20%)  is treated as a
    data anomaly (bad tick, un-adjusted split, etc.).

    Action taken
    ------------
    - The Close on that row is clipped to  prev_close × (1 ± 0.20).
    - Open / High / Low are scaled by the same clip ratio so the candle shape
      is preserved but the magnitude is bounded.
    - All clipped rows are logged at WARNING level.
    """
    df = df.copy()

    prev_close = df["Close"].shift(1)
    daily_ret  = (df["Close"] - prev_close) / prev_close.replace(0, np.nan)

    upper_flag = daily_ret >  MAX_RETURN_CLIP
    lower_flag = daily_ret < -MAX_RETURN_CLIP
    flagged    = upper_flag | lower_flag

    if flagged.any():
        n = int(flagged.sum())
        flagged_dates = df.index[flagged].strftime("%Y-%m-%d").tolist()
        logger.warning(
            "[%s] Clipping %d row(s) with |return| > %.0f%%: %s",
            symbol, n, MAX_RETURN_CLIP * 100, flagged_dates,
        )

        clipped_close_upper = prev_close * (1 + MAX_RETURN_CLIP)
        clipped_close_lower = prev_close * (1 - MAX_RETURN_CLIP)

        original_close = df["Close"].copy()

        df.loc[upper_flag, "Close"] = clipped_close_upper[upper_flag]
        df.loc[lower_flag, "Close"] = clipped_close_lower[lower_flag]

        # Compute the clip scale factor and apply to OHLC to keep shape
        clip_ratio = df["Close"] / original_close.replace(0, np.nan)
        clip_ratio = clip_ratio.fillna(1.0)

        for col in ["Open", "High", "Low"]:
            df.loc[flagged, col] = df.loc[flagged, col] * clip_ratio[flagged]

        # Re-enforce High >= Low after clipping
        df["High"] = df[["High", "Low"]].max(axis=1)

        # Update Adj_Close to match
        if "Adj_Close" in df.columns:
            df.loc[flagged, "Adj_Close"] = df.loc[flagged, "Close"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Candle Validity Filter
# ─────────────────────────────────────────────────────────────────────────────

def _filter_invalid_candles(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Remove structurally invalid OHLCV rows.

    Rules
    -----
    - high < low           → OHLC inversion (data error)
    - close <= 0           → negative / zero price
    - volume == 0          → non-trading / phantom row
    - total_range < 0.01%  → zero-range / stale candle
    """
    df = df.copy()
    before = len(df)

    total_range = (df["High"] - df["Low"]).abs()
    min_range   = MIN_RANGE_PCT * df["Close"].abs()

    bad_mask = (
        (df["High"] < df["Low"])          |   # OHLC inversion
        (df["Close"] <= 0)                |   # non-positive price
        (df["Volume"] == 0)               |   # no trades
        (total_range < min_range)             # zero-range candle
    )

    n_bad = int(bad_mask.sum())
    if n_bad:
        bad_dates = df.index[bad_mask].strftime("%Y-%m-%d").tolist()
        logger.warning(
            "[%s] Dropping %d invalid candle(s): %s",
            symbol, n_bad,
            bad_dates[:10],   # show at most 10 in the log
        )
        df = df[~bad_mask]

    after = len(df)
    logger.debug("[%s] Candle filter: %d -> %d rows (%d dropped).", symbol, before, after, before - after)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Data Type Enforcement
# ─────────────────────────────────────────────────────────────────────────────

def _enforce_dtypes(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Guarantee correct dtypes for all output columns.

    - Open, High, Low, Close, Adj_Close → float64
    - Volume                            → int64
    - Index                             → DatetimeIndex (UTC-naive)
    """
    df = df.copy()

    float_cols = [c for c in ["Open", "High", "Low", "Close", "Adj_Close"] if c in df.columns]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    if "Volume" in df.columns:
        df["Volume"] = (
            pd.to_numeric(df["Volume"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )

    # Ensure DatetimeIndex (drop tz info if present to keep things simple)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"

    # Final NaN sweep – only drop rows missing any price column
    df = df.dropna(subset=float_cols)

    logger.debug("[%s] Dtypes enforced. Final shape: %s.", symbol, df.shape)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Output helpers
# ─────────────────────────────────────────────────────────────────────────────

_OUTPUT_COLS = ["Open", "High", "Low", "Close", "Volume", "Adj_Close"]


def _save_processed(df: pd.DataFrame, symbol: str, processed_dir: Path) -> None:
    """Persist the cleaned DataFrame to data/processed/{SYMBOL}.csv."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace(".", "_").replace("/", "-")
    out  = processed_dir / f"{safe}.csv"
    df.to_csv(out)
    logger.info("[%s] Saved processed file -> %s (%d rows).", symbol, out.name, len(df))


def _select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the columns defined in _OUTPUT_COLS (in that order)."""
    present = [c for c in _OUTPUT_COLS if c in df.columns]
    return df[present]


# ─────────────────────────────────────────────────────────────────────────────
# Main single-stock pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_one(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    processed_dir: Path = PROCESSED_DIR,
    save: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Run the full preprocessing pipeline on a single OHLCV DataFrame.

    Parameters
    ----------
    df            : Raw OHLCV DataFrame as returned by yfinance / load_all_data().
    symbol        : Ticker string used for logging and output filename.
    processed_dir : Directory where the cleaned CSV is saved.
    save          : If True, persist the result to disk.

    Returns
    -------
    Cleaned pd.DataFrame on success, or None if the stock is unusable after
    cleaning (e.g. fewer than 60 valid rows remain).

    Pipeline order
    --------------
    1. Corporate action adjustment
    2. Missing day handling
    3. Outlier clipping
    4. Candle validity filter
    5. Data type enforcement
    6. Column selection + save
    """
    if df is None or df.empty:
        logger.warning("[%s] Received empty DataFrame – skipping.", symbol)
        return None

    logger.info("[%s] Starting preprocessing (%d raw rows).", symbol, len(df))

    try:
        df = _adjust_corporate_actions(df, symbol)      # Step 1
        df = _handle_missing_days(df, symbol)           # Step 2
        df = _clip_outliers(df, symbol)                 # Step 3
        df = _filter_invalid_candles(df, symbol)        # Step 4
        df = _enforce_dtypes(df, symbol)                # Step 5
        df = _select_output_columns(df)                 # Step 6 – column order

    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] Pipeline failed: %s", symbol, exc, exc_info=True)
        return None

    MIN_ROWS = 60
    if len(df) < MIN_ROWS:
        logger.warning(
            "[%s] Only %d rows after cleaning (minimum %d) – discarding.",
            symbol, len(df), MIN_ROWS,
        )
        return None

    if save:
        _save_processed(df, symbol, processed_dir)

    logger.info("[%s] Preprocessing complete – %d clean rows.", symbol, len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Parallel universe pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_all(
    raw_data_dict: Dict[str, pd.DataFrame],
    processed_dir: Path = PROCESSED_DIR,
    max_workers: int = 8,
) -> Dict[str, pd.DataFrame]:
    """
    Apply the preprocessing pipeline to every stock in `raw_data_dict` using
    a thread pool for I/O-bound parallelism.

    Parameters
    ----------
    raw_data_dict : {symbol: raw_df} dict as returned by load_all_data().
    processed_dir : Directory in which cleaned CSVs are saved.
    max_workers   : Thread pool size (default 8; tune to CPU count).

    Returns
    -------
    {symbol: cleaned_df}  – only stocks that survived the pipeline are included.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, pd.DataFrame] = {}
    failed:  List[str]              = []

    total = len(raw_data_dict)
    logger.info(
        "Preprocessing %d stocks with %d workers …", total, max_workers
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(preprocess_one, df, symbol, processed_dir): symbol
            for symbol, df in raw_data_dict.items()
        }

        for i, future in enumerate(as_completed(future_to_symbol), start=1):
            symbol = future_to_symbol[future]
            try:
                cleaned = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] Unhandled exception: %s", symbol, exc)
                cleaned = None

            if cleaned is not None:
                results[symbol] = cleaned
            else:
                failed.append(symbol)

            if i % 50 == 0 or i == total:
                logger.info("  Progress: %d / %d done.", i, total)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Preprocessing complete")
    logger.info("  Passed  : %d / %d", len(results), total)
    logger.info("  Dropped : %d  %s", len(failed), failed if failed else "")
    logger.info("=" * 60)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience loader
# ─────────────────────────────────────────────────────────────────────────────

def load_processed(processed_dir: Path = PROCESSED_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load all previously cleaned CSVs from `processed_dir`.

    Returns
    -------
    {symbol: DataFrame}
    """
    data: Dict[str, pd.DataFrame] = {}
    csv_files = sorted(processed_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No processed files found in %s.", processed_dir)
        return data

    for csv_path in csv_files:
        stem   = csv_path.stem
        symbol = (stem[:-3] + ".NS") if stem.endswith("_NS") else stem.replace("_", ".")
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            data[symbol] = df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load %s: %s", csv_path.name, exc)

    logger.info("Loaded %d processed files from %s.", len(data), processed_dir)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Preprocess raw Nifty 500 OHLCV CSVs."
    )
    parser.add_argument("--raw-dir",       default="data/raw",       help="Raw CSV directory")
    parser.add_argument("--processed-dir", default="data/processed", help="Output directory")
    parser.add_argument("--workers",       type=int, default=8,      help="Thread pool size")
    parser.add_argument("--symbols",       nargs="*", metavar="SYM", help="Process specific symbols only")
    args = parser.parse_args()

    from src.data_ingestion import load_all_data  # noqa: E402

    raw_dir       = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    raw_data = load_all_data(raw_dir=raw_dir, min_trading_days=1)

    if args.symbols:
        raw_data = {s: df for s, df in raw_data.items() if s in args.symbols}

    preprocess_all(raw_data, processed_dir=processed_dir, max_workers=args.workers)
