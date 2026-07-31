"""
data_ingestion.py
-----------------
Downloads and loads OHLCV data for the Nifty 500 universe from Yahoo Finance.

Key functions
-------------
fetch_nifty500_data(symbols, period, interval)
    Download historical data for all symbols → data/raw/{SYMBOL}.csv

load_all_data(raw_dir, min_trading_days)
    Load all CSVs → {symbol: DataFrame}, filtered by minimum row count

Usage (CLI)
-----------
    python src/data_ingestion.py                    # download full universe
    python src/data_ingestion.py --symbols TCS INFY # specific tickers only
    python src/data_ingestion.py --load-only        # just load & report
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from .symbols import NIFTY500

# ────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_ingestion")

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")
DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"
MAX_RETRIES = 3
RETRY_DELAY = 5        # seconds between retries
BATCH_DELAY = 0.3      # polite pause between ticker downloads
MIN_TRADING_DAYS = 400


# ────────────────────────────────────────────────────────────────────────────
# Download helpers
# ────────────────────────────────────────────────────────────────────────────

def _csv_path(symbol: str, raw_dir: Path) -> Path:
    """Return the expected CSV path for a given symbol."""
    safe = symbol.replace(".", "_").replace("/", "-")
    return raw_dir / f"{safe}.csv"


def _download_one(
    symbol: str,
    period: str,
    interval: str,
    raw_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for a single symbol with retry logic.

    Returns the DataFrame on success, or None if all retries fail.
    Skips symbols whose CSV already exists on disk.
    """
    csv = _csv_path(symbol, raw_dir)

    # ── Resume-friendly: skip already-downloaded files ──────────────────────
    if csv.exists():
        logger.debug("[SKIP] %s — already downloaded (%s)", symbol, csv.name)
        return pd.read_csv(csv, index_col=0, parse_dates=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if df is None or df.empty:
                logger.warning(
                    "[WARN] %s — empty response (attempt %d/%d)",
                    symbol, attempt, MAX_RETRIES,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                continue

            # Flatten multi-level column index that yfinance sometimes returns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index.name = "Date"
            df.to_csv(csv)
            logger.info("[OK]   %s - %d rows saved -> %s", symbol, len(df), csv.name)
            return df

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[ERR]  %s — attempt %d/%d failed: %s",
                symbol, attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                logger.info("       Retrying in %ds …", RETRY_DELAY)
                time.sleep(RETRY_DELAY)

    logger.error("[FAIL] %s — all %d attempts exhausted. Skipping.", symbol, MAX_RETRIES)
    return None


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def fetch_nifty500_data(
    symbols: List[str] = NIFTY500,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    raw_dir: Path = RAW_DIR,
) -> Dict[str, pd.DataFrame]:
    """
    Download historical OHLCV data for the given list of NSE symbols.

    Parameters
    ----------
    symbols  : list of Yahoo Finance ticker strings, e.g. ["RELIANCE.NS", "TCS.NS"]
    period   : yfinance period string, default "2y" (2 years of history)
    interval : bar interval, default "1d" (daily)
    raw_dir  : directory in which per-symbol CSVs are saved

    Returns
    -------
    dict mapping symbol → DataFrame (only successfully downloaded tickers)

    Behaviour
    ---------
    - Already-downloaded CSVs are skipped (resumable run).
    - Each symbol is retried up to MAX_RETRIES (3) times on failure.
    - A RETRY_DELAY (5 s) pause is inserted between retry attempts.
    - A short BATCH_DELAY (0.3 s) pause is inserted between symbols.
    - Per-symbol success/failure is logged at INFO / ERROR level.
    - A summary is printed at the end.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting download: %d symbols | period=%s | interval=%s | dest=%s",
        len(symbols), period, interval, raw_dir,
    )

    successful: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []

    for i, symbol in enumerate(symbols, start=1):
        logger.info("(%d/%d) %s", i, len(symbols), symbol)
        df = _download_one(symbol, period, interval, raw_dir)
        if df is not None:
            successful[symbol] = df
        else:
            failed.append(symbol)
        time.sleep(BATCH_DELAY)

    # ── Summary ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Download complete")
    logger.info("  Successful : %d", len(successful))
    logger.info("  Failed     : %d  %s", len(failed), failed if failed else "")
    logger.info("=" * 60)

    return successful


def load_all_data(
    raw_dir: Path = RAW_DIR,
    min_trading_days: int = MIN_TRADING_DAYS,
) -> Dict[str, pd.DataFrame]:
    """
    Load every CSV in `raw_dir` and return a filtered symbol dictionary.

    Parameters
    ----------
    raw_dir           : directory containing per-symbol CSV files
    min_trading_days  : minimum number of rows a ticker must have to be kept
                        (default 400 ≈ ~20 months of trading days)

    Returns
    -------
    dict mapping symbol → DataFrame (only tickers with enough history)

    Side-effects
    ------------
    Prints a human-readable summary to stdout:
      • Total files found on disk
      • Stocks loaded (passed the filter)
      • Stocks dropped (insufficient history)
      • Date range of the loaded data
    """
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s. Run fetch_nifty500_data() first.", raw_dir)
        return {}

    logger.info("Found %d CSV files in %s — loading …", len(csv_files), raw_dir)

    all_data: Dict[str, pd.DataFrame] = {}
    dropped: List[Tuple[str, int]] = []

    for csv_path in csv_files:
        # Reconstruct symbol from filename.
        # _csv_path() turns RELIANCE.NS -> RELIANCE_NS.csv
        # so we reverse: strip .csv, replace trailing _NS with .NS
        stem = csv_path.stem          # e.g. "RELIANCE_NS", "BAJAJ-AUTO_NS"
        if stem.endswith("_NS"):
            symbol = stem[:-3] + ".NS"   # RELIANCE_NS -> RELIANCE.NS
        else:
            symbol = stem.replace("_", ".")  # fallback

        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", csv_path.name, exc)
            continue

        if df.empty or len(df) < min_trading_days:
            rows = len(df)
            dropped.append((symbol, rows))
            logger.debug(
                "[DROP] %s — only %d rows (< %d required)", symbol, rows, min_trading_days
            )
            continue

        all_data[symbol] = df
        logger.debug("[LOAD] %s — %d rows", symbol, len(df))

    # ── Build summary ────────────────────────────────────────────────────────
    if all_data:
        all_dates = pd.concat(
            [df.index.to_series() for df in all_data.values()]
        )
        date_min = all_dates.min().strftime("%Y-%m-%d")
        date_max = all_dates.max().strftime("%Y-%m-%d")
    else:
        date_min = date_max = "N/A"

    print("\n" + "=" * 55)
    print("  Nifty 500 Data Load Summary")
    print("=" * 55)
    print(f"  CSV files on disk   : {len(csv_files)}")
    print(f"  Stocks loaded       : {len(all_data)}")
    print(f"  Stocks dropped      : {len(dropped)}  (< {min_trading_days} trading days)")
    print(f"  Date range          : {date_min} -> {date_max}")
    if dropped:
        print(f"\n  Dropped tickers     : {[s for s, _ in dropped]}")
    print("=" * 55 + "\n")

    return all_data


# ────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Nifty 500 data ingestion — download & load OHLCV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--symbols", nargs="*", metavar="SYM",
        help="Override symbol list, e.g. --symbols TCS.NS INFY.NS",
    )
    p.add_argument(
        "--period", default=DEFAULT_PERIOD,
        help=f"yfinance period string (default: {DEFAULT_PERIOD})",
    )
    p.add_argument(
        "--interval", default=DEFAULT_INTERVAL,
        help=f"Bar interval (default: {DEFAULT_INTERVAL})",
    )
    p.add_argument(
        "--raw-dir", default=str(RAW_DIR), metavar="DIR",
        help=f"Directory to save raw CSVs (default: {RAW_DIR})",
    )
    p.add_argument(
        "--min-days", type=int, default=MIN_TRADING_DAYS, metavar="N",
        help=f"Minimum trading days filter for load (default: {MIN_TRADING_DAYS})",
    )
    p.add_argument(
        "--load-only", action="store_true",
        help="Skip downloading; just load existing CSVs and print summary",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    raw_dir = Path(args.raw_dir)

    if not args.load_only:
        symbols = args.symbols if args.symbols else NIFTY500
        fetch_nifty500_data(
            symbols=symbols,
            period=args.period,
            interval=args.interval,
            raw_dir=raw_dir,
        )

    load_all_data(raw_dir=raw_dir, min_trading_days=args.min_days)
