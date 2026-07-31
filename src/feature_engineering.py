"""
feature_engineering.py
-----------------------
Transforms a tagged OHLCV DataFrame (output of pattern_detection.tag_dataframe)
into a fully feature-engineered DataFrame ready for ML training.

Feature groups
--------------
  1. Price features          (candle geometry)
  2. Trend features          (EMAs, slopes, alignment, HH/LL)
  3. Volatility features     (ATR, Bollinger Bands)
  4. Gap features            (gap size, direction, fill)
  5. Volume features         (SMA, ratio, spike, trend)
  6. Momentum features       (RSI, MACD, ROC, composite score)
  7. Support/Resistance      (rolling high/low, distance, breakout)
  8. Pattern features        (15 binary flags + meta from pattern_detection)
  9. Target variable         (forward_return_5d, target_label)

Public API
----------
engineer_features(df, symbol, features_dir, save) -> pd.DataFrame | None
engineer_all_features(processed_data_dict, features_dir, max_workers) -> pd.DataFrame
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .pattern_detection import tag_dataframe, ALL_PATTERN_NAMES

logger = logging.getLogger(__name__)

FEATURES_DIR = Path("data/features")

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast   = _ema(close, fast)
    ema_slow   = _ema(close, slow)
    macd_line  = ema_fast - ema_slow
    macd_sig   = _ema(macd_line, signal)
    macd_hist  = macd_line - macd_sig
    return macd_line, macd_sig, macd_hist


def _bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


# ─────────────────────────────────────────────────────────────────────────────
# Feature groups
# ─────────────────────────────────────────────────────────────────────────────

def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    df["body_size"]        = (c - o).abs()
    df["upper_wick"]       = h - df[["Open", "Close"]].max(axis=1)
    df["lower_wick"]       = df[["Open", "Close"]].min(axis=1) - l
    df["total_range"]      = h - l
    df["body_to_range_ratio"] = df["body_size"] / df["total_range"].replace(0, np.nan)
    df["wick_to_body_ratio"]  = (df["upper_wick"] + df["lower_wick"]) / (df["body_size"] + 1e-9)
    df["candle_direction"]    = np.where(c > o, 1, -1)
    return df


def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    h = df["High"]
    l = df["Low"]

    df["ema_20"]  = _ema(c, 20)
    df["ema_50"]  = _ema(c, 50)
    df["ema_200"] = _ema(c, 200)

    df["price_vs_ema20"] = (c - df["ema_20"]) / df["ema_20"].replace(0, np.nan)
    df["price_vs_ema50"] = (c - df["ema_50"]) / df["ema_50"].replace(0, np.nan)

    df["ema20_slope"] = (df["ema_20"] - df["ema_20"].shift(5)) / 5
    df["ema50_slope"] = (df["ema_50"] - df["ema_50"].shift(5)) / 5

    # trend_strength: +1 fully bullish (20>50>200), -1 fully bearish, 0 mixed
    e20, e50, e200 = df["ema_20"], df["ema_50"], df["ema_200"]
    bull_score = ((e20 > e50).astype(int) + (e50 > e200).astype(int) +
                  (e20 > e200).astype(int))   # 0-3
    bear_score = ((e20 < e50).astype(int) + (e50 < e200).astype(int) +
                  (e20 < e200).astype(int))   # 0-3
    df["trend_strength"] = (bull_score - bear_score) / 3.0  # -1..+1

    df["higher_highs"] = (
        (h > h.shift(1)) & (h.shift(1) > h.shift(2))
    ).astype(np.int8)

    df["lower_lows"] = (
        (l < l.shift(1)) & (l.shift(1) < l.shift(2))
    ).astype(np.int8)

    return df


def _add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["Close"], df["High"], df["Low"]

    df["atr_14"]   = _atr(h, l, c, 14)
    df["atr_ratio"] = df["atr_14"] / c.replace(0, np.nan)

    atr_shift = df["atr_14"].shift(5)
    df["volatility_expansion"]   = (df["atr_14"] > atr_shift * 1.2).astype(np.int8)
    df["volatility_contraction"] = (df["atr_14"] < atr_shift * 0.8).astype(np.int8)

    bb_upper, bb_mid, bb_lower = _bollinger(c, 20, 2.0)
    df["bbands_upper"]  = bb_upper
    df["bbands_lower"]  = bb_lower
    bb_width_denom      = c.replace(0, np.nan)
    df["bb_width"]      = (bb_upper - bb_lower) / bb_width_denom
    bb_range            = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"]   = (c - bb_lower) / bb_range

    return df


def _add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)

    df["gap"]      = o - prev_c
    df["gap_pct"]  = df["gap"] / prev_c.replace(0, np.nan)
    df["gap_up"]   = (df["gap_pct"] >  0.005).astype(np.int8)
    df["gap_down"] = (df["gap_pct"] < -0.005).astype(np.int8)

    # gap_fill_pct: fraction of the gap filled intraday
    # Positive gap: filled if low touches prev_close; pct = (open - low) / gap
    # Negative gap: filled if high touches prev_close; pct = (high - open) / |gap|
    gap_abs = df["gap"].abs().replace(0, np.nan)
    pos_fill = (o - l)  / gap_abs   # how far price moved back into gap (up-gap)
    neg_fill = (h - o)  / gap_abs   # how far price moved back (down-gap)
    df["gap_fill_pct"] = np.where(
        df["gap_up"],   pos_fill.clip(0, 1),
        np.where(df["gap_down"], neg_fill.clip(0, 1), 0.0),
    )

    return df


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    v = df["Volume"].astype(float)

    df["volume_sma20"]    = v.rolling(20).mean()
    df["volume_ratio"]    = v / df["volume_sma20"].replace(0, np.nan)
    df["volume_spike"]    = (df["volume_ratio"] > 2.0).astype(np.int8)
    df["relative_volume"] = v / v.rolling(5).mean().replace(0, np.nan)
    df["volume_trend"]    = (
        (v > v.shift(1)) & (v.shift(1) > v.shift(2))
    ).astype(np.int8)

    return df


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]

    df["rsi_14"]        = _rsi(c, 14)
    df["rsi_oversold"]  = (df["rsi_14"] < 35).astype(np.int8)
    df["rsi_overbought"] = (df["rsi_14"] > 65).astype(np.int8)

    macd_line, macd_sig, macd_hist = _macd(c)
    df["macd_line"]    = macd_line
    df["macd_signal"]  = macd_sig
    df["macd_hist"]    = macd_hist

    # macd_crossover: +1 when line crosses above signal, -1 when below, else 0
    prev_above = macd_line.shift(1) > macd_sig.shift(1)
    curr_above = macd_line          > macd_sig
    df["macd_crossover"] = np.where(
        (~prev_above) & curr_above,  1,
        np.where(prev_above & (~curr_above), -1, 0),
    ).astype(np.int8)

    df["roc_10"] = c.pct_change(10)

    # momentum_score: composite [−1, +1]
    rsi_pos    = (df["rsi_14"] - 50) / 50              # −1..+1
    macd_dir   = np.sign(df["macd_hist"])               # −1, 0, +1
    roc_sign   = np.sign(df["roc_10"])                  # −1, 0, +1
    df["momentum_score"] = (rsi_pos + macd_dir + roc_sign) / 3.0

    return df


def _add_support_resistance_features(df: pd.DataFrame) -> pd.DataFrame:
    h, l, c = df["High"], df["Low"], df["Close"]

    df["rolling_high_20"] = h.rolling(20).max()
    df["rolling_low_20"]  = l.rolling(20).min()

    df["distance_from_resistance"] = (
        (df["rolling_high_20"] - c) / c.replace(0, np.nan)
    )
    df["distance_from_support"] = (
        (c - df["rolling_low_20"]) / c.replace(0, np.nan)
    )

    # Breakout: close above previous period's rolling high
    df["breakout_strength"] = (
        c > df["rolling_high_20"].shift(1)
    ).astype(np.int8)

    return df


def _add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all 15 binary pattern columns exist (they come from tag_dataframe).
    Convert pattern_direction string to numeric.
    Add pattern_candle_count from num_candles meta (defaults to 0 if absent).
    """
    # Binary flags – ensure all 15 exist and are int8
    for name in ALL_PATTERN_NAMES:
        col = f"pat_{name}"
        if col not in df.columns:
            df[col] = np.int8(0)
        else:
            df[col] = df[col].fillna(0).astype(np.int8)

    # pattern_strength – already float; fill missing with 0
    if "pattern_strength" not in df.columns:
        df["pattern_strength"] = 0.0
    df["pattern_strength"] = df["pattern_strength"].fillna(0.0).astype(float)

    # Numeric direction: 1=bullish, -1=bearish, 0=neutral/empty
    if "pattern_direction" in df.columns:
        dir_map = {"bullish": 1, "bearish": -1, "neutral": 0, "": 0}
        df["pattern_direction_num"] = (
            df["pattern_direction"].map(dir_map).fillna(0).astype(np.int8)
        )
    else:
        df["pattern_direction_num"] = np.int8(0)

    # pattern_candle_count: derive from pattern_type
    if "pattern_type" in df.columns:
        type_map = {"single": 1, "multi": 2, "": 0}
        df["pattern_candle_count"] = (
            df["pattern_type"].map(type_map).fillna(0).astype(np.int8)
        )
    else:
        df["pattern_candle_count"] = np.int8(0)

    # 1. pattern_signed_strength
    df["pattern_signed_strength"] = (df["pattern_strength"] * df["pattern_direction_num"]).astype(np.float32)

    # Helper to get column or return series of zeros
    def get_col(col_name, default_val=0):
        if col_name in df.columns:
            return df[col_name]
        return pd.Series(default_val, index=df.index)

    trend_strength = get_col("trend_strength")
    pat_hammer = get_col("pat_hammer")
    pat_shooting_star = get_col("pat_shooting_star")
    pat_bullish_engulfing = get_col("pat_bullish_engulfing")
    volume_spike = get_col("volume_spike")
    pat_bearish_engulfing = get_col("pat_bearish_engulfing")
    pat_doji = get_col("pat_doji")
    distance_from_resistance = get_col("distance_from_resistance")
    distance_from_support = get_col("distance_from_support")
    pat_bullish_marubozu = get_col("pat_bullish_marubozu")
    breakout_strength = get_col("breakout_strength")
    pat_morning_star = get_col("pat_morning_star")
    pat_evening_star = get_col("pat_evening_star")
    rsi_oversold = get_col("rsi_oversold")
    rsi_overbought = get_col("rsi_overbought")
    pat_bullish_harami = get_col("pat_bullish_harami")
    pat_bearish_harami = get_col("pat_bearish_harami")

    df["hammer_in_downtrend"] = (pat_hammer * (trend_strength < -0.33)).astype(np.float32)
    df["shooting_star_in_uptrend"] = (pat_shooting_star * (trend_strength > 0.33)).astype(np.float32)
    df["engulfing_high_volume"] = (pat_bullish_engulfing * volume_spike).astype(np.float32)
    df["bearish_engulfing_vol"] = (pat_bearish_engulfing * volume_spike).astype(np.float32)
    df["doji_near_resistance"] = (pat_doji * (distance_from_resistance < 0.02)).astype(np.float32)
    df["doji_near_support"] = (pat_doji * (distance_from_support < 0.02)).astype(np.float32)
    df["marubozu_with_breakout"] = (pat_bullish_marubozu * breakout_strength).astype(np.float32)
    df["star_rsi_extreme"] = ((pat_morning_star | pat_evening_star) * (rsi_oversold | rsi_overbought)).astype(np.float32)
    df["harami_low_volume"] = ((pat_bullish_harami | pat_bearish_harami) * (1 - volume_spike)).astype(np.float32)

    df["pattern_bullish_score"] = (df["pattern_strength"] * np.maximum(0, df["pattern_direction_num"])).astype(np.float32)
    df["pattern_bearish_score"] = (df["pattern_strength"] * np.maximum(0, -df["pattern_direction_num"])).astype(np.float32)

    return df


def _add_target(df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
    c = df["Close"]
    df["forward_return_5d"] = (c.shift(-5) - c) / c.replace(0, np.nan)

    if "atr_ratio" in df.columns:
        atr_ratio = df["atr_ratio"]
    else:
        atr_14 = df.get("atr_14")
        if atr_14 is None:
            atr_14 = _atr(df["High"], df["Low"], c, 14)
        atr_ratio = atr_14 / c.replace(0, np.nan)

    normalised_return = df["forward_return_5d"] / (atr_ratio + 1e-9)
    df["target_label"] = np.where(
        normalised_return > 0.5, 1,
        np.where(normalised_return < -0.5, -1, 0),
    ).astype(np.int8)

    bull_count = int((df["target_label"] == 1).sum())
    bear_count = int((df["target_label"] == -1).sum())
    neutral_count = int((df["target_label"] == 0).sum())

    logger.info("[%s] Target: Bull=%d Bear=%d Neutral=%d",
                symbol, bull_count, bear_count, neutral_count)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# String / object columns to drop before saving
# ─────────────────────────────────────────────────────────────────────────────

_DROP_STRING_COLS = [
    "pattern_name", "pattern_direction", "pattern_type", "prior_trend",
    # Raw OHLCV kept as-is; Adj_Close is informational, kept
]


# ─────────────────────────────────────────────────────────────────────────────
# Single-stock pipeline
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    features_dir: Path = FEATURES_DIR,
    save: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Run the complete feature engineering pipeline on a single tagged OHLCV DataFrame.

    Parameters
    ----------
    df           : Output of preprocessing.preprocess_one (must contain OHLCV columns).
                   pattern_detection.tag_dataframe is called internally.
    symbol       : Ticker name for logging and file naming.
    features_dir : Directory in which to save the output CSV.
    save         : Persist result to data/features/{symbol}.csv if True.

    Returns
    -------
    Feature-engineered DataFrame (NaN rows dropped) or None if too few rows remain.
    """
    if df is None or df.empty:
        logger.warning("[%s] Empty input – skipping.", symbol)
        return None

    logger.info("[%s] Engineering features (%d rows) …", symbol, len(df))

    try:
        # Tag patterns first (if not already tagged)
        if "pat_doji" not in df.columns:
            df = tag_dataframe(df)

        # Apply each feature group
        df = _add_price_features(df)
        df = _add_trend_features(df)
        df = _add_volatility_features(df)
        df = _add_gap_features(df)
        df = _add_volume_features(df)
        df = _add_momentum_features(df)
        df = _add_support_resistance_features(df)
        df = _add_pattern_features(df)
        df = _add_target(df, symbol)

    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] Feature engineering failed: %s", symbol, exc, exc_info=True)
        return None

    # Drop string/object helper columns before NaN-drop
    drop_cols = [c for c in _DROP_STRING_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Enforce all numeric
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with any NaN (removes warmup period + last 5 target rows)
    before = len(df)
    df = df.dropna()
    after  = len(df)
    logger.debug("[%s] Dropped %d NaN rows (%d → %d).", symbol, before - after, before, after)

    MIN_ROWS = 50
    if after < MIN_ROWS:
        logger.warning(
            "[%s] Only %d rows remain after NaN drop (min %d) – discarding.",
            symbol, after, MIN_ROWS,
        )
        return None

    if save:
        features_dir.mkdir(parents=True, exist_ok=True)
        safe = symbol.replace(".", "_").replace("/", "-")
        out  = features_dir / f"{safe}_features.csv"
        df.to_csv(out)
        logger.info("[%s] Saved -> %s (%d rows, %d features).", symbol, out.name, len(df), len(df.columns))

    logger.info("[%s] Done – shape %s.", symbol, df.shape)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Universe pipeline
# ─────────────────────────────────────────────────────────────────────────────

def engineer_all_features(
    processed_data_dict: Dict[str, pd.DataFrame],
    features_dir: Path = FEATURES_DIR,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Apply engineer_features to every stock in parallel and return a single
    combined DataFrame with an added 'symbol' column.

    Parameters
    ----------
    processed_data_dict : {symbol: preprocessed_df} from preprocessing.preprocess_all
    features_dir        : Directory for per-symbol feature CSVs
    max_workers         : Thread pool size

    Returns
    -------
    Combined pd.DataFrame with all symbols stacked (DatetimeIndex preserved).
    Returns empty DataFrame if all stocks fail.
    """
    features_dir.mkdir(parents=True, exist_ok=True)
    frames:  List[pd.DataFrame] = []
    failed:  List[str]          = []
    total = len(processed_data_dict)

    logger.info("Feature engineering: %d stocks, %d workers …", total, max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {
            executor.submit(engineer_features, df, sym, features_dir, True): sym
            for sym, df in processed_data_dict.items()
        }

        for i, future in enumerate(as_completed(future_to_sym), start=1):
            sym = future_to_sym[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] Unhandled: %s", sym, exc)
                result = None

            if result is not None:
                result = result.copy()
                result.insert(0, "symbol", sym)
                frames.append(result)
            else:
                failed.append(sym)

            if i % 50 == 0 or i == total:
                logger.info("  Progress: %d / %d done.", i, total)

    logger.info("=" * 60)
    logger.info("Feature engineering complete")
    logger.info("  Passed  : %d / %d", len(frames), total)
    logger.info("  Dropped : %d  %s", len(failed), failed if failed else "")
    logger.info("=" * 60)

    if not frames:
        logger.warning("No feature DataFrames produced.")
        return pd.DataFrame()

    combined = pd.concat(frames, axis=0).sort_index()
    logger.info("Combined shape: %s  |  Features: %d", combined.shape, combined.shape[1] - 1)

    # Save combined
    out = features_dir / "combined_features.csv"
    combined.to_csv(out)
    logger.info("Combined features saved -> %s", out)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Feature column catalogue (for downstream use)
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return all ML-ready feature columns (excludes raw OHLCV, target, symbol).
    Intended for use in model.py to select X columns.
    """
    exclude = {
        "Open", "High", "Low", "Close", "Volume", "Adj_Close",
        "forward_return_5d", "target_label", "symbol",
        # pattern string meta (already encoded numerically)
        "pattern_name", "pattern_direction", "pattern_type", "prior_trend",
    }
    return [c for c in df.columns if c not in exclude]


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

    parser = argparse.ArgumentParser(description="Feature engineering for Nifty 500.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--features-dir",  default="data/features")
    parser.add_argument("--workers",       type=int, default=8)
    parser.add_argument("--symbols",       nargs="*", metavar="SYM")
    args = parser.parse_args()

    from src.preprocessing import load_processed  # noqa: E402

    processed = load_processed(Path(args.processed_dir))
    if args.symbols:
        processed = {s: df for s, df in processed.items() if s in args.symbols}

    combined = engineer_all_features(
        processed,
        features_dir=Path(args.features_dir),
        max_workers=args.workers,
    )
    print(f"\nCombined shape: {combined.shape}")
    print(f"Symbols: {combined['symbol'].unique().tolist() if 'symbol' in combined.columns else 'N/A'}")
    print(f"Target distribution:\n{combined['target_label'].value_counts().to_string()}")
