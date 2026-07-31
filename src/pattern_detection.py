"""
pattern_detection.py
--------------------
Rule-based candlestick pattern detection engine for the Nifty 500 universe.

Detects 15 patterns (7 single-candle, 8 multi-candle) on daily OHLCV data.
Each detected event is returned as a structured dict with strength scoring.

Public API
----------
detect_all_patterns(df)  -> list[dict]   # scan full DataFrame, return events
tag_dataframe(df)        -> pd.DataFrame # add binary + meta columns in-place
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global thresholds
# ─────────────────────────────────────────────────────────────────────────────

MIN_RANGE_PCT  = 0.01   # 1%  of close  – reject candles smaller than this
MAX_RANGE_PCT  = 0.10   # 10% of close  – reject abnormally large candles
TREND_WINDOW   = 10     # days for slope calculation
SLOPE_FLAT_TOL = 1e-5   # treat slope as flat if |slope| < this (per-unit)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern event dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatternEvent:
    pattern_name:       str
    pattern_type:       str           # "single" | "multi"
    direction:          str           # "bullish" | "bearish" | "neutral"
    pattern_start_date: pd.Timestamp
    pattern_end_date:   pd.Timestamp
    num_candles:        int
    raw_strength:       float         # 0.0 – 1.0
    volume_confirm:     bool = False
    prior_trend:        str  = "neutral"   # "up" | "down" | "neutral"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pattern_start_date"] = self.pattern_start_date
        d["pattern_end_date"]   = self.pattern_end_date
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Low-level candle helpers
# ─────────────────────────────────────────────────────────────────────────────

def _candle_parts(o, h, l, c) -> Tuple[float, float, float, float, float]:
    """Return (body, body_abs, upper_shadow, lower_shadow, total_range)."""
    body        = c - o
    body_abs    = abs(body)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    total_range  = h - l
    return body, body_abs, upper_shadow, lower_shadow, total_range


def _body_to_range(body_abs: float, total_range: float) -> float:
    return body_abs / total_range if total_range > 0 else 0.0


def _is_range_valid(total_range: float, close: float) -> bool:
    """Enforce MIN_RANGE_PCT ≤ total_range/close ≤ MAX_RANGE_PCT."""
    if close <= 0:
        return False
    ratio = total_range / close
    return MIN_RANGE_PCT <= ratio <= MAX_RANGE_PCT


def _prior_trend(closes: np.ndarray) -> str:
    """
    Compute linear regression slope over the last TREND_WINDOW closes.
    Returns "up", "down", or "neutral".
    """
    n = len(closes)
    if n < 2:
        return "neutral"
    x  = np.arange(n, dtype=float)
    # Normalise slope by mean price so it's scale-independent
    mean_p = closes.mean() if closes.mean() != 0 else 1.0
    slope  = np.polyfit(x, closes, 1)[0] / mean_p
    if slope > SLOPE_FLAT_TOL:
        return "up"
    if slope < -SLOPE_FLAT_TOL:
        return "down"
    return "neutral"


def _volume_above_avg(vol_series: pd.Series, idx: int, window: int = 20) -> bool:
    """Return True if volume at `idx` is above the rolling average."""
    if idx < 1:
        return False
    start = max(0, idx - window)
    avg   = vol_series.iloc[start:idx].mean()
    return bool(vol_series.iloc[idx] > avg) if avg > 0 else False


def _strength_single(
    body_abs: float,
    total_range: float,
    upper_shadow: float,
    lower_shadow: float,
    vol_ratio: float,
) -> float:
    """
    Strength for single-candle patterns (0-1).
    Weighted blend of body dominance, shadow balance, and volume.
    """
    btr       = _body_to_range(body_abs, total_range)
    vol_score = min(vol_ratio / 2.0, 1.0) if vol_ratio > 0 else 0.5
    return round(min(0.5 * btr + 0.3 * (1 - (upper_shadow + lower_shadow) / (total_range + 1e-9)) + 0.2 * vol_score, 1.0), 4)


def _strength_multi(
    body_abs_last: float,
    total_range_last: float,
    vol_confirm: bool,
) -> float:
    """Strength for multi-candle patterns (0-1)."""
    btr       = _body_to_range(body_abs_last, total_range_last)
    vol_bonus = 0.2 if vol_confirm else 0.0
    return round(min(0.8 * btr + vol_bonus, 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Single-candle detectors
# ─────────────────────────────────────────────────────────────────────────────

def _detect_single(
    idx: int,
    df: pd.DataFrame,
    vol_avg: pd.Series,
) -> Optional[PatternEvent]:
    """
    Run all single-candle checks for the candle at position `idx`.
    Returns the first matching PatternEvent or None.
    """
    row = df.iloc[idx]
    o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
    v           = float(row["Volume"])
    date        = df.index[idx]

    body, body_abs, us, ls, tr = _candle_parts(o, h, l, c)

    if not _is_range_valid(tr, c):
        return None

    # Prior trend from last TREND_WINDOW closes (excluding current)
    start_t = max(0, idx - TREND_WINDOW)
    trend   = _prior_trend(df["Close"].iloc[start_t:idx].values)

    avg_vol    = vol_avg.iloc[idx] if vol_avg.iloc[idx] > 0 else 1.0
    vol_ratio  = v / avg_vol
    vol_confirm = vol_ratio >= 1.2

    btr = _body_to_range(body_abs, tr)
    bullish = c >= o

    # ── 1. Bullish Marubozu ──────────────────────────────────────────────
    if bullish and btr >= 0.95 and ls <= 0.05 * body_abs and us <= 0.05 * body_abs:
        return PatternEvent(
            "bullish_marubozu", "single", "bullish", date, date, 1,
            _strength_single(body_abs, tr, us, ls, vol_ratio), vol_confirm, trend,
        )

    # ── 2. Bearish Marubozu ──────────────────────────────────────────────
    if not bullish and btr >= 0.95 and ls <= 0.05 * body_abs and us <= 0.05 * body_abs:
        return PatternEvent(
            "bearish_marubozu", "single", "bearish", date, date, 1,
            _strength_single(body_abs, tr, us, ls, vol_ratio), vol_confirm, trend,
        )

    # ── 3. Doji ──────────────────────────────────────────────────────────
    if body_abs <= 0.05 * tr and us > 0 and ls > 0:
        strength = _strength_single(body_abs, tr, us, ls, vol_ratio)
        return PatternEvent(
            "doji", "single", "neutral", date, date, 1,
            strength, vol_confirm, trend,
        )

    # ── 4. Spinning Top ──────────────────────────────────────────────────
    if btr <= 0.35 and us >= 0.25 * tr and ls >= 0.25 * tr:
        strength = _strength_single(body_abs, tr, us, ls, vol_ratio)
        return PatternEvent(
            "spinning_top", "single", "neutral", date, date, 1,
            strength, vol_confirm, trend,
        )

    # ── 5. Hammer / Hanging Man ──────────────────────────────────────────
    # Body in upper 30% of range → body_low >= l + 0.70 * tr
    body_low = min(o, c)
    if (ls >= 2 * body_abs and us <= 0.1 * tr and body_abs > 0
            and body_low >= l + 0.70 * tr):
        if trend == "down":
            name, direction = "hammer", "bullish"
        elif trend == "up":
            name, direction = "hanging_man", "bearish"
        else:
            name, direction = "hammer", "bullish"   # neutral → treat as hammer

        strength = _strength_single(body_abs, tr, us, ls, vol_ratio)
        return PatternEvent(
            name, "single", direction, date, date, 1,
            strength, vol_confirm, trend,
        )

    # ── 6. Shooting Star ─────────────────────────────────────────────────
    # Body in lower 30% of range → body_high <= l + 0.30 * tr
    body_high = max(o, c)
    if (us >= 2 * body_abs and ls <= 0.1 * tr and body_abs > 0
            and body_high <= l + 0.30 * tr and trend == "up"):
        strength = _strength_single(body_abs, tr, us, ls, vol_ratio)
        return PatternEvent(
            "shooting_star", "single", "bearish", date, date, 1,
            strength, vol_confirm, trend,
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Multi-candle detectors
# ─────────────────────────────────────────────────────────────────────────────

def _detect_two_candle(
    idx: int,
    df: pd.DataFrame,
    vol_avg: pd.Series,
) -> Optional[PatternEvent]:
    """Run all two-candle checks ending at `idx`."""
    if idx < 1:
        return None

    p1_row = df.iloc[idx - 1]
    p2_row = df.iloc[idx]
    date_start = df.index[idx - 1]
    date_end   = df.index[idx]

    o1, h1, l1, c1 = float(p1_row["Open"]), float(p1_row["High"]), float(p1_row["Low"]), float(p1_row["Close"])
    o2, h2, l2, c2 = float(p2_row["Open"]), float(p2_row["High"]), float(p2_row["Low"]), float(p2_row["Close"])
    v1, v2         = float(p1_row["Volume"]), float(p2_row["Volume"])

    _, b1_abs, us1, ls1, tr1 = _candle_parts(o1, h1, l1, c1)
    _, b2_abs, us2, ls2, tr2 = _candle_parts(o2, h2, l2, c2)

    if not (_is_range_valid(tr1, c1) and _is_range_valid(tr2, c2)):
        return None

    start_t = max(0, idx - 1 - TREND_WINDOW)
    trend   = _prior_trend(df["Close"].iloc[start_t: idx - 1].values)

    avg_vol2    = vol_avg.iloc[idx] if vol_avg.iloc[idx] > 0 else 1.0
    vol_confirm = v2 / avg_vol2 >= 1.2

    p1_bull = c1 >= o1
    p2_bull = c2 >= o2
    mid1    = (o1 + c1) / 2.0

    strength = _strength_multi(b2_abs, tr2, vol_confirm)

    # ── 8. Bullish Engulfing ─────────────────────────────────────────────
    if (not p1_bull and p2_bull and trend == "down"
            and o2 < c1 and c2 > o1):
        return PatternEvent("bullish_engulfing", "multi", "bullish",
                            date_start, date_end, 2, strength, vol_confirm, trend)

    # ── 9. Bearish Engulfing ─────────────────────────────────────────────
    if (p1_bull and not p2_bull and trend == "up"
            and o2 > c1 and c2 < o1):
        return PatternEvent("bearish_engulfing", "multi", "bearish",
                            date_start, date_end, 2, strength, vol_confirm, trend)

    # ── 10. Piercing Pattern ─────────────────────────────────────────────
    if (not p1_bull and p2_bull and trend == "down"
            and o2 < l1 and c2 > mid1 and c2 < o1):
        return PatternEvent("piercing_pattern", "multi", "bullish",
                            date_start, date_end, 2, strength, vol_confirm, trend)

    # ── 11. Dark Cloud Cover ─────────────────────────────────────────────
    if (p1_bull and not p2_bull and trend == "up"
            and o2 > h1 and c2 < mid1 and c2 > o1):
        return PatternEvent("dark_cloud_cover", "multi", "bearish",
                            date_start, date_end, 2, strength, vol_confirm, trend)

    # ── 12. Bullish Harami ───────────────────────────────────────────────
    # P1 large bearish, P2 small bullish body inside P1 body
    if (not p1_bull and p2_bull and b1_abs > 0 and b2_abs > 0
            and o2 > c1 and c2 < o1              # P2 body inside P1 body
            and b2_abs < 0.5 * b1_abs):          # P2 smaller than P1
        return PatternEvent("bullish_harami", "multi", "bullish",
                            date_start, date_end, 2, strength, vol_confirm, trend)

    # ── 13. Bearish Harami ───────────────────────────────────────────────
    if (p1_bull and not p2_bull and b1_abs > 0 and b2_abs > 0
            and o2 < c1 and c2 > o1
            and b2_abs < 0.5 * b1_abs):
        return PatternEvent("bearish_harami", "multi", "bearish",
                            date_start, date_end, 2, strength, vol_confirm, trend)

    return None


def _detect_three_candle(
    idx: int,
    df: pd.DataFrame,
    vol_avg: pd.Series,
) -> Optional[PatternEvent]:
    """Run all three-candle checks ending at `idx`."""
    if idx < 2:
        return None

    p1_row = df.iloc[idx - 2]
    p2_row = df.iloc[idx - 1]
    p3_row = df.iloc[idx]
    date_start = df.index[idx - 2]
    date_end   = df.index[idx]

    o1, h1, l1, c1 = float(p1_row["Open"]), float(p1_row["High"]), float(p1_row["Low"]), float(p1_row["Close"])
    o2, h2, l2, c2 = float(p2_row["Open"]), float(p2_row["High"]), float(p2_row["Low"]), float(p2_row["Close"])
    o3, h3, l3, c3 = float(p3_row["Open"]), float(p3_row["High"]), float(p3_row["Low"]), float(p3_row["Close"])
    v1, v3         = float(p1_row["Volume"]), float(p3_row["Volume"])

    _, b1_abs, _, _, tr1 = _candle_parts(o1, h1, l1, c1)
    _, b2_abs, _, _, tr2 = _candle_parts(o2, h2, l2, c2)
    _, b3_abs, _, _, tr3 = _candle_parts(o3, h3, l3, c3)

    if not (_is_range_valid(tr1, c1) and _is_range_valid(tr3, c3)):
        return None

    start_t = max(0, idx - 2 - TREND_WINDOW)
    trend   = _prior_trend(df["Close"].iloc[start_t: idx - 2].values)

    avg_vol3    = vol_avg.iloc[idx] if vol_avg.iloc[idx] > 0 else 1.0
    vol_confirm = v3 > v1   # volume on P3 > P1 for confirmation

    p1_bull = c1 >= o1
    p3_bull = c3 >= o3
    mid1    = (o1 + c1) / 2.0
    btr2    = _body_to_range(b2_abs, tr2)

    strength = _strength_multi(b3_abs, tr3, vol_confirm)

    # ── 14. Morning Star ─────────────────────────────────────────────────
    # P1: large bearish | P2: small body, gaps down | P3: bullish, > mid P1
    if (not p1_bull and p3_bull and trend == "down"
            and b1_abs > 0.5 * tr1          # P1 large
            and btr2 <= 0.35                # P2 small body (doji / spinning top)
            and max(o2, c2) < c1            # P2 gaps down from P1 close
            and c3 > mid1                   # P3 closes above P1 midpoint
            and vol_confirm):               # volume confirmation
        return PatternEvent("morning_star", "multi", "bullish",
                            date_start, date_end, 3, strength, vol_confirm, trend)

    # ── 15. Evening Star ─────────────────────────────────────────────────
    # P1: large bullish | P2: small body, gaps up | P3: bearish, < mid P1
    if (p1_bull and not p3_bull and trend == "up"
            and b1_abs > 0.5 * tr1
            and btr2 <= 0.35
            and min(o2, c2) > c1            # P2 gaps up from P1 close
            and c3 < mid1
            and vol_confirm):
        return PatternEvent("evening_star", "multi", "bearish",
                            date_start, date_end, 3, strength, vol_confirm, trend)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main scanner
# ─────────────────────────────────────────────────────────────────────────────

def detect_all_patterns(df: pd.DataFrame) -> List[dict]:
    """
    Scan the entire DataFrame and return a list of detected pattern events.

    Parameters
    ----------
    df : Cleaned OHLCV DataFrame with DatetimeIndex.
         Must contain columns: Open, High, Low, Close, Volume.

    Returns
    -------
    List of dicts, one per detected pattern, with keys:
        pattern_name, pattern_type, direction,
        pattern_start_date, pattern_end_date,
        num_candles, raw_strength,
        volume_confirm, prior_trend
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    df = df.sort_index().copy()

    # Pre-compute rolling average volume (20-day) for volume scoring
    vol_avg = df["Volume"].rolling(20, min_periods=1).mean()

    events: List[dict] = []

    for idx in range(len(df)):
        # Three-candle (highest priority – check first to avoid double-counting)
        event = _detect_three_candle(idx, df, vol_avg)
        if event:
            events.append(event.to_dict())
            continue

        # Two-candle
        event = _detect_two_candle(idx, df, vol_avg)
        if event:
            events.append(event.to_dict())
            continue

        # Single-candle (only if no multi-candle found at this position)
        event = _detect_single(idx, df, vol_avg)
        if event:
            events.append(event.to_dict())

    logger.info("detect_all_patterns: %d events detected in %d rows.", len(events), len(df))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame tagger
# ─────────────────────────────────────────────────────────────────────────────

# All 15 pattern column names
ALL_PATTERN_NAMES: List[str] = [
    "bullish_marubozu", "bearish_marubozu", "spinning_top", "doji",
    "hammer", "hanging_man", "shooting_star",
    "bullish_engulfing", "bearish_engulfing",
    "piercing_pattern", "dark_cloud_cover",
    "bullish_harami", "bearish_harami",
    "morning_star", "evening_star",
]


def tag_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect all patterns and annotate `df` with result columns.

    Added columns
    -------------
    pat_{name}          int8   – binary flag (0/1) for each of the 15 patterns
    pattern_name        str    – name of the last detected pattern on each row
                                 (empty string if no pattern)
    pattern_direction   str    – "bullish" | "bearish" | "neutral" | ""
    pattern_strength    float  – raw_strength of the last detected pattern (0-1)
    pattern_type        str    – "single" | "multi" | ""
    prior_trend         str    – "up" | "down" | "neutral"

    Parameters
    ----------
    df : Cleaned OHLCV DataFrame.

    Returns
    -------
    Copy of `df` with the new columns appended.
    """
    df = df.copy()

    # Initialise binary columns
    for name in ALL_PATTERN_NAMES:
        df[f"pat_{name}"] = np.int8(0)

    # Initialise meta columns
    df["pattern_name"]      = ""
    df["pattern_direction"] = ""
    df["pattern_strength"]  = 0.0
    df["pattern_type"]      = ""
    df["prior_trend"]       = ""

    events = detect_all_patterns(df)

    for evt in events:
        end_date = evt["pattern_end_date"]
        name     = evt["pattern_name"]

        if end_date not in df.index:
            continue

        col = f"pat_{name}"
        if col in df.columns:
            df.at[end_date, col] = np.int8(1)

        # Only overwrite meta cols if no stronger pattern already tagged this row
        existing_str = df.at[end_date, "pattern_strength"]
        if evt["raw_strength"] >= existing_str:
            df.at[end_date, "pattern_name"]      = name
            df.at[end_date, "pattern_direction"]  = evt["direction"]
            df.at[end_date, "pattern_strength"]   = evt["raw_strength"]
            df.at[end_date, "pattern_type"]       = evt["pattern_type"]
            df.at[end_date, "prior_trend"]        = evt["prior_trend"]

    # Summary
    total_tagged = (df["pattern_name"] != "").sum()
    by_name = {n: int(df[f"pat_{n}"].sum()) for n in ALL_PATTERN_NAMES if df[f"pat_{n}"].sum() > 0}
    logger.info(
        "tag_dataframe: %d rows tagged. Counts: %s",
        total_tagged, by_name,
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pattern occurrence filter (for model training)
# ─────────────────────────────────────────────────────────────────────────────

def filter_rare_patterns(
    df: pd.DataFrame,
    min_occurrences: int = 50,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop pat_* columns that appear fewer than `min_occurrences` times.
    Returns the filtered DataFrame and the list of kept pattern names.
    """
    pat_cols = [c for c in df.columns if c.startswith("pat_")]
    keep     = [c for c in pat_cols if df[c].sum() >= min_occurrences]
    drop     = [c for c in pat_cols if c not in keep]
    if drop:
        logger.info("Dropping %d rare pattern columns: %s", len(drop), drop)
    return df.drop(columns=drop), [c[4:] for c in keep]   # strip "pat_" prefix
