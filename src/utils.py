"""
utils.py
--------
Shared helper utilities used across the project:
  - Logger setup
  - Config loading
  - Report saving
  - Directory management
  - Plotting helpers
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(
    name: str = "nifty500",
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure and return a logger with a console handler (and optional file handler).

    Args:
        name:     Logger name.
        level:    Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
        log_file: Optional path to write log output to a file.

    Returns:
        Configured Logger instance.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt=datefmt)

        ch = logging.StreamHandler()
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(numeric_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file and return as a dict."""
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def ensure_dirs(cfg: dict) -> None:
    """Create all directories referenced in config if they don't exist."""
    paths = [
        cfg["data"]["raw_dir"],
        cfg["data"]["processed_dir"],
        cfg["data"]["features_dir"],
        cfg["output"]["reports_dir"],
        cfg["output"]["model_dir"],
        cfg["output"]["plots_dir"],
    ]
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Report utilities
# ---------------------------------------------------------------------------

def save_report(
    df: pd.DataFrame,
    filename: str,
    reports_dir: Path,
    index: bool = True,
) -> Path:
    """Save a DataFrame as a CSV report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / filename
    df.to_csv(out, index=index)
    logging.getLogger("nifty500").info("Report saved: %s", out)
    return out


def save_metrics(
    metrics: Dict[str, Any],
    filename: str,
    reports_dir: Path,
) -> Path:
    """Save a metrics dictionary as a single-row CSV."""
    df = pd.DataFrame([metrics])
    return save_report(df, filename, reports_dir, index=False)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_equity_curve(
    returns: pd.Series,
    title: str = "Equity Curve",
    save_path: Optional[Path] = None,
) -> None:
    """Plot and optionally save a cumulative returns (equity) curve."""
    cumulative = (1 + returns).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cumulative.index, cumulative.values, linewidth=1.5, color="#2196F3")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.fill_between(cumulative.index, 1, cumulative.values,
                    where=(cumulative.values >= 1), alpha=0.15, color="green")
    ax.fill_between(cumulative.index, 1, cumulative.values,
                    where=(cumulative.values < 1), alpha=0.15, color="red")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
) -> None:
    """Plot a styled confusion matrix."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["HOLD", "BUY"], yticklabels=["HOLD", "BUY"],
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_feature_importance(
    importances: pd.Series,
    top_n: int = 20,
    title: str = "Top Feature Importances",
    save_path: Optional[Path] = None,
) -> None:
    """Horizontal bar chart of top-N feature importances."""
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    top.plot.barh(ax=ax, color="#4CAF50")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_signal_distribution(
    signals_df: pd.DataFrame,
    proba_col: str = "ensemble_proba",
    save_path: Optional[Path] = None,
) -> None:
    """Histogram of ensemble probability scores across the universe."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(signals_df[proba_col].dropna(), bins=50, color="#9C27B0", alpha=0.8, edgecolor="white")
    ax.axvline(0.65, color="red", linestyle="--", label="Threshold (0.65)")
    ax.set_title("Signal Probability Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Ensemble Probability")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def format_signal_table(signals_df: pd.DataFrame) -> str:
    """Pretty-print a signal DataFrame for console output."""
    if signals_df.empty:
        return "No signals generated."
    cols = [c for c in ["ticker", "date", "ensemble_proba", "signal"] if c in signals_df.columns]
    return signals_df[cols].to_string(index=False)
