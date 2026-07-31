"""
model.py  –  Unified XGBoost + LightGBM ensemble pipeline for Nifty 500.

Functions
---------
train_model(features_df)                     -> dict of trained models + artifacts
evaluate_model(features_df)                  -> metrics dict
predict_single(symbol, date, features_df)    -> prediction dict
"""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
OUTPUTS_DIR = Path("outputs")
XGB_MODEL_PATH    = OUTPUTS_DIR / "xgb_model.pkl"
LGB_MODEL_PATH    = OUTPUTS_DIR / "lgb_model.pkl"
FEATURE_COLS_PATH = OUTPUTS_DIR / "feature_columns.json"
LABEL_ENC_PATH    = OUTPUTS_DIR / "label_encoder.pkl"
SHAP_PLOT_PATH    = OUTPUTS_DIR / "shap_importance.png"
METRICS_PATH      = OUTPUTS_DIR / "eval_metrics.json"

# ─────────────────────────────────────────────────────────────────────────────
# Column helpers
# ─────────────────────────────────────────────────────────────────────────────

# Raw / target / meta columns – never used as features
_EXCLUDE = {
    "symbol", "Open", "High", "Low", "Close", "Volume", "Adj_Close",
    "forward_return_5d", "target_label",
    "pattern_name", "pattern_direction", "pattern_type", "prior_trend",
}

# Binary (0/1) columns – passed through without scaling
_BINARY_PREFIXES = ("pat_", "gap_up", "gap_down", "volume_spike", "volume_trend",
                    "rsi_oversold", "rsi_overbought", "higher_highs", "lower_lows",
                    "volatility_expansion", "volatility_contraction",
                    "breakout_strength", "candle_direction", "pattern_direction_num",
                    "pattern_candle_count", "macd_crossover")


def _is_binary(col: str) -> bool:
    for prefix in _BINARY_PREFIXES:
        if col.startswith(prefix) or col == prefix:
            return True
    return False


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _EXCLUDE
            and pd.api.types.is_numeric_dtype(df[c])]


# ─────────────────────────────────────────────────────────────────────────────
# Time-based train / test split
# ─────────────────────────────────────────────────────────────────────────────

def _time_split(
    df: pd.DataFrame,
    train_months: int = 18,
    test_months:  int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time – no shuffling. Adapts test window if data is short."""
    df = df.sort_index()
    # Adaptively shrink test window if total data < train+test months
    total_days = (df.index.max() - df.index.min()).days
    total_months = total_days / 30
    if total_months < (train_months + test_months):
        test_months = max(1, int(total_months * 0.25))  # use 25% as test
        logger.warning("Short data (%.1f mo) - using test_months=%d", total_months, test_months)
    cutoff = df.index.max() - pd.DateOffset(months=test_months)
    train  = df[df.index <= cutoff]
    test   = df[df.index >  cutoff]
    logger.info(
        "Split: train %s -> %s (%d rows) | test %s -> %s (%d rows)",
        train.index.min().date() if not train.empty else "N/A",
        train.index.max().date() if not train.empty else "N/A",
        len(train),
        test.index.min().date()  if not test.empty  else "N/A",
        test.index.max().date()  if not test.empty  else "N/A",
        len(test),
    )
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class FeaturePreprocessor:
    """Scales continuous features; leaves binary columns untouched."""

    def __init__(self):
        self.scaler      = StandardScaler()
        self.cont_cols:  List[str] = []
        self.bin_cols:   List[str] = []
        self.all_cols:   List[str] = []

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        self.all_cols  = X.columns.tolist()
        self.cont_cols = [c for c in self.all_cols if not _is_binary(c)]
        self.bin_cols  = [c for c in self.all_cols if     _is_binary(c)]
        if self.cont_cols:
            self.scaler.fit(X[self.cont_cols])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.all_cols].copy()
        if self.cont_cols:
            X[self.cont_cols] = self.scaler.transform(X[self.cont_cols])
        return X.values.astype(np.float32)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_xgb(num_class: int, class_weights: np.ndarray) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective        = "multi:softprob",
        num_class        = num_class,
        n_estimators     = 500,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        eval_metric      = "mlogloss",
        early_stopping_rounds = 50,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
    )


def _build_lgb(num_class: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective        = "multiclass",
        num_class        = num_class,
        n_estimators     = 500,
        max_depth        = 6,
        learning_rate    = 0.05,
        num_leaves       = 63,
        feature_fraction = 0.8,
        bagging_fraction = 0.8,
        bagging_freq     = 1,
        early_stopping_round = 50,
        random_state     = 42,
        n_jobs           = -1,
        verbose          = -1,
        class_weight     = "balanced",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SHAP importance plot
# ─────────────────────────────────────────────────────────────────────────────

def _save_shap_plot(
    xgb_model: xgb.XGBClassifier,
    X_val: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
) -> None:
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        explainer = shap.TreeExplainer(xgb_model)
        sample    = X_val[: min(300, len(X_val))]
        shap_out  = explainer(sample)   # Explanation object (modern API)

        # shap_out.values shape: (n_samples, n_features, n_classes) for multiclass
        vals = shap_out.values
        if vals.ndim == 3:
            # Average absolute SHAP across all classes
            mean_abs = np.abs(vals).mean(axis=(0, 2))  # (n_features,)
        elif vals.ndim == 2:
            mean_abs = np.abs(vals).mean(axis=0)
        else:
            mean_abs = np.abs(vals)

        importance = pd.Series(mean_abs, index=feature_names)
        top        = importance.nlargest(top_n).sort_values()

        fig, ax = plt.subplots(figsize=(10, 8))
        top.plot.barh(ax=ax, color="#4C72B0", edgecolor="white")
        ax.set_title(f"SHAP - Top {top_n} Features", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean |SHAP value|")
        plt.tight_layout()
        fig.savefig(SHAP_PLOT_PATH, dpi=150)
        plt.close(fig)
        logger.info("SHAP plot saved -> %s", SHAP_PLOT_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP plot failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern-wise accuracy helper
# ─────────────────────────────────────────────────────────────────────────────

def _pattern_accuracy(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    le: LabelEncoder,
) -> Dict[str, float]:
    """Accuracy breakdown per candlestick pattern on the test set."""
    from src.pattern_detection import ALL_PATTERN_NAMES

    y_true  = le.transform(test_df["target_label"].values)
    results = {}
    for name in ALL_PATTERN_NAMES:
        col = f"pat_{name}"
        if col not in test_df.columns:
            continue
        mask = test_df[col].values == 1
        if mask.sum() < 5:   # not enough samples
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        results[name] = round(float(acc), 4)
    return dict(sorted(results.items(), key=lambda x: -x[1]))


# ─────────────────────────────────────────────────────────────────────────────
# train_model
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    features_df: pd.DataFrame,
    train_months: int = 18,
    test_months:  int = 6,
    outputs_dir:  Path = OUTPUTS_DIR,
) -> dict:
    """
    Train XGBoost + LightGBM ensemble on combined features DataFrame.

    Parameters
    ----------
    features_df  : Output of engineer_all_features() – contains 'symbol' and
                   'target_label' columns alongside all feature columns.
    train_months : Months of history used for training.
    test_months  : Months held out as test set.
    outputs_dir  : Where models and artefacts are saved.

    Returns
    -------
    dict with keys: xgb_model, lgb_model, preprocessor, label_encoder,
                    feature_cols, train_metrics, test_metrics
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== train_model: %d rows, %d columns ===", *features_df.shape)

    # ── 1. Label encode target (-1, 0, 1) → (0, 1, 2)
    le = LabelEncoder()
    y_all = le.fit_transform(features_df["target_label"].values)
    num_class = len(le.classes_)
    logger.info("Classes: %s  (encoded 0..%d)", le.classes_, num_class - 1)

    # ── 2. Feature columns
    feat_cols = _get_feature_cols(features_df)
    logger.info("Feature columns: %d", len(feat_cols))

    # ── 3. Time split
    train_df, test_df = _time_split(features_df, train_months, test_months)
    if len(train_df) < 100 or len(test_df) < 20:
        raise ValueError(f"Insufficient data: train={len(train_df)}, test={len(test_df)}")

    X_train_raw = train_df[feat_cols]
    X_test_raw  = test_df[feat_cols]
    y_train     = le.transform(train_df["target_label"].values)
    y_test      = le.transform(test_df["target_label"].values)

    # ── 4. Preprocessing (scale continuous, keep binary)
    prep = FeaturePreprocessor()
    X_train = prep.fit_transform(X_train_raw)
    X_test  = prep.transform(X_test_raw)

    # Use last TimeSeriesSplit fold as XGB/LGB eval set
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X_train))
    tr_idx, val_idx = splits[-1]
    X_tr,  X_val  = X_train[tr_idx],  X_train[val_idx]
    y_tr,  y_val  = y_train[tr_idx],  y_train[val_idx]

    # Class weights for XGBoost sample_weight
    from sklearn.utils.class_weight import compute_sample_weight
    sw_tr = compute_sample_weight("balanced", y_tr)

    # ── 5. Train XGBoost
    logger.info("Training XGBoost …")
    xgb_model = _build_xgb(num_class, None)
    xgb_model.fit(
        X_tr, y_tr,
        sample_weight      = sw_tr,
        eval_set           = [(X_val, y_val)],
        verbose            = False,
    )
    logger.info("XGB best iteration: %d", xgb_model.best_iteration)

    # ── 6. Train LightGBM
    logger.info("Training LightGBM …")
    lgb_model = _build_lgb(num_class)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set          = [(X_val, y_val)],
        callbacks         = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    # ── 7. Ensemble predictions on test set
    xgb_proba = xgb_model.predict_proba(X_test)   # (n, 3)
    lgb_proba = lgb_model.predict_proba(X_test)   # (n, 3)
    ens_proba = (xgb_proba + lgb_proba) / 2.0
    y_pred    = np.argmax(ens_proba, axis=1)

    # ── 8. Metrics
    acc  = accuracy_score(y_test, y_pred)
    cr   = classification_report(y_test, y_pred,
                                 target_names=[str(c) for c in le.classes_],
                                 output_dict=True)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    try:
        auc = roc_auc_score(y_test, ens_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    pat_acc = _pattern_accuracy(test_df, y_pred, le)

    test_metrics = {
        "accuracy"      : round(float(acc), 4),
        "roc_auc_macro" : round(float(auc), 4),
        "classification_report": cr,
        "confusion_matrix"     : cm,
        "pattern_accuracy"     : pat_acc,
    }

    logger.info("Test accuracy: %.4f | ROC-AUC (macro OvR): %.4f", acc, auc)
    logger.info("\n%s", classification_report(
        y_test, y_pred, target_names=[str(c) for c in le.classes_]))

    # ── 9. SHAP plot
    _save_shap_plot(xgb_model, X_test, feat_cols)

    # ── 10. Save artefacts
    joblib.dump(xgb_model,  outputs_dir / "xgb_model.pkl")
    joblib.dump(lgb_model,  outputs_dir / "lgb_model.pkl")
    joblib.dump(le,         outputs_dir / "label_encoder.pkl")
    joblib.dump(prep,       outputs_dir / "preprocessor.pkl")

    with open(outputs_dir / "feature_columns.json", "w") as fh:
        json.dump(feat_cols, fh, indent=2)
    with open(outputs_dir / "eval_metrics.json", "w") as fh:
        json.dump(test_metrics, fh, indent=2, default=str)

    logger.info("All artefacts saved to %s", outputs_dir)

    return {
        "xgb_model"     : xgb_model,
        "lgb_model"     : lgb_model,
        "preprocessor"  : prep,
        "label_encoder" : le,
        "feature_cols"  : feat_cols,
        "test_metrics"  : test_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    features_df: pd.DataFrame,
    test_months: int = 6,
    outputs_dir: Path = OUTPUTS_DIR,
) -> dict:
    """
    Load saved models and evaluate on the held-out test window.
    Prints a full metrics report to stdout.
    """
    # Load artefacts
    xgb_model = joblib.load(outputs_dir / "xgb_model.pkl")
    lgb_model = joblib.load(outputs_dir / "lgb_model.pkl")
    le        = joblib.load(outputs_dir / "label_encoder.pkl")
    prep      = joblib.load(outputs_dir / "preprocessor.pkl")
    with open(outputs_dir / "feature_columns.json") as fh:
        feat_cols = json.load(fh)

    # Test slice
    features_df = features_df.sort_index()
    cutoff   = features_df.index.max() - pd.DateOffset(months=test_months)
    test_df  = features_df[features_df.index > cutoff]

    missing = [c for c in feat_cols if c not in test_df.columns]
    if missing:
        logger.warning("Missing feature columns in evaluation data: %s", missing)
        for c in missing:
            test_df[c] = 0.0

    X_test = prep.transform(test_df[feat_cols])
    y_test = le.transform(test_df["target_label"].values)

    xgb_proba = xgb_model.predict_proba(X_test)
    lgb_proba = lgb_model.predict_proba(X_test)
    ens_proba = (xgb_proba + lgb_proba) / 2.0
    y_pred    = np.argmax(ens_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, ens_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    cm      = confusion_matrix(y_test, y_pred)
    pat_acc = _pattern_accuracy(test_df, y_pred, le)

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Test rows        : {len(test_df)}")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  ROC-AUC (macro)  : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(
        y_test, y_pred, target_names=[str(c) for c in le.classes_]
    ))
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  Classes: {le.classes_.tolist()}")
    print(cm)
    print("\n  Pattern-wise Accuracy:")
    for pat, a in pat_acc.items():
        count = int(test_df[f"pat_{pat}"].sum()) if f"pat_{pat}" in test_df.columns else 0
        print(f"    {pat:<28} {a:.4f}  {count} rows")
    
    # ── Additional metrics (FIX 3) ──
    print("\n  Pattern-row Accuracy Split:")
    pattern_mask = (test_df["pattern_strength"] > 0).values
    if pattern_mask.sum() > 0:
        pat_acc_val = accuracy_score(y_test[pattern_mask], y_pred[pattern_mask])
        print(f"    Accuracy on pattern rows  : {pat_acc_val:.4f}  ({int(pattern_mask.sum())} rows)")
    else:
        print("    Accuracy on pattern rows  : N/A  (0 rows)")
        
    no_pattern_mask = ~pattern_mask
    if no_pattern_mask.sum() > 0:
        nopat_acc_val = accuracy_score(y_test[no_pattern_mask], y_pred[no_pattern_mask])
        print(f"    Accuracy on no-pattern    : {nopat_acc_val:.4f}  ({int(no_pattern_mask.sum())} rows)")
    else:
        print("    Accuracy on no-pattern    : N/A  (0 rows)")

    print("\n  Confidence-tier Accuracy:")
    conf = np.max(ens_proba, axis=1)
    
    high_mask = conf >= 0.70
    med_mask = (conf >= 0.55) & (conf < 0.70)
    low_mask = conf < 0.55
    
    high_acc = accuracy_score(y_test[high_mask], y_pred[high_mask]) if high_mask.sum() > 0 else float("nan")
    med_acc = accuracy_score(y_test[med_mask], y_pred[med_mask]) if med_mask.sum() > 0 else float("nan")
    low_acc = accuracy_score(y_test[low_mask], y_pred[low_mask]) if low_mask.sum() > 0 else float("nan")
    
    print(f"    HIGH (>=0.70)             : {high_acc:.4f}  ({int(high_mask.sum())} rows)" if not np.isnan(high_acc) else f"    HIGH (>=0.70)             : N/A  ({int(high_mask.sum())} rows)")
    print(f"    MEDIUM (0.55-0.70)        : {med_acc:.4f}  ({int(med_mask.sum())} rows)" if not np.isnan(med_acc) else f"    MEDIUM (0.55-0.70)        : N/A  ({int(med_mask.sum())} rows)")
    print(f"    LOW (<0.55)               : {low_acc:.4f}  ({int(low_mask.sum())} rows)" if not np.isnan(low_acc) else f"    LOW (<0.55)               : N/A  ({int(low_mask.sum())} rows)")
    
    print("=" * 60 + "\n")

    metrics = {
        "accuracy"         : round(float(acc), 4),
        "roc_auc_macro"    : round(float(auc), 4),
        "confusion_matrix" : cm.tolist(),
        "pattern_accuracy" : pat_acc,
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# predict_single
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(
    symbol:      str,
    date:        str,
    features_df: pd.DataFrame,
    outputs_dir: Path = OUTPUTS_DIR,
) -> dict:
    """
    Return the ensemble prediction for a single (symbol, date) pair.

    Parameters
    ----------
    symbol      : e.g. "TCS.NS"
    date        : e.g. "2026-05-20"
    features_df : Combined feature DataFrame (output of engineer_all_features).
    outputs_dir : Where artefacts are stored.

    Returns
    -------
    dict with keys:
        symbol, date, predicted_label, predicted_class,
        prob_bearish, prob_neutral, prob_bullish,
        xgb_label, lgb_label, confidence
    """
    xgb_model = joblib.load(outputs_dir / "xgb_model.pkl")
    lgb_model = joblib.load(outputs_dir / "lgb_model.pkl")
    le        = joblib.load(outputs_dir / "label_encoder.pkl")
    prep      = joblib.load(outputs_dir / "preprocessor.pkl")
    with open(outputs_dir / "feature_columns.json") as fh:
        feat_cols = json.load(fh)

    # Filter to symbol + date
    ts = pd.Timestamp(date)
    if "symbol" in features_df.columns:
        row_df = features_df[
            (features_df["symbol"] == symbol) &
            (features_df.index == ts)
        ]
    else:
        row_df = features_df[features_df.index == ts]

    if row_df.empty:
        return {
            "symbol": symbol, "date": date,
            "error": f"No data found for {symbol} on {date}",
        }

    row_df = row_df.iloc[[0]]
    for c in feat_cols:
        if c not in row_df.columns:
            row_df[c] = 0.0

    X = prep.transform(row_df[feat_cols])

    xgb_proba = xgb_model.predict_proba(X)[0]   # (3,)
    lgb_proba = lgb_model.predict_proba(X)[0]   # (3,)
    ens_proba = (xgb_proba + lgb_proba) / 2.0

    pred_idx   = int(np.argmax(ens_proba))
    pred_label = int(le.inverse_transform([pred_idx])[0])
    xgb_label  = int(le.inverse_transform([int(np.argmax(xgb_proba))])[0])
    lgb_label  = int(le.inverse_transform([int(np.argmax(lgb_proba))])[0])
    confidence = float(ens_proba[pred_idx])

    class_map  = {-1: "bearish", 0: "neutral", 1: "bullish"}
    # Map class indices to original labels
    classes    = le.classes_.tolist()   # e.g. [-1, 0, 1]

    result = {
        "symbol"          : symbol,
        "date"            : date,
        "predicted_label" : pred_label,
        "predicted_class" : class_map.get(pred_label, str(pred_label)),
        "prob_bearish"    : round(float(ens_proba[classes.index(-1)]), 4) if -1 in classes else None,
        "prob_neutral"    : round(float(ens_proba[classes.index(0)]),  4) if  0 in classes else None,
        "prob_bullish"    : round(float(ens_proba[classes.index(1)]),  4) if  1 in classes else None,
        "xgb_label"       : xgb_label,
        "lgb_label"       : lgb_label,
        "confidence"      : round(confidence, 4),
    }
    logger.info("predict_single: %s @ %s -> %s (conf=%.3f)",
                symbol, date, result["predicted_class"], confidence)
    return result


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

    _FCSV = "data/features/combined_features.csv"
    _ODIR = "outputs"

    parser = argparse.ArgumentParser(description="Train / evaluate the trading model.")
    sub    = parser.add_subparsers(dest="cmd")

    train_p = sub.add_parser("train", help="Train both models")
    train_p.add_argument("--features-csv", default=_FCSV)
    train_p.add_argument("--outputs-dir",  default=_ODIR)

    eval_p = sub.add_parser("evaluate", help="Evaluate saved models")
    eval_p.add_argument("--features-csv", default=_FCSV)
    eval_p.add_argument("--outputs-dir",  default=_ODIR)

    pred_p = sub.add_parser("predict", help="Single prediction")
    pred_p.add_argument("--features-csv", default=_FCSV)
    pred_p.add_argument("--outputs-dir",  default=_ODIR)
    pred_p.add_argument("--symbol", required=True)
    pred_p.add_argument("--date",   required=True)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(0)

    logger.info("Loading features from %s ...", args.features_csv)
    df  = pd.read_csv(args.features_csv, index_col=0, parse_dates=True)
    out = Path(args.outputs_dir)

    if args.cmd == "train":
        train_model(df, outputs_dir=out)
    elif args.cmd == "evaluate":
        evaluate_model(df, outputs_dir=out)
    elif args.cmd == "predict":
        result = predict_single(args.symbol, args.date, df, outputs_dir=out)
        print(json.dumps(result, indent=2))
