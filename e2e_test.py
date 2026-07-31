"""
e2e_test.py  --  End-to-end validation of the Nifty 500 trading pipeline.
Run: .venv\Scripts\python e2e_test.py
"""
import sys, logging, warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING,           # quiet sub-libs
    format="%(levelname)-8s | %(name)s | %(message)s")
logging.getLogger("run_pipeline").setLevel(logging.INFO)

from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR       = Path("data/raw")
PROC_DIR      = Path("data/processed")
FEAT_DIR      = Path("data/features")
OUT_DIR       = Path("outputs")
REPORTS_DIR   = OUT_DIR / "reports"

SYMS_20 = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "AXISBANK.NS","SBIN.NS","WIPRO.NS","LT.NS","MARUTI.NS",
    "BAJFINANCE.NS","TITAN.NS","NESTLEIND.NS","HINDUNILVR.NS","ASIANPAINT.NS",
    "SUNPHARMA.NS","DRREDDY.NS","ONGC.NS","NTPC.NS","POWERGRID.NS",
]

SEP  = "=" * 62
SEP2 = "-" * 62

def hdr(title): print(f"\n{SEP}\n  {title}\n{SEP}")
def ok(msg):    print(f"  [OK]  {msg}")
def warn(msg):  print(f"  [!!]  {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load raw data
# ─────────────────────────────────────────────────────────────────────────────
hdr("STEP 1 -- Raw data ingestion check")
from src.data_ingestion import load_all_data
raw_data = load_all_data(raw_dir=RAW_DIR, min_trading_days=400)
raw_data = {s: df for s, df in raw_data.items() if s in SYMS_20}
print(f"  Stocks loaded: {len(raw_data)} / 20")
for sym, df in raw_data.items():
    print(f"    {sym:<18} {len(df):>4} rows  "
          f"{str(df.index.min().date())} -> {str(df.index.max().date())}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Preprocessing validation
# ─────────────────────────────────────────────────────────────────────────────
hdr("STEP 2 -- Preprocessing + data quality checks")
from src.preprocessing import preprocess_one
processed = {}
price_cols = ["Open","High","Low","Close","Volume","Adj_Close"]

for sym, df in raw_data.items():
    raw_rows = len(df)
    clean = preprocess_one(df, symbol=sym, processed_dir=PROC_DIR, save=True)
    if clean is None:
        warn(f"{sym}: preprocessing returned None")
        continue

    issues = []
    nan_count = clean.isnull().sum().sum()
    if nan_count: issues.append(f"{nan_count} NaNs")

    inf_count = np.isinf(clean.select_dtypes(include=np.number)).sum().sum()
    if inf_count: issues.append(f"{inf_count} Infs")

    neg_price = (clean[["Open","High","Low","Close"]] <= 0).sum().sum()
    if neg_price: issues.append(f"{neg_price} non-positive prices")

    # Check date continuity (no gaps > 5 calendar days on weekdays)
    idx = clean.index.sort_values()
    gaps = pd.Series(idx).diff().dt.days.dropna()
    big_gaps = gaps[gaps > 5]
    if len(big_gaps): issues.append(f"{len(big_gaps)} gaps>5d")

    status = "OK" if not issues else "WARN: " + ", ".join(issues)
    print(f"  {sym:<18} raw={raw_rows:>4} -> clean={len(clean):>4}  [{status}]")
    processed[sym] = clean

ok(f"Preprocessing complete: {len(processed)}/{len(raw_data)} stocks passed")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Pattern detection summary
# ─────────────────────────────────────────────────────────────────────────────
hdr("STEP 3 -- Pattern detection across all 20 stocks")
from src.pattern_detection import detect_all_patterns, tag_dataframe, ALL_PATTERN_NAMES

pattern_agg = {}   # name -> list of (body_to_range_ratio, volume_ratio)
for sym, df in processed.items():
    tagged = tag_dataframe(df)
    events = detect_all_patterns(df)
    for ev in events:
        name = ev["pattern_name"]
        row_date = ev["pattern_end_date"]
        if row_date in tagged.index:
            r = tagged.loc[row_date]
            btr = float(r.get("body_to_range_ratio", 0) or 0)
            vr  = float(r.get("volume_ratio", 1)        or 1)
        else:
            btr, vr = 0.0, 1.0
        pattern_agg.setdefault(name, []).append((btr, vr))

print(f"\n  {'Pattern':<26} {'Count':>6} {'Avg B/R':>8} {'Avg VR':>8}")
print(f"  {'-'*26} {'-'*6} {'-'*8} {'-'*8}")
detected_names = set()
for name in ALL_PATTERN_NAMES:
    vals = pattern_agg.get(name, [])
    count = len(vals)
    avg_btr = np.mean([v[0] for v in vals]) if vals else 0
    avg_vr  = np.mean([v[1] for v in vals]) if vals else 0
    flag = "" if count > 0 else "  <-- not detected"
    print(f"  {name:<26} {count:>6} {avg_btr:>8.3f} {avg_vr:>8.3f}{flag}")
    if count > 0:
        detected_names.add(name)

total_events = sum(len(v) for v in pattern_agg.values())
print(f"\n  Total pattern events : {total_events}")
print(f"  Patterns detected    : {len(detected_names)} / {len(ALL_PATTERN_NAMES)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Feature engineering validation
# ─────────────────────────────────────────────────────────────────────────────
hdr("STEP 4 -- Feature engineering validation")
from src.feature_engineering import engineer_features, get_feature_columns
from src.pattern_detection import tag_dataframe

all_feat_frames = []
for sym, df in processed.items():
    tagged = tag_dataframe(df)
    feat = engineer_features(tagged, symbol=sym, features_dir=FEAT_DIR, save=True)
    if feat is not None:
        feat.insert(0, "symbol", sym)
        all_feat_frames.append(feat)

combined = pd.concat(all_feat_frames).sort_index()
combined.to_csv(FEAT_DIR / "combined_features.csv")

feat_cols = get_feature_columns(combined)
print(f"  Feature columns      : {len(feat_cols)}")
print(f"  Total rows           : {len(combined)}")
print(f"  Stocks in combined   : {combined['symbol'].nunique()}")

nan_total = combined[feat_cols].isnull().sum().sum()
inf_total = np.isinf(combined[feat_cols].select_dtypes(include=np.number)).sum().sum()
print(f"  NaN values           : {nan_total}")
print(f"  Inf values           : {inf_total}")
if nan_total or inf_total:
    warn("NaN/Inf found -- dropping")
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols)
    print(f"  Rows after clean     : {len(combined)}")

tgt = combined["target_label"].value_counts(normalize=True).sort_index() * 100
print(f"\n  Target distribution:")
labels = {-1:"bearish", 0:"neutral", 1:"bullish"}
for lbl, pct in tgt.items():
    print(f"    {labels.get(lbl, str(lbl)):<8} ({lbl:+d}) : {pct:>5.1f}%")

ok("Feature engineering validated")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Model training + evaluation
# ─────────────────────────────────────────────────────────────────────────────
hdr("STEP 5 -- Model training (18m train / 6m test)")
from src.model import train_model, evaluate_model
import json

result = train_model(combined, train_months=18, test_months=6, outputs_dir=OUT_DIR)
metrics = result["test_metrics"]

print(f"\n  Accuracy      : {metrics['accuracy']:.4f}")
print(f"  ROC-AUC macro : {metrics['roc_auc_macro']:.4f}")
print(f"\n  Per-class metrics:")
cr = metrics["classification_report"]
print(f"  {'Class':<10} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Supp':>6}")
print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
for cls in ["-1","0","1"]:
    if cls in cr:
        r = cr[cls]
        print(f"  {cls:<10} {r['precision']:>6.3f} {r['recall']:>6.3f} "
              f"{r['f1-score']:>6.3f} {int(r['support']):>6}")

# SHAP top 10
import joblib, shap
import matplotlib; matplotlib.use("Agg")
xgb_m = joblib.load(OUT_DIR / "xgb_model.pkl")
prep  = joblib.load(OUT_DIR / "preprocessor.pkl")
with open(OUT_DIR / "feature_columns.json") as fh:
    fc = json.load(fh)

# Build test slice
cutoff = combined.index.max() - pd.DateOffset(months=6)
test_df = combined[combined.index > cutoff]
X_test  = prep.transform(test_df[fc])

explainer = shap.TreeExplainer(xgb_m)
shap_out  = explainer(X_test[:200])
vals = shap_out.values
if vals.ndim == 3:
    mean_abs = np.abs(vals).mean(axis=(0, 2))
else:
    mean_abs = np.abs(vals).mean(axis=0)

importance = pd.Series(mean_abs, index=fc).nlargest(10)
print(f"\n  Top 10 features by SHAP:")
for feat, val in importance.items():
    bar = "#" * max(1, int(val / importance.max() * 20))
    print(f"    {feat:<32} {val:.5f}  {bar}")

ok("Model training complete")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Signal generation for latest date
# ─────────────────────────────────────────────────────────────────────────────
hdr("STEP 6 -- Signal generation (latest date)")
from src.signals import generate_signal, generate_daily_report

latest_date = str(combined.index.max().date())
latest_df   = combined[combined.index == combined.index.max()]

lgb_m = joblib.load(OUT_DIR / "lgb_model.pkl")
le    = joblib.load(OUT_DIR / "label_encoder.pkl")
classes = le.classes_.tolist()

all_signals = []
for _, row in latest_df.iterrows():
    sym = str(row.get("symbol", "?"))
    for c in fc:
        if c not in row.index:
            row[c] = 0.0
    X = prep.transform(pd.DataFrame([row])[fc])
    xgb_p = xgb_m.predict_proba(X)[0]
    lgb_p = lgb_m.predict_proba(X)[0]
    ens   = (xgb_p + lgb_p) / 2.0
    probs = {
        "bullish_prob": float(ens[classes.index(1)])  if 1  in classes else 0.33,
        "bearish_prob": float(ens[classes.index(-1)]) if -1 in classes else 0.33,
        "neutral_prob": float(ens[classes.index(0)])  if 0  in classes else 0.33,
    }
    row_dict = row.to_dict()
    prow = processed.get(sym)
    if prow is not None and not prow.empty:
        last = prow.sort_index().iloc[-1].to_dict()
        for k in ("Open","High","Low","Close","Volume"):
            row_dict[k] = last.get(k, row_dict.get(k, 0))
    sig = generate_signal(sym, latest_date, row_dict, probs)
    all_signals.append(sig)

all_signals.sort(key=lambda x: x["confidence_value"], reverse=True)
print(f"\n  Date: {latest_date}  |  Signals: {len(all_signals)}")
print(f"\n  {'#':<3} {'Symbol':<16} {'Pattern':<22} {'Dir':<9} "
      f"{'Bull%':>6} {'Bear%':>6} {'Conf':>6} {'Signal':<5} {'Rej'}")
print(f"  {'-'*3} {'-'*16} {'-'*22} {'-'*9} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*3}")
for i, s in enumerate(all_signals[:5], 1):
    rej = "Y" if s["reject_signal"] else "N"
    print(f"  {i:<3} {s['symbol']:<16} {s['pattern_detected']:<22} "
          f"{s['pattern_direction']:<9} "
          f"{s['bullish_probability']*100:>5.1f}% "
          f"{s['bearish_probability']*100:>5.1f}% "
          f"{s['confidence_value']:>6.3f} "
          f"{s['signal']:<5} {rej}")

valid = generate_daily_report(latest_date, all_signals, reports_dir=REPORTS_DIR)
print(f"\n  Valid signals saved  : {len(valid)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Backtest
# ─────────────────────────────────────────────────────────────────────────────
hdr("STEP 7 -- Backtest simulation")
from src.backtest import run_backtest

# Build signals from test period (last 6 months of all stocks)
cutoff = combined.index.max() - pd.DateOffset(months=6)
test_slice = combined[combined.index > cutoff]

rows = []
for _, row in test_slice.iterrows():
    sym = str(row.get("symbol","?"))
    for c in fc:
        if c not in row.index: row[c] = 0.0
    X = prep.transform(pd.DataFrame([row])[fc])
    xgb_p = xgb_m.predict_proba(X)[0]
    lgb_p = lgb_m.predict_proba(X)[0]
    ens   = (xgb_p + lgb_p) / 2.0
    probs = {
        "bullish_prob": float(ens[classes.index(1)])  if 1  in classes else 0.33,
        "bearish_prob": float(ens[classes.index(-1)]) if -1 in classes else 0.33,
        "neutral_prob": float(ens[classes.index(0)])  if 0  in classes else 0.33,
    }
    prow = processed.get(sym)
    row_dict = row.to_dict()
    if prow is not None and not prow.empty:
        last = prow.sort_index().iloc[-1].to_dict()
        for k in ("Open","High","Low","Close","Volume"):
            row_dict[k] = last.get(k, row_dict.get(k, 0))
    sig = generate_signal(sym, str(pd.Timestamp(row.name).date()), row_dict, probs)
    if not sig["reject_signal"] and sig["signal"] in ("BUY","SELL"):
        rows.append({
            "symbol":           sig["symbol"],
            "date":             sig["date"],
            "signal":           sig["signal"],
            "entry_price":      sig["entry_price"],
            "stoploss":         sig["stoploss"],
            "confidence_score": sig["confidence_score"],
            "confidence_value": sig["confidence_value"],
            "pattern_detected": sig["pattern_detected"],
            "pattern_strength": sig["pattern_strength"],
        })

bt_signals = pd.DataFrame(rows) if rows else pd.DataFrame()
print(f"  Backtest signals     : {len(bt_signals)}")

if bt_signals.empty:
    print("  No actionable signals in test period -- confidence threshold not met.")
    print("  This is expected with short history. Check outputs with 2y data.")
else:
    bt_price = {s: df for s, df in processed.items()}
    bt_metrics = run_backtest(
        signals_path    = bt_signals,
        price_data_dict = bt_price,
        initial_capital = 1_000_000,
        max_positions   = 10,
        outputs_dir     = OUT_DIR,
    )

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  END-TO-END TEST COMPLETE")
print(SEP)
