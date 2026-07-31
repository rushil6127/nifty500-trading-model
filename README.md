# NIFTY 500 Candlestick ML Trading Model

An AI-powered swing trading signal generator for Indian stock markets,
built on candlestick pattern recognition + XGBoost/LightGBM ensemble.

## What It Does
- Scans 100 NIFTY 500 stocks every evening after market close
- Detects 15 candlestick patterns (single-day and multi-day)
- Generates probabilistic BUY/SELL/HOLD signals with confidence scores
- Provides entry price, stop-loss, and risk percentage for each signal

## Tech Stack
- Python 3.11
- XGBoost + LightGBM ensemble
- yfinance (NSE data via Yahoo Finance)
- pandas, numpy, scikit-learn, SHAP
- 67 engineered features across price, trend, volatility, volume, momentum

## How to Run

### Setup
pip install -r requirements.txt

### Download Data
python src/data_ingestion.py

### Train Model + Generate Signals
python src/run_pipeline.py --date 2026-05-23 --retrain

### Generate Signals Only (model already trained)
python src/run_pipeline.py --date 2026-05-23

## Signal Output Example
Symbol         Pattern             Signal  Confidence  Entry    SL      Risk
TATAMOTORS.NS  bullish_engulfing   BUY     HIGH        820.45  798.20  2.71%
SUNPHARMA.NS   hammer_in_downtrend BUY     MEDIUM      1205.30 1178.60 2.21%

## Project Structure
src/
  data_ingestion.py    — Downloads OHLCV data for NIFTY 500
  preprocessing.py     — Cleans and adjusts for corporate actions
  pattern_detection.py — Detects 15 candlestick patterns
  feature_engineering.py — 67 ML features including pattern interactions
  model.py             — XGBoost + LightGBM ensemble with walk-forward CV
  signals.py           — Probabilistic signal generation
  backtest.py          — Historical simulation with Indian market rules
  run_pipeline.py      — End-to-end orchestration

## Disclaimer
This project is for educational purposes only.
Not financial advice. Always do your own research before investing.
