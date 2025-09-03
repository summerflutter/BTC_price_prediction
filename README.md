# BTC Return Forecasting — Walk-Forward, ML + Deep Learning

A modular template to forecast Bitcoin **returns** (daily/hourly/minute) and compare classical ML models and deep learning models with **walk-forward** evaluation and per-window hyperparameter tuning.

## Highlights
- Clean feature pipeline with **lagged returns** + **technical indicators**.
- Optional merge of **exogenous assets** (equities, rates, gold, etc.).
- Unified interface for: Ridge, RandomForest, XGBoost*; LSTM & Transformer (PyTorch).
- **Strict no-leakage** scaling and windowing.
- Consecutive **Train / Val / Test** splits in **walk-forward** fashion; tune on Val, refit on Train+Val, test out-of-sample.
- Metrics: **R²** (main), MSE/MAE, Directional Accuracy, hitrate. Plots for expanding/cumulative R² and predictions vs. actuals.
- Config-driven (`configs/example_daily.yaml`).

\*XGBoost is optional. If not installed, it is skipped gracefully.

## Quickstart
1. Create a CSV with at least two columns: `timestamp` (ISO8601) and `price` (BTC close). See `data/example_btc.csv` for format (placeholder).
2. Adjust `configs/example_daily.yaml`: point `data.main_csv` to your file; set frequency (`D` or `H`) and windows.
3. Install deps (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```
4. Run the pipeline:
   ```bash
   python -m scripts.run_pipeline --config configs/example_daily.yaml
   ```
5. Results (predictions, metrics, plots) land in `results/<run_id>/`.

## Notes
- For minute data, ensure you downsample or cap sequence lengths reasonably. Consider using returns at **k-minute** intervals to limit noise.
- Keep features computed from **information available at time t** only. This template uses `shift(1)` on returns by default to avoid look-ahead.
- You can register your own models in `src/models/registry.py`.
- For heavy tuning, consider Optuna; a simple random search is provided.

---

**Educational purpose only.** Not financial advice.
