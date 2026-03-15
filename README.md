# Adaptive Trading AI Framework

Research-grade modular Python framework for cycle-aware return forecasting with adaptive meta-learning.

## Highlights
- Data pipeline with OHLCV preprocessing, detrending, rolling z-score normalization, and optional wavelet denoising.
- Adaptive signal stack: rolling FFT, Hilbert transform, wavelet decomposition, bandpass support, and Kalman cycle tracking.
- Feature engineering from cycle/spectral/market/regime domains.
- Regime detection via Gaussian Mixture Models.
- Hybrid predictive modeling: RandomForest + GradientBoosting + PyTorch Spectral Transformer.
- Closed-loop meta-optimization for continuous hyperparameter adaptation.
- Walk-forward backtesting with rolling retraining and common trading metrics.
- Visualization engine for key research diagnostics.

## Install
```bash
pip install numpy pandas scipy pywavelets scikit-learn statsmodels torch matplotlib hmmlearn xgboost lightgbm
```

## Quick start
```python
from adaptive_trading_ai.framework import build_default_framework

framework = build_default_framework()
bt_df, metrics, artifacts = framework.run(raw_ohlcv_df)
print(metrics)
```

## Package structure
- `src/adaptive_trading_ai/data.py` — preprocessing and returns/volatility pipeline.
- `src/adaptive_trading_ai/signals.py` — Fourier/Hilbert/Wavelet/Kalman processing.
- `src/adaptive_trading_ai/features.py` — feature synthesis.
- `src/adaptive_trading_ai/regime.py` — regime labeling with probabilities.
- `src/adaptive_trading_ai/models.py` — classical + deep spectral-attention models.
- `src/adaptive_trading_ai/meta.py` — adaptive hyperparameter optimizer.
- `src/adaptive_trading_ai/backtest.py` — walk-forward simulation and metrics.
- `src/adaptive_trading_ai/viz.py` — research plots.
- `src/adaptive_trading_ai/framework.py` — end-to-end orchestration.


## Master UI (Flask)
Run a browser dashboard that can launch AI runs and track saved bot performance:

```bash
python -m adaptive_trading_ai.ui
```

Then open `http://localhost:5000`.

Features:
- Launch new AI bot runs from the UI.
- Configure synthetic market length and backtest train/test window sizes.
- Persist and view saved bots with Sharpe ratio, cumulative return, win rate, drawdown, turnover, and meta-update count.

Saved runs are stored in `saved_bots.json` in the project root by default.
