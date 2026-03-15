from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


@dataclass
class FeatureEngineer:
    momentum_window: int = 10
    sharpe_window: int = 50

    def build_features(
        self,
        base_df: pd.DataFrame,
        spectral_df: pd.DataFrame,
        hilbert_df: pd.DataFrame,
        kalman_df: pd.DataFrame,
        regime_df: pd.DataFrame,
    ) -> pd.DataFrame:
        df = base_df.copy()
        df = df.join(hilbert_df, how="left").join(kalman_df, how="left")
        df = df.join(spectral_df[["dominant_frequency", "dominant_strength", "spectral_entropy", "spectral_energy"]], how="left")
        df = df.join(regime_df, how="left")

        df["momentum"] = df["pct_return"].rolling(self.momentum_window).sum()
        df["acceleration"] = df["momentum"].diff()
        df["return_autocorr"] = self._rolling_autocorr(df["pct_return"], lag=1, window=self.sharpe_window)
        roll_mean = df["pct_return"].rolling(self.sharpe_window).mean()
        roll_std = df["pct_return"].rolling(self.sharpe_window).std().replace(0, np.nan)
        df["rolling_sharpe"] = (roll_mean / (roll_std + 1e-8)).fillna(0.0)
        df["cycle_strength"] = df["amplitude"] * df["dominant_strength"]

        for band, col in [(5, "short_cycle"), (20, "medium_cycle"), (60, "long_cycle")]:
            df[col] = df["pct_return"].rolling(band).mean()

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df

    @staticmethod
    def _rolling_autocorr(series: pd.Series, lag: int = 1, window: int = 50) -> pd.Series:
        vals = np.full(len(series), np.nan)
        arr = series.values
        for i in range(window, len(series)):
            chunk = arr[i - window : i]
            vals[i] = acf(chunk, nlags=lag, fft=False)[lag]
        return pd.Series(vals, index=series.index).fillna(0.0)
