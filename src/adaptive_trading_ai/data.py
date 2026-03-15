from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal


@dataclass
class DataLoader:
    """Prepares multi-timeframe OHLCV data for modeling."""

    timeframe: str = "1D"
    wavelet_denoise: bool = False
    detrend: bool = True
    zscore_window: int = 100
    vol_window: int = 30

    def load_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        missing = required.difference(frame.columns.str.lower())
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = frame.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_index()

        if self.timeframe:
            df = self._resample(df, self.timeframe)

        df["pct_return"] = df["close"].pct_change().fillna(0.0)
        df["log_return"] = np.log(df["close"]).diff().fillna(0.0)
        df["rolling_volatility"] = df["log_return"].rolling(self.vol_window).std().bfill()
        df["normalized_return"] = df["pct_return"] / (df["rolling_volatility"] + 1e-8)

        if self.detrend:
            df["detrended_return"] = signal.detrend(df["normalized_return"].values)
        else:
            df["detrended_return"] = df["normalized_return"]

        mean = df["detrended_return"].rolling(self.zscore_window).mean().bfill()
        std = df["detrended_return"].rolling(self.zscore_window).std().bfill().replace(0, np.nan)
        df["zscore_return"] = ((df["detrended_return"] - mean) / (std + 1e-8)).fillna(0.0)

        if self.wavelet_denoise:
            try:
                import pywt

                coeffs = pywt.wavedec(df["zscore_return"].values, "db4", level=3)
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                uthresh = sigma * np.sqrt(2 * np.log(len(df)))
                coeffs[1:] = [pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs[1:]]
                df["zscore_return"] = pywt.waverec(coeffs, "db4")[: len(df)]
            except ImportError:
                pass

        df["vol_adjusted_return"] = df["zscore_return"] / (df["rolling_volatility"] + 1e-8)
        return df

    @staticmethod
    def _resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        ohlc = df[["open", "high", "low", "close"]].resample(timeframe).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        )
        vol = df[["volume"]].resample(timeframe).sum()
        merged = pd.concat([ohlc, vol], axis=1).dropna()
        return merged
