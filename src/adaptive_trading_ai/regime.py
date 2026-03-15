from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


@dataclass
class RegimeDetector:
    n_regimes: int = 4
    covariance_type: str = "full"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[["pct_return", "rolling_volatility", "normalized_return"]].fillna(0.0).values
        gmm = GaussianMixture(n_components=self.n_regimes, covariance_type=self.covariance_type, random_state=42)
        gmm.fit(features)
        probs = gmm.predict_proba(features)
        labels = probs.argmax(axis=1)

        regime_names = self._map_regimes(labels, df["rolling_volatility"].values, df["pct_return"].values)
        out = pd.DataFrame(index=df.index)
        out["regime_label"] = labels
        out["regime_name"] = regime_names
        for i in range(self.n_regimes):
            out[f"regime_prob_{i}"] = probs[:, i]
        return out

    @staticmethod
    def _map_regimes(labels: np.ndarray, vol: np.ndarray, returns: np.ndarray) -> np.ndarray:
        names = np.empty_like(labels, dtype=object)
        for lbl in np.unique(labels):
            idx = labels == lbl
            mean_ret = returns[idx].mean()
            mean_vol = vol[idx].mean()
            if mean_vol > np.nanpercentile(vol, 70):
                regime = "high_volatility"
            elif mean_vol < np.nanpercentile(vol, 30):
                regime = "low_volatility"
            elif mean_ret > 0:
                regime = "trending"
            else:
                regime = "mean_reverting"
            names[idx] = regime
        return names
