from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from .backtest import WalkForwardBacktester
from .data import DataLoader
from .features import FeatureEngineer
from .meta import MetaOptimizer
from .regime import RegimeDetector
from .signals import FourierAnalyzer, HilbertCycleDetector, KalmanCycleTracker, SignalProcessor, WaveletAnalyzer


@dataclass
class AdaptiveTradingFramework:
    data_loader: DataLoader
    fourier: FourierAnalyzer
    wavelet: WaveletAnalyzer
    hilbert: HilbertCycleDetector
    kalman: KalmanCycleTracker
    feature_engineer: FeatureEngineer
    regime_detector: RegimeDetector
    backtester: WalkForwardBacktester
    meta_optimizer: MetaOptimizer

    def run(self, raw_df: pd.DataFrame, target_col: str = "pct_return") -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, pd.DataFrame]]:
        processed = self.data_loader.load_frame(raw_df)
        regime = self.regime_detector.detect(processed)

        returns = processed["vol_adjusted_return"]
        spectral = self.fourier.rolling_fft(returns)
        hilbert = self.hilbert.detect(returns)
        kalman = self.kalman.track(returns)
        _wavelet = self.wavelet.transform(returns)

        features = self.feature_engineer.build_features(processed, spectral, hilbert, kalman, regime)

        base_config = {"lr": 1e-3, "hidden_dim": 64, "layers": 2, "heads": 4, "l2": 1e-4}
        bt_df, metrics = self.backtester.run(features, target_col=target_col, model_config=base_config, meta_optimizer=self.meta_optimizer)

        if not spectral.empty:
            last = spectral.iloc[-1]
            reconstructed = self.fourier.reconstruct_signal(
                returns.loc[spectral.index],
                frequencies=last["harmonic_frequencies"],
                amplitudes=last["harmonic_amplitudes"],
            )
        else:
            reconstructed = returns.copy()

        artifacts = {
            "processed": processed,
            "features": features,
            "spectral": spectral,
            "hilbert": hilbert,
            "kalman": kalman,
            "regime": regime,
            "reconstructed": reconstructed.to_frame(name="reconstructed_cycle"),
        }
        return bt_df, metrics, artifacts


def build_default_framework() -> AdaptiveTradingFramework:
    return AdaptiveTradingFramework(
        data_loader=DataLoader(),
        fourier=FourierAnalyzer(),
        wavelet=WaveletAnalyzer(),
        hilbert=HilbertCycleDetector(),
        kalman=KalmanCycleTracker(),
        feature_engineer=FeatureEngineer(),
        regime_detector=RegimeDetector(),
        backtester=WalkForwardBacktester(),
        meta_optimizer=MetaOptimizer(),
    )
