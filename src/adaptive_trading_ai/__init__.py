"""Adaptive trading AI framework for cycle-aware return forecasting."""

from .data import DataLoader
from .signals import SignalProcessor, FourierAnalyzer, WaveletAnalyzer, HilbertCycleDetector, KalmanCycleTracker
from .features import FeatureEngineer
from .regime import RegimeDetector
from .models import AdaptiveModel, SpectralTransformer
from .meta import MetaOptimizer
from .backtest import WalkForwardBacktester
from .viz import VisualizationEngine
from .framework import AdaptiveTradingFramework, build_default_framework

__all__ = [
    "DataLoader",
    "SignalProcessor",
    "FourierAnalyzer",
    "WaveletAnalyzer",
    "HilbertCycleDetector",
    "KalmanCycleTracker",
    "FeatureEngineer",
    "RegimeDetector",
    "AdaptiveModel",
    "SpectralTransformer",
    "MetaOptimizer",
    "WalkForwardBacktester",
    "VisualizationEngine",
    "AdaptiveTradingFramework",
    "build_default_framework",
]
