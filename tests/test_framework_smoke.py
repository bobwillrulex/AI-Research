import numpy as np
import pandas as pd

from adaptive_trading_ai.framework import build_default_framework


def test_framework_smoke_runs():
    n = 1400
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    t = np.arange(n)
    close = 100 + np.cumsum(0.02 * np.sin(2 * np.pi * t / 40) + 0.01 * np.random.default_rng(42).normal(size=n))
    frame = pd.DataFrame(
        {
            "open": close + 0.01,
            "high": close + 0.03,
            "low": close - 0.03,
            "close": close,
            "volume": 1000 + 10 * np.sin(2 * np.pi * t / 20),
        },
        index=idx,
    )

    fw = build_default_framework()
    bt_df, metrics, artifacts = fw.run(frame)

    assert not bt_df.empty
    assert "sharpe_ratio" in metrics
    assert "spectral" in artifacts
