import pytest
flask = pytest.importorskip("flask")

import pandas as pd

from adaptive_trading_ai.ui import create_app


class DummyBacktester:
    def __init__(self):
        self.train_size = 260
        self.test_size = 40


class DummyFramework:
    def __init__(self):
        self.backtester = DummyBacktester()

    def run(self, raw_df: pd.DataFrame, target_col: str = "pct_return"):
        metrics = {
            "sharpe_ratio": 1.25,
            "cumulative_return": 0.18,
            "win_rate": 0.55,
            "max_drawdown": -0.07,
            "turnover": 0.14,
            "meta_updates": 3,
        }
        return pd.DataFrame(), metrics, {}


def test_dashboard_and_run(tmp_path):
    app = create_app(registry_path=tmp_path / "bots.json", framework_factory=DummyFramework)
    client = app.test_client()

    resp = client.get("/")
    assert resp.status_code == 200
    assert b"No bots saved yet" in resp.data

    run_resp = client.post(
        "/run",
        data={"name": "Alpha", "periods": "420", "train_size": "200", "test_size": "40"},
        follow_redirects=True,
    )
    assert run_resp.status_code == 200
    assert b"Ran Alpha successfully" in run_resp.data
    assert b"Alpha" in run_resp.data
    assert b"1.250" in run_resp.data