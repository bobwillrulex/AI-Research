import json

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
        bt = pd.DataFrame({"equity": [1.0, 1.03]})
        return bt, metrics, {"note": "dummy"}


class PartiallyFailingFramework(DummyFramework):
    def run(self, raw_df: pd.DataFrame, target_col: str = "pct_return"):
        if len(raw_df) % 2 == 0:
            raise RuntimeError("simulated failure")
        return super().run(raw_df, target_col)


def test_dashboard_run_leaderboard_and_detail(tmp_path):
    app = create_app(
        registry_path=tmp_path / "bots.json",
        artifacts_dir=tmp_path / "artifacts",
        framework_factory=DummyFramework,
    )
    client = app.test_client()

    resp = client.get("/")
    assert resp.status_code == 200
    assert b"No bots saved yet" in resp.data

    run_resp = client.post(
        "/run",
        data={
            "name": "Alpha",
            "symbols": "AAPL,MSFT",
            "data_source": "synthetic",
            "periods": "421",
            "train_size": "200",
            "test_size": "40",
            "missing_policy": "drop",
        },
        follow_redirects=True,
    )
    assert run_resp.status_code == 200
    assert b"Ran 2/2 bot(s) successfully" in run_resp.data
    assert b"Alpha-AAPL" in run_resp.data
    assert b"Alpha-MSFT" in run_resp.data

    registry_items = json.loads((tmp_path / "bots.json").read_text())
    assert len(registry_items) == 2
    detail_run_id = registry_items[0]["run_id"]

    detail_resp = client.get(f"/bots/{detail_run_id}")
    assert detail_resp.status_code == 200
    assert b"Run Diagnostics" in detail_resp.data
    assert b"dummy" in detail_resp.data


def test_validation_returns_400(tmp_path):
    app = create_app(
        registry_path=tmp_path / "bots.json",
        artifacts_dir=tmp_path / "artifacts",
        framework_factory=DummyFramework,
    )
    client = app.test_client()

    resp = client.post(
        "/run",
        data={
            "name": "Alpha",
            "symbols": "BAD SYMBOL",
            "data_source": "synthetic",
            "periods": "120",
            "train_size": "100",
            "test_size": "40",
        },
    )
    assert resp.status_code == 400
    assert b"Invalid ticker format" in resp.data
    assert b"Train size + test size must be &lt;= periods." in resp.data


def test_partial_symbol_failure_keeps_successes(tmp_path):
    app = create_app(
        registry_path=tmp_path / "bots.json",
        artifacts_dir=tmp_path / "artifacts",
        framework_factory=PartiallyFailingFramework,
    )
    client = app.test_client()

    resp = client.post(
        "/run",
        data={
            "name": "Alpha",
            "symbols": "AAPL",
            "data_source": "synthetic",
            "periods": "421",
            "train_size": "200",
            "test_size": "40",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Ran 1/1 bot(s) successfully" in resp.data

    resp2 = client.post(
        "/run",
        data={
            "name": "Alpha",
            "symbols": "MSFT",
            "data_source": "synthetic",
            "periods": "420",
            "train_size": "200",
            "test_size": "40",
        },
        follow_redirects=True,
    )
    assert resp2.status_code == 200
    assert b"Ran 0/1 bot(s) successfully" in resp2.data
    assert b"Failures:" in resp2.data

    registry_items = json.loads((tmp_path / "bots.json").read_text())
    assert len(registry_items) == 1
