from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np
import pandas as pd
from flask import Flask, abort, redirect, render_template_string, request, url_for

from .framework import AdaptiveTradingFramework, build_default_framework


@dataclass
class BotRunRecord:
    run_id: str
    name: str
    symbol: str
    data_source: str
    created_at: str
    sharpe_ratio: float
    cumulative_return: float
    win_rate: float
    max_drawdown: float
    turnover: float
    meta_updates: int
    artifact_path: str


class BotRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[BotRunRecord]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text())
        return [BotRunRecord(**item) for item in raw]

    def get(self, run_id: str) -> BotRunRecord | None:
        for item in self.load():
            if item.run_id == run_id:
                return item
        return None

    def add(self, record: BotRunRecord) -> None:
        records = self.load()
        records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(item) for item in records], indent=2))


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Adaptive Trading AI Master UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }
    h1, h2 { color: #f8fafc; }
    a { color: #93c5fd; }
    .grid { display: grid; gap: 1rem; grid-template-columns: 1fr 2fr; align-items: start; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 1rem; }
    label { display: block; margin: 0.6rem 0 0.2rem; }
    input, select { width: 100%; padding: 0.5rem; border-radius: 4px; border: 1px solid #334155; background: #020617; color: #e2e8f0; }
    button { margin-top: 1rem; padding: 0.65rem 1rem; border: none; background: #2563eb; color: white; border-radius: 5px; cursor: pointer; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; border-bottom: 1px solid #1f2937; padding: 0.5rem; }
    .flash { margin-bottom: 1rem; padding: 0.7rem; background: #064e3b; border: 1px solid #065f46; border-radius: 4px; }
    .hint { color: #94a3b8; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>Adaptive Trading AI Multi-Bot Master UI</h1>
  {% if message %}
    <div class="flash">{{ message }}</div>
  {% endif %}
  <div class="grid">
    <section class="card">
      <h2>Run Bots</h2>
      <form method="post" action="{{ url_for('run_bot') }}">
        <label for="name">Bot family name</label>
        <input id="name" name="name" placeholder="MomentumLab" required />

        <label for="symbols">Ticker selection (comma separated)</label>
        <input id="symbols" name="symbols" placeholder="AAPL,MSFT,BTC-USD" required />

        <label for="data_source">Data source</label>
        <select id="data_source" name="data_source">
          <option value="synthetic" selected>Synthetic</option>
          <option value="csv">CSV path (single file or folder)</option>
          <option value="yfinance">Yahoo Finance (requires yfinance)</option>
        </select>

        <label for="data_path">CSV path (optional)</label>
        <input id="data_path" name="data_path" placeholder="./data or ./data/AAPL.csv" />

        <label for="periods">Synthetic market periods</label>
        <input id="periods" name="periods" type="number" min="320" value="420" required />

        <label for="train_size">Train window</label>
        <input id="train_size" name="train_size" type="number" min="120" value="260" required />

        <label for="test_size">Test window</label>
        <input id="test_size" name="test_size" type="number" min="20" value="40" required />

        <button type="submit">Run One Bot Per Symbol</button>
      </form>
      <p class="hint">Each selected symbol is executed as an independent bot with persisted artifacts.</p>
    </section>

    <section class="card">
      <h2>Leaderboard</h2>
      {% if records %}
      <table>
        <thead>
          <tr>
            <th>Name</th><th>Symbol</th><th>Source</th><th>Created</th><th>Sharpe</th><th>Cumulative Return</th><th>Win Rate</th><th>Max DD</th><th>Turnover</th><th>Meta Updates</th>
          </tr>
        </thead>
        <tbody>
          {% for item in records %}
          <tr>
            <td><a href="{{ url_for('bot_detail', run_id=item.run_id) }}">{{ item.name }}</a></td>
            <td>{{ item.symbol }}</td>
            <td>{{ item.data_source }}</td>
            <td>{{ item.created_at }}</td>
            <td>{{ '%.3f'|format(item.sharpe_ratio) }}</td>
            <td>{{ '%.3f'|format(item.cumulative_return) }}</td>
            <td>{{ '%.3f'|format(item.win_rate) }}</td>
            <td>{{ '%.3f'|format(item.max_drawdown) }}</td>
            <td>{{ '%.3f'|format(item.turnover) }}</td>
            <td>{{ item.meta_updates }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% else %}
        <p>No bots saved yet. Run one from the panel.</p>
      {% endif %}
    </section>
  </div>
</body>
</html>
"""

DETAIL_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Bot Detail</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 1rem; max-width: 1000px; }
    a { color: #93c5fd; }
    pre { background: #020617; padding: 1rem; border-radius: 8px; overflow-x: auto; }
  </style>
</head>
<body>
  <p><a href="{{ url_for('dashboard') }}">← Back to leaderboard</a></p>
  <section class="card">
    <h1>{{ record.name }} ({{ record.symbol }})</h1>
    <p>Source: {{ record.data_source }} | Created: {{ record.created_at }}</p>
    <p>Sharpe: {{ '%.3f'|format(record.sharpe_ratio) }} | Return: {{ '%.3f'|format(record.cumulative_return) }} | Win rate: {{ '%.3f'|format(record.win_rate) }}</p>
    <p>Max DD: {{ '%.3f'|format(record.max_drawdown) }} | Turnover: {{ '%.3f'|format(record.turnover) }} | Meta updates: {{ record.meta_updates }}</p>
    <h2>Persisted Artifact</h2>
    <pre>{{ artifact_json }}</pre>
  </section>
</body>
</html>
"""


def _synthetic_ohlcv(periods: int, seed: int = 42) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=periods, freq="D")
    t = np.arange(periods)
    rng = np.random.default_rng(seed)
    drift = 0.01 * np.sin(2 * np.pi * t / 32)
    noise = 0.02 * rng.normal(size=periods)
    close = 100 + np.cumsum(drift + noise)
    return pd.DataFrame(
        {
            "open": close + 0.02,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 1000 + 20 * np.sin(2 * np.pi * t / 16),
        },
        index=idx,
    )


def _load_csv_ohlcv(symbol: str, data_path: str | None) -> pd.DataFrame:
    if not data_path:
        raise ValueError("CSV path is required for csv data source")
    base = Path(data_path)
    csv_path = base / f"{symbol}.csv" if base.is_dir() else base
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV data not found for {symbol}: {csv_path}")
    df = pd.read_csv(csv_path)
    lower_cols = {c.lower(): c for c in df.columns}
    if "datetime" in lower_cols:
        dt_col = lower_cols["datetime"]
    elif "date" in lower_cols:
        dt_col = lower_cols["date"]
    elif "timestamp" in lower_cols:
        dt_col = lower_cols["timestamp"]
    else:
        dt_col = df.columns[0]
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.set_index(dt_col)
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


def _load_yfinance_ohlcv(symbol: str, periods: int) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("yfinance is not installed") from exc
    period_window = max(periods, 365)
    raw = yf.download(symbol, period=f"{period_window}d", interval="1d", progress=False)
    if raw.empty:
        raise ValueError(f"No yfinance data for {symbol}")
    raw.columns = [str(c).lower() for c in raw.columns]
    return raw[["open", "high", "low", "close", "volume"]].dropna().tail(periods)


def _resolve_ohlcv(symbol: str, data_source: str, periods: int, data_path: str | None) -> pd.DataFrame:
    if data_source == "synthetic":
        seed = abs(hash(symbol)) % 10_000
        return _synthetic_ohlcv(periods=periods, seed=seed)
    if data_source == "csv":
        return _load_csv_ohlcv(symbol=symbol, data_path=data_path)
    if data_source == "yfinance":
        return _load_yfinance_ohlcv(symbol=symbol, periods=periods)
    raise ValueError(f"Unsupported data source: {data_source}")


def _persist_artifact(artifacts_dir: Path, run_id: str, payload: dict) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts_dir / f"{run_id}.json"
    artifact_path.write_text(json.dumps(payload, indent=2, default=str))
    return artifact_path


def create_app(
    registry_path: str | Path = "saved_bots.json",
    artifacts_dir: str | Path = "saved_bot_artifacts",
    framework_factory: Callable[[], AdaptiveTradingFramework] = build_default_framework,
) -> Flask:
    app = Flask(__name__)
    registry = BotRegistry(registry_path)
    artifacts_dir = Path(artifacts_dir)

    @app.get("/")
    def dashboard() -> str:
        records = sorted(registry.load(), key=lambda x: x.sharpe_ratio, reverse=True)
        message = request.args.get("message", "")
        return render_template_string(HTML_TEMPLATE, records=records, message=message)

    @app.get("/bots/<run_id>")
    def bot_detail(run_id: str) -> str:
        record = registry.get(run_id)
        if record is None:
            abort(404)
        artifact_path = Path(record.artifact_path)
        artifact_json = artifact_path.read_text() if artifact_path.exists() else '{"error": "artifact missing"}'
        return render_template_string(DETAIL_TEMPLATE, record=record, artifact_json=artifact_json)

    @app.post("/run")
    def run_bot():
        family_name = request.form.get("name", "Unnamed Bot").strip()
        data_source = request.form.get("data_source", "synthetic").strip().lower()
        symbols = [s.strip().upper() for s in request.form.get("symbols", "").split(",") if s.strip()]
        data_path = request.form.get("data_path", "").strip() or None
        periods = int(request.form.get("periods", "420"))
        train_size = int(request.form.get("train_size", "260"))
        test_size = int(request.form.get("test_size", "40"))

        if not symbols:
            return redirect(url_for("dashboard", message="No symbols supplied."))

        saved = 0
        for symbol in symbols:
            framework = framework_factory()
            framework.backtester.train_size = train_size
            framework.backtester.test_size = test_size

            frame = _resolve_ohlcv(symbol=symbol, data_source=data_source, periods=periods, data_path=data_path)
            bt_df, metrics, artifacts = framework.run(frame)

            run_id = uuid4().hex
            payload = {
                "run_id": run_id,
                "name": f"{family_name}-{symbol}",
                "symbol": symbol,
                "data_source": data_source,
                "metrics": metrics,
                "framework_artifacts": artifacts,
                "backtest_preview": bt_df.tail(20).to_dict(orient="records") if hasattr(bt_df, "tail") else [],
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            artifact_path = _persist_artifact(artifacts_dir=artifacts_dir, run_id=run_id, payload=payload)
            registry.add(
                BotRunRecord(
                    run_id=run_id,
                    name=f"{family_name}-{symbol}",
                    symbol=symbol,
                    data_source=data_source,
                    created_at=payload["created_at"],
                    sharpe_ratio=float(metrics.get("sharpe_ratio", 0.0)),
                    cumulative_return=float(metrics.get("cumulative_return", 0.0)),
                    win_rate=float(metrics.get("win_rate", 0.0)),
                    max_drawdown=float(metrics.get("max_drawdown", 0.0)),
                    turnover=float(metrics.get("turnover", 0.0)),
                    meta_updates=int(metrics.get("meta_updates", 0)),
                    artifact_path=str(artifact_path),
                )
            )
            saved += 1

        return redirect(url_for("dashboard", message=f"Ran {saved} bot(s) successfully."))

    return app


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
