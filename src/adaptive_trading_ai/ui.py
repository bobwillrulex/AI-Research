from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template_string, request, url_for

from .framework import AdaptiveTradingFramework, build_default_framework


@dataclass
class BotRunRecord:
    name: str
    created_at: str
    sharpe_ratio: float
    cumulative_return: float
    win_rate: float
    max_drawdown: float
    turnover: float
    meta_updates: int


class BotRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[BotRunRecord]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text())
        return [BotRunRecord(**item) for item in raw]

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
    .grid { display: grid; gap: 1rem; grid-template-columns: 1fr 2fr; align-items: start; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 1rem; }
    label { display: block; margin: 0.6rem 0 0.2rem; }
    input { width: 100%; padding: 0.5rem; border-radius: 4px; border: 1px solid #334155; background: #020617; color: #e2e8f0; }
    button { margin-top: 1rem; padding: 0.65rem 1rem; border: none; background: #2563eb; color: white; border-radius: 5px; cursor: pointer; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; border-bottom: 1px solid #1f2937; padding: 0.5rem; }
    .flash { margin-bottom: 1rem; padding: 0.7rem; background: #064e3b; border: 1px solid #065f46; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>Adaptive Trading AI Master UI</h1>
  {% if message %}
    <div class="flash">{{ message }}</div>
  {% endif %}
  <div class="grid">
    <section class="card">
      <h2>Run AI Bot</h2>
      <form method="post" action="{{ url_for('run_bot') }}">
        <label for="name">Bot name</label>
        <input id="name" name="name" placeholder="MomentumBot-v1" required />

        <label for="periods">Synthetic market periods</label>
        <input id="periods" name="periods" type="number" min="320" value="420" required />

        <label for="train_size">Train window</label>
        <input id="train_size" name="train_size" type="number" min="120" value="260" required />

        <label for="test_size">Test window</label>
        <input id="test_size" name="test_size" type="number" min="20" value="40" required />

        <button type="submit">Run Bot</button>
      </form>
    </section>

    <section class="card">
      <h2>Saved AI Bots Performance</h2>
      {% if records %}
      <table>
        <thead>
          <tr>
            <th>Name</th><th>Created</th><th>Sharpe</th><th>Cumulative Return</th><th>Win Rate</th><th>Max DD</th><th>Turnover</th><th>Meta Updates</th>
          </tr>
        </thead>
        <tbody>
          {% for item in records %}
          <tr>
            <td>{{ item.name }}</td>
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


def create_app(
    registry_path: str | Path = "saved_bots.json",
    framework_factory: Callable[[], AdaptiveTradingFramework] = build_default_framework,
) -> Flask:
    app = Flask(__name__)
    registry = BotRegistry(registry_path)

    @app.get("/")
    def dashboard() -> str:
        records = sorted(registry.load(), key=lambda x: x.created_at, reverse=True)
        message = request.args.get("message", "")
        return render_template_string(HTML_TEMPLATE, records=records, message=message)

    @app.post("/run")
    def run_bot():
        name = request.form.get("name", "Unnamed Bot").strip()
        periods = int(request.form.get("periods", "420"))
        train_size = int(request.form.get("train_size", "260"))
        test_size = int(request.form.get("test_size", "40"))

        framework = framework_factory()
        framework.backtester.train_size = train_size
        framework.backtester.test_size = test_size

        frame = _synthetic_ohlcv(periods=periods)
        _, metrics, _ = framework.run(frame)

        registry.add(
            BotRunRecord(
                name=name,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                sharpe_ratio=float(metrics.get("sharpe_ratio", 0.0)),
                cumulative_return=float(metrics.get("cumulative_return", 0.0)),
                win_rate=float(metrics.get("win_rate", 0.0)),
                max_drawdown=float(metrics.get("max_drawdown", 0.0)),
                turnover=float(metrics.get("turnover", 0.0)),
                meta_updates=int(metrics.get("meta_updates", 0)),
            )
        )

        return redirect(url_for("dashboard", message=f"Ran {name} successfully."))

    return app


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
