from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np
import pandas as pd
from flask import Flask, abort, redirect, render_template, request, url_for

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


TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")
MAX_SYMBOLS = 25
MIN_ROWS = 120


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


def _normalize_ohlcv(df: pd.DataFrame, symbol: str, missing_policy: str = "drop") -> pd.DataFrame:
    aliases = {
        "open": {"open", "o"},
        "high": {"high", "h"},
        "low": {"low", "l"},
        "close": {"close", "adj close", "adj_close", "adjusted_close", "c"},
        "volume": {"volume", "vol", "v"},
    }
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    rename: dict[str, str] = {}
    for canon, opts in aliases.items():
        for opt in opts:
            if opt in lower_cols:
                rename[lower_cols[opt]] = canon
                break

    frame = df.rename(columns=rename)
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV column(s) for {symbol}: {', '.join(missing)}")

    frame = frame[required].copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.ffill().bfill() if missing_policy == "ffill" else frame.dropna()

    if len(frame) < MIN_ROWS:
        raise ValueError(f"Insufficient rows for {symbol}. Need at least {MIN_ROWS}, got {len(frame)}")
    return frame


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
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True)
    return df.set_index(dt_col)


def _load_yfinance_ohlcv(symbol: str, periods: int) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("yfinance is not installed") from exc
    period_window = max(periods, 365)
    raw = yf.download(symbol, period=f"{period_window}d", interval="1d", progress=False)
    if raw.empty:
        raise ValueError(f"No yfinance data for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index, utc=True)
    return raw.tail(periods)


def _resolve_ohlcv(symbol: str, data_source: str, periods: int, data_path: str | None, missing_policy: str) -> pd.DataFrame:
    if data_source == "synthetic":
        seed = abs(hash(symbol)) % 10_000
        raw = _synthetic_ohlcv(periods=periods, seed=seed)
    elif data_source == "csv":
        raw = _load_csv_ohlcv(symbol=symbol, data_path=data_path)
    elif data_source == "yfinance":
        raw = _load_yfinance_ohlcv(symbol=symbol, periods=periods)
    else:
        raise ValueError(f"Unsupported data source: {data_source}")
    return _normalize_ohlcv(raw, symbol=symbol, missing_policy=missing_policy)


def _persist_artifact(artifacts_dir: Path, run_id: str, payload: dict) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts_dir / f"{run_id}.json"
    artifact_path.write_text(json.dumps(payload, indent=2, default=str))
    return artifact_path


def _parse_form(request_form) -> tuple[dict, list[str]]:
    errors: list[str] = []

    def parse_int(field: str, default: int) -> int:
        raw = request_form.get(field, str(default)).strip()
        try:
            return int(raw)
        except ValueError:
            errors.append(f"{field.replace('_', ' ').title()} must be an integer.")
            return default

    symbols = [s.strip().upper() for s in request_form.get("symbols", "").split(",") if s.strip()]
    symbols = list(dict.fromkeys(symbols))

    payload = {
        "family_name": request_form.get("name", "Unnamed Bot").strip() or "Unnamed Bot",
        "data_source": request_form.get("data_source", "synthetic").strip().lower(),
        "symbols": symbols,
        "data_path": request_form.get("data_path", "").strip() or None,
        "periods": parse_int("periods", 420),
        "train_size": parse_int("train_size", 260),
        "test_size": parse_int("test_size", 40),
        "missing_policy": request_form.get("missing_policy", "drop").strip().lower() or "drop",
    }

    if not symbols:
        errors.append("Provide at least one symbol.")
    if len(symbols) > MAX_SYMBOLS:
        errors.append(f"Maximum symbol count is {MAX_SYMBOLS}; got {len(symbols)}.")

    bad_symbols = [symbol for symbol in symbols if not TICKER_PATTERN.match(symbol)]
    if bad_symbols:
        errors.append(f"Invalid ticker format: {', '.join(bad_symbols)}")

    if payload["periods"] < MIN_ROWS:
        errors.append(f"Periods must be >= {MIN_ROWS}.")
    if payload["train_size"] <= 0 or payload["test_size"] <= 0:
        errors.append("Train/test size must be positive.")
    if payload["test_size"] >= payload["periods"]:
        errors.append("Test size must be less than periods.")
    if payload["train_size"] + payload["test_size"] > payload["periods"]:
        errors.append("Train size + test size must be <= periods.")

    if payload["data_source"] == "csv":
        if not payload["data_path"]:
            errors.append("CSV data source requires a file or directory path.")
        else:
            base = Path(payload["data_path"])
            if not base.exists():
                errors.append(f"CSV path does not exist: {base}")
            elif base.is_file() and base.suffix.lower() != ".csv":
                errors.append("CSV path file must end with .csv")

    if payload["missing_policy"] not in {"drop", "ffill"}:
        errors.append("Missing-value policy must be one of: drop, ffill")

    return payload, errors


def create_app(
    registry_path: str | Path = "saved_bots.json",
    artifacts_dir: str | Path = "saved_bot_artifacts",
    framework_factory: Callable[[], AdaptiveTradingFramework] = build_default_framework,
) -> Flask:
    app = Flask(__name__)
    registry = BotRegistry(registry_path)
    artifacts_dir = Path(artifacts_dir)

    def _dashboard_payload(form_data: dict | None = None, errors: list[str] | None = None) -> dict:
        records = registry.load()
        sort_key = request.args.get("sort", "sharpe_ratio")
        sort_dir = request.args.get("dir", "desc")
        descending = sort_dir != "asc"

        symbol_filter = request.args.get("symbol", "").strip().upper()
        source_filter = request.args.get("source", "").strip().lower()

        if symbol_filter:
            records = [r for r in records if r.symbol == symbol_filter]
        if source_filter:
            records = [r for r in records if r.data_source == source_filter]

        valid_sort = {"sharpe_ratio", "cumulative_return", "max_drawdown", "win_rate", "created_at"}
        if sort_key not in valid_sort:
            sort_key = "sharpe_ratio"
        records = sorted(records, key=lambda rec: getattr(rec, sort_key), reverse=descending)

        page = max(int(request.args.get("page", "1")), 1)
        per_page = 10
        total_pages = max((len(records) - 1) // per_page + 1, 1)
        page = min(page, total_pages)
        page_records = records[(page - 1) * per_page : page * per_page]

        compare_ids = request.args.getlist("compare")
        compare_items = [r for r in records if r.run_id in compare_ids][:4]

        return {
            "records": page_records,
            "message": request.args.get("message", ""),
            "errors": errors or [],
            "form_data": form_data or {},
            "page": page,
            "total_pages": total_pages,
            "sort_key": sort_key,
            "sort_dir": sort_dir,
            "symbol_filter": symbol_filter,
            "source_filter": source_filter,
            "compare_items": compare_items,
        }

    @app.get("/")
    def dashboard() -> str:
        return render_template("dashboard.html", **_dashboard_payload())

    @app.get("/bots/<run_id>")
    def bot_detail(run_id: str) -> str:
        record = registry.get(run_id)
        if record is None:
            abort(404)
        artifact_path = Path(record.artifact_path)
        artifact = json.loads(artifact_path.read_text()) if artifact_path.exists() else {"error": "artifact missing"}
        diagnostics = {
            "backtest_rows": len(artifact.get("backtest_preview", [])),
            "artifact_keys": len(artifact.get("framework_artifacts", {})),
            "train_size": artifact.get("config", {}).get("train_size"),
            "test_size": artifact.get("config", {}).get("test_size"),
        }
        return render_template("bot_detail.html", record=record, artifact=artifact, diagnostics=diagnostics)

    @app.post("/run")
    def run_bot():
        params, errors = _parse_form(request.form)
        if errors:
            return render_template("dashboard.html", **_dashboard_payload(form_data=params, errors=errors)), 400

        saved = 0
        failures: list[str] = []
        for symbol in params["symbols"]:
            run_id = uuid4().hex
            try:
                framework = framework_factory()
                framework.backtester.train_size = params["train_size"]
                framework.backtester.test_size = params["test_size"]

                frame = _resolve_ohlcv(
                    symbol=symbol,
                    data_source=params["data_source"],
                    periods=params["periods"],
                    data_path=params["data_path"],
                    missing_policy=params["missing_policy"],
                )
                bt_df, metrics, artifacts = framework.run(frame)
                payload = {
                    "run_id": run_id,
                    "name": f"{params['family_name']}-{symbol}",
                    "symbol": symbol,
                    "data_source": params["data_source"],
                    "status": "succeeded",
                    "metrics": metrics,
                    "framework_artifacts": artifacts,
                    "backtest_preview": bt_df.tail(50).to_dict(orient="records") if hasattr(bt_df, "tail") else [],
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "config": params,
                    "repro_hash": uuid4().hex,
                }
                artifact_path = _persist_artifact(artifacts_dir=artifacts_dir, run_id=run_id, payload=payload)
                registry.add(
                    BotRunRecord(
                        run_id=run_id,
                        name=f"{params['family_name']}-{symbol}",
                        symbol=symbol,
                        data_source=params["data_source"],
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
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{symbol}: {exc}")
                _persist_artifact(
                    artifacts_dir=artifacts_dir,
                    run_id=run_id,
                    payload={
                        "run_id": run_id,
                        "name": f"{params['family_name']}-{symbol}",
                        "symbol": symbol,
                        "data_source": params["data_source"],
                        "status": "failed",
                        "error": str(exc),
                        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "config": params,
                    },
                )

        message = f"Ran {saved}/{len(params['symbols'])} bot(s) successfully."
        if failures:
            message += " Failures: " + " | ".join(failures)
        return redirect(url_for("dashboard", message=message))

    return app


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
