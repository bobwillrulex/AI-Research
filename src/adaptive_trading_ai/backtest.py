from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .meta import MetaOptimizer
from .models import AdaptiveModel


@dataclass
class WalkForwardBacktester:
    train_size: int = 1000
    test_size: int = 200
    threshold: float = 0.0

    def run(
        self,
        features: pd.DataFrame,
        target_col: str,
        model_config: Dict[str, float],
        meta_optimizer: MetaOptimizer,
    ) -> tuple[pd.DataFrame, Dict[str, float]]:
        history: List[Dict[str, float]] = []
        predictions = pd.Series(index=features.index, dtype=float)
        positions = pd.Series(index=features.index, dtype=float)

        start = 0
        config = dict(model_config)
        while start + self.train_size + self.test_size <= len(features):
            train = features.iloc[start : start + self.train_size]
            test = features.iloc[start + self.train_size : start + self.train_size + self.test_size]

            X_train = train.drop(columns=[target_col, "regime_name"], errors="ignore")
            y_train = train[target_col]
            X_test = test.drop(columns=[target_col, "regime_name"], errors="ignore")
            y_test = test[target_col]

            model = AdaptiveModel(
                lr=config["lr"],
                hidden_dim=int(config["hidden_dim"]),
                layers=int(config["layers"]),
                heads=int(config["heads"]),
                l2=config["l2"],
            )
            model.fit(X_train, y_train, epochs=15)
            pred = model.predict(X_test)
            pred_idx = X_test.index[-len(pred) :]
            predictions.loc[pred_idx] = pred
            model.online_update(X_test.tail(128), y_test.tail(128), steps=3)

            pos = np.where(pred > self.threshold, 1.0, np.where(pred < -self.threshold, -1.0, 0.0))
            positions.loc[pred_idx] = pos
            realized = y_test.values[-len(pred) :]
            pnl = pos * realized

            metrics = {
                "window_start": float(start),
                "rolling_sharpe": float(np.mean(pnl) / (np.std(pnl) + 1e-8) * np.sqrt(252)),
                "prediction_rmse": float(np.sqrt(np.mean((pred - realized) ** 2))),
                "rmse_target": 0.01,
            }
            history.append(metrics)
            config = meta_optimizer.step(config, metrics)
            start += self.test_size

        bt = pd.DataFrame({"prediction": predictions, "position": positions}, index=features.index).dropna()
        bt["actual_return"] = features.loc[bt.index, target_col]
        bt["strategy_return"] = bt["position"] * bt["actual_return"]
        bt["equity_curve"] = (1 + bt["strategy_return"]).cumprod()

        summary = self.compute_metrics(bt["strategy_return"], bt["position"])
        summary["meta_updates"] = len(history)
        return bt, summary

    @staticmethod
    def compute_metrics(returns: pd.Series, position: pd.Series) -> Dict[str, float]:
        cum = float((1 + returns).prod() - 1)
        sharpe = float(returns.mean() / (returns.std() + 1e-8) * np.sqrt(252))
        downside = returns[returns < 0].std()
        sortino = float(returns.mean() / (downside + 1e-8) * np.sqrt(252))
        equity = (1 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        max_dd = float(drawdown.min())
        win_rate = float((returns > 0).mean())
        turnover = float(position.diff().abs().fillna(0).mean())
        return {
            "cumulative_return": cum,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "turnover": turnover,
        }
