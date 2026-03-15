from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class MetaOptimizer:
    """Adjusts training hyperparameters based on rolling performance."""

    lr_bounds: tuple[float, float] = (1e-5, 5e-3)
    hidden_bounds: tuple[int, int] = (32, 256)
    l2_bounds: tuple[float, float] = (1e-6, 5e-3)

    def step(self, config: Dict[str, float], metrics: Dict[str, float]) -> Dict[str, float]:
        new_config = dict(config)
        sharpe = metrics.get("rolling_sharpe", 0.0)
        rmse = metrics.get("prediction_rmse", 1.0)

        if sharpe < 0:
            new_config["lr"] = min(config["lr"] * 1.2, self.lr_bounds[1])
            new_config["l2"] = min(config["l2"] * 1.2, self.l2_bounds[1])
        else:
            new_config["lr"] = max(config["lr"] * 0.95, self.lr_bounds[0])
            new_config["l2"] = max(config["l2"] * 0.95, self.l2_bounds[0])

        if rmse > metrics.get("rmse_target", 0.01):
            new_config["hidden_dim"] = int(np.clip(config["hidden_dim"] + 16, *self.hidden_bounds))
        else:
            new_config["hidden_dim"] = int(np.clip(config["hidden_dim"] - 8, *self.hidden_bounds))

        new_config["layers"] = max(1, int(config.get("layers", 2) + (1 if sharpe < -0.2 else 0)))
        new_config["heads"] = max(2, int(config.get("heads", 4)))
        return new_config
