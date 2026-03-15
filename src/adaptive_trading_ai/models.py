from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SpectralAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.mha(x, x, x)
        return self.norm(out + x)


class SpectralTransformer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64, num_layers: int = 2, heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.blocks = nn.ModuleList([SpectralAttention(hidden_dim, heads=heads) for _ in range(num_layers)])
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.mlp(h[:, -1, :]).squeeze(-1)


@dataclass
class AdaptiveModel:
    sequence_length: int = 32
    lr: float = 1e-3
    hidden_dim: int = 64
    layers: int = 2
    heads: int = 4
    l2: float = 1e-4
    device: str = "cpu"
    scaler: StandardScaler = field(default_factory=StandardScaler)

    def __post_init__(self) -> None:
        self.rf = RandomForestRegressor(n_estimators=200, random_state=42)
        self.gbr = GradientBoostingRegressor(random_state=42)
        self.net: SpectralTransformer | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, epochs: int = 20) -> Dict[str, float]:
        X_vals = self.scaler.fit_transform(X.values)
        self.rf.fit(X_vals, y.values)
        self.gbr.fit(X_vals, y.values)

        seq_x, seq_y = self._to_sequences(X_vals, y.values)
        if len(seq_x) == 0:
            return {"rf_rmse": np.nan, "gbr_rmse": np.nan, "dl_rmse": np.nan}

        feature_dim = seq_x.shape[-1]
        self.net = SpectralTransformer(feature_dim, hidden_dim=self.hidden_dim, num_layers=self.layers, heads=self.heads).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)
        loss_fn = nn.MSELoss()

        x_t = torch.tensor(seq_x, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(seq_y, dtype=torch.float32, device=self.device)
        for _ in range(epochs):
            self.net.train()
            self.optimizer.zero_grad()
            pred = self.net(x_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            self.optimizer.step()
            scheduler.step(loss.detach())

        metrics = self.evaluate(X, y)
        return metrics

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        X_vals = self.scaler.transform(X.values)
        rf_pred = self.rf.predict(X_vals)
        gbr_pred = self.gbr.predict(X_vals)
        dl_pred = self.predict(X)
        return {
            "rf_rmse": float(np.sqrt(mean_squared_error(y.values, rf_pred))),
            "gbr_rmse": float(np.sqrt(mean_squared_error(y.values, gbr_pred))),
            "dl_rmse": float(np.sqrt(mean_squared_error(y.values[-len(dl_pred) :], dl_pred))),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_vals = self.scaler.transform(X.values)
        ensemble = 0.5 * self.rf.predict(X_vals) + 0.5 * self.gbr.predict(X_vals)
        if self.net is None:
            return ensemble

        seq_x, _ = self._to_sequences(X_vals, np.zeros(len(X_vals)))
        if len(seq_x) == 0:
            return ensemble

        self.net.eval()
        with torch.no_grad():
            x_t = torch.tensor(seq_x, dtype=torch.float32, device=self.device)
            dl = self.net(x_t).cpu().numpy()
        ensemble_tail = ensemble[-len(dl) :]
        return 0.4 * ensemble_tail + 0.6 * dl

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        imp = self.rf.feature_importances_
        order = np.argsort(imp)[::-1]
        return pd.DataFrame({"feature": np.array(feature_names)[order], "importance": imp[order]})

    def online_update(self, X_recent: pd.DataFrame, y_recent: pd.Series, steps: int = 5) -> None:
        if self.net is None or self.optimizer is None:
            return
        X_vals = self.scaler.transform(X_recent.values)
        seq_x, seq_y = self._to_sequences(X_vals, y_recent.values)
        if len(seq_x) == 0:
            return
        x_t = torch.tensor(seq_x, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(seq_y, dtype=torch.float32, device=self.device)
        loss_fn = nn.MSELoss()
        self.net.train()
        for _ in range(steps):
            self.optimizer.zero_grad()
            pred = self.net(x_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            self.optimizer.step()

    def _to_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seq_x, seq_y = [], []
        for i in range(self.sequence_length, len(X)):
            seq_x.append(X[i - self.sequence_length : i])
            seq_y.append(y[i])
        if not seq_x:
            return np.array([]), np.array([])
        return np.stack(seq_x), np.array(seq_y)
