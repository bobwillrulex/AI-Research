from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


class VisualizationEngine:
    def plot_cycles(self, returns: pd.Series, reconstructed: pd.Series) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 4))
        returns.plot(ax=ax, label="returns", alpha=0.7)
        reconstructed.plot(ax=ax, label="reconstructed_cycle", alpha=0.8)
        ax.legend()
        ax.set_title("Return Series vs Reconstructed Cycles")
        return fig

    def plot_dominant_frequency(self, spectral_df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 4))
        spectral_df["dominant_frequency"].plot(ax=ax)
        ax.set_title("Dominant Frequency Drift Over Time")
        return fig

    def plot_phase(self, hilbert_df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 4))
        hilbert_df["phase"].plot(ax=ax)
        ax.set_title("Cycle Phase")
        return fig

    def plot_predictions(self, bt_df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 4))
        bt_df["prediction"].plot(ax=ax, label="pred")
        bt_df["actual_return"].plot(ax=ax, label="actual", alpha=0.7)
        ax.legend()
        ax.set_title("Predicted vs Actual Returns")
        return fig

    def plot_equity(self, bt_df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 4))
        bt_df["equity_curve"].plot(ax=ax)
        ax.set_title("Strategy Equity Curve")
        return fig

    def plot_feature_importance(self, importance: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        importance.head(20).sort_values("importance").plot.barh(x="feature", y="importance", ax=ax)
        ax.set_title("Feature Importance")
        return fig

    def plot_meta_performance(self, history: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 4))
        history[["rolling_sharpe", "prediction_rmse"]].plot(ax=ax)
        ax.set_title("Meta-learning Rolling Performance")
        return fig
