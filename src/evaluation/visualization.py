"""Forecast visualization suite.

Generates publication-quality plots for:
- Forecast vs actual comparison
- Per-horizon error degradation curves
- Model comparison charts
- Attention weight heatmaps
- Prediction interval calibration
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

logger = logging.getLogger(__name__)

# Professional plot styling
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "actual": "#2C3E50",
    "lstm": "#E74C3C",
    "transformer": "#3498DB",
    "arima": "#2ECC71",
    "prophet": "#9B59B6",
    "holt_winters": "#F39C12",
    "naive": "#95A5A6",
    "seasonal_naive": "#7F8C8D",
}


def plot_forecast_comparison(
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    dates: np.ndarray | None = None,
    title: str = "Forecast Comparison",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 6),
) -> None:
    """Plot actual values against multiple model forecasts.

    Args:
        actual: Ground truth values.
        forecasts: Dictionary mapping model names to prediction arrays.
        dates: Optional datetime array for x-axis.
        title: Plot title.
        save_path: Path to save the figure (shows if None).
        figsize: Figure dimensions.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = dates if dates is not None else np.arange(len(actual))

    # Plot actual
    ax.plot(x, actual, color=COLORS["actual"], linewidth=2, label="Actual", zorder=5)

    # Plot each forecast
    for name, pred in forecasts.items():
        color = COLORS.get(name.lower().replace("-", "_").replace(" ", "_"), "#34495E")
        ax.plot(x[:len(pred)], pred, color=color, linewidth=1.5,
                label=name, alpha=0.8, linestyle="--")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend(loc="upper left", framealpha=0.9)

    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved forecast comparison plot to %s", save_path)
    plt.close(fig)


def plot_horizon_error(
    horizon_errors: dict[str, dict[int, float]],
    metric_name: str = "MAE",
    title: str = "Per-Horizon Error Degradation",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Plot how forecast error degrades with increasing horizon.

    This is one of the most important diagnostic plots: it shows which
    models maintain accuracy over longer horizons and which fall apart.

    Args:
        horizon_errors: Dict mapping model names to {horizon_step: error} dicts.
        metric_name: Name of the metric being plotted.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure dimensions.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, errors in horizon_errors.items():
        horizons = sorted(errors.keys())
        values = [errors[h] for h in horizons]
        color = COLORS.get(name.lower().replace("-", "_").replace(" ", "_"), "#34495E")
        ax.plot(horizons, values, color=color, linewidth=2, marker="o",
                markersize=4, label=name, alpha=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Forecast Horizon (steps ahead)", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved horizon error plot to %s", save_path)
    plt.close(fig)


def plot_model_comparison_bar(
    results: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    title: str = "Model Comparison",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """Create a grouped bar chart comparing models across metrics.

    Args:
        results: Dict mapping model names to {metric: value} dicts.
        metrics: List of metrics to plot (default: MAE, RMSE, SMAPE).
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure dimensions.
    """
    if metrics is None:
        metrics = ["mae", "rmse", "smape"]

    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = []
        colors = []
        for name in model_names:
            values.append(results[name].get(metric, 0))
            colors.append(
                COLORS.get(name.lower().replace("-", "_").replace(" ", "_"), "#34495E")
            )

        bars = ax.bar(range(n_models), values, color=colors, alpha=0.8, edgecolor="white")
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
        ax.set_title(metric.upper(), fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        # Highlight best model
        best_idx = np.argmin(values)
        bars[best_idx].set_edgecolor("#27AE60")
        bars[best_idx].set_linewidth(2)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved model comparison plot to %s", save_path)
    plt.close(fig)


def plot_attention_weights(
    weights: np.ndarray,
    title: str = "LSTM Temporal Attention Weights",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (12, 4),
) -> None:
    """Plot attention weight heatmap for interpretability.

    Shows which historical timesteps the LSTM attention mechanism
    considers most relevant for forecasting. Peaks typically appear at
    seasonal lags (e.g., 24 hours ago for daily patterns).

    Args:
        weights: Attention weights, shape (n_samples, input_length).
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure dimensions.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Average attention across samples
    avg_weights = np.mean(weights, axis=0) if weights.ndim > 1 else weights

    ax.bar(range(len(avg_weights)), avg_weights, color="#3498DB", alpha=0.7)
    ax.set_xlabel("Input Timestep (hours ago)", fontsize=12)
    ax.set_ylabel("Attention Weight", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved attention weights plot to %s", save_path)
    plt.close(fig)
