"""Hyperparameter and experiment configuration.

Centralizes all configuration to make experiments reproducible.
Every training run should be fully determined by its config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TrainingConfig:
    """Training hyperparameters for deep learning models."""

    # Architecture
    model_type: Literal["lstm", "transformer"] = "lstm"
    input_length: int = 96
    forecast_horizon: int = 24
    hidden_size: int = 128
    num_layers: int = 2
    n_heads: int = 8  # Transformer only
    d_ff: int = 256  # Transformer only
    patch_length: int = 16  # Transformer only
    dropout: float = 0.1

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 10  # Early stopping patience
    min_delta: float = 1e-4  # Minimum improvement for early stopping

    # Scheduler
    scheduler: Literal["cosine", "plateau", "none"] = "cosine"
    warmup_epochs: int = 5

    # Training
    gradient_clip: float = 1.0
    seed: int = 42

    # Device
    device: str = "auto"  # 'auto', 'cuda', 'mps', 'cpu'


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    name: str = "etth1_comparison"
    dataset: str = "ETTh1"
    target_col: str = "OT"

    # Data
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    normalization: Literal["standard", "minmax"] = "standard"

    # Features
    lag_periods: list[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24])
    rolling_windows: list[int] = field(default_factory=lambda: [6, 12, 24, 48])
    add_calendar: bool = True
    add_fourier: bool = True

    # Evaluation
    forecast_horizons: list[int] = field(default_factory=lambda: [24, 48, 96, 168])
    walk_forward_splits: int = 5

    # Models to train
    models: list[str] = field(
        default_factory=lambda: [
            "naive",
            "seasonal_naive",
            "arima",
            "prophet",
            "holt_winters",
            "lstm",
            "transformer",
        ]
    )

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # MLflow
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "timeseries-forecasting"
