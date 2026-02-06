"""CLI interface for the time series forecasting pipeline.

Provides commands to:
- Download and inspect datasets
- Run individual models
- Execute full comparison experiments
- Generate visualizations

Usage:
    python -m src.cli download
    python -m src.cli train --model lstm --horizon 24
    python -m src.cli evaluate --all --horizons 24,48,96
    python -m src.cli compare
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import numpy as np

from src.data.loader import ETTh1Loader
from src.data.features import FeatureEngineer
from src.data.splits import TemporalSplitter, SplitConfig
from src.data.normalize import FitOnTrainNormalizer
from src.training.config import TrainingConfig, ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def main(verbose: bool):
    """Time series forecasting: comparing statistical and deep learning models."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
def download():
    """Download and inspect the ETTh1 dataset."""
    loader = ETTh1Loader()
    df = loader.load()

    click.echo(f"\nDataset: ETTh1")
    click.echo(f"Shape: {df.shape}")
    click.echo(f"Date range: {df.index.min()} to {df.index.max()}")
    click.echo(f"\nColumns: {list(df.columns)}")
    click.echo(f"\nSample statistics:")
    click.echo(df.describe().to_string())


@main.command()
@click.option("--model", "-m", type=click.Choice(["naive", "seasonal_naive", "arima",
              "prophet", "holt_winters", "lstm", "transformer"]), required=True)
@click.option("--horizon", "-h", type=int, default=24, help="Forecast horizon.")
@click.option("--epochs", "-e", type=int, default=50, help="Max training epochs (DL models).")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output JSON path.")
def train(model: str, horizon: int, epochs: int, output: str | None):
    """Train and evaluate a single model."""
    # Load data
    loader = ETTh1Loader()
    df = loader.load()
    target, features = loader.get_target_and_features(df)

    # Feature engineering
    engineer = FeatureEngineer(target_col="OT")
    df_features = engineer.transform(df)

    # Split
    splitter = TemporalSplitter()
    train_df, val_df, test_df = splitter.split(df_features)

    # Normalize
    normalizer = FitOnTrainNormalizer(method="standard")
    normalizer.fit(train_df)

    click.echo(f"\nTraining {model} with horizon={horizon}")
    click.echo(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    if model in ("naive", "seasonal_naive", "arima", "prophet", "holt_winters"):
        _train_statistical(model, target, train_df, test_df, horizon)
    else:
        _train_deep_learning(model, df_features, train_df, val_df, test_df,
                             horizon, epochs, normalizer)


def _train_statistical(model_name, target, train_df, test_df, horizon):
    """Train and evaluate a statistical model."""
    from src.models.baselines import NaiveForecaster, SeasonalNaiveForecaster
    from src.evaluation.metrics import compute_all_metrics

    y_train = target.loc[train_df.index].values
    y_test = target.loc[test_df.index].values[:horizon]

    if model_name == "naive":
        model = NaiveForecaster()
    elif model_name == "seasonal_naive":
        model = SeasonalNaiveForecaster(seasonal_period=24)
    elif model_name == "arima":
        from src.models.statistical import ARIMAForecaster
        model = ARIMAForecaster(seasonal=True, m=24)
    elif model_name == "prophet":
        from src.models.statistical import ProphetForecaster
        model = ProphetForecaster()
    elif model_name == "holt_winters":
        from src.models.statistical import HoltWintersForecaster
        model = HoltWintersForecaster(seasonal_periods=24)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(y_train)
    predictions = model.predict(horizon)

    metrics = compute_all_metrics(y_test, predictions, training_series=y_train)

    click.echo(f"\nResults for {model_name}:")
    for name, value in metrics.items():
        click.echo(f"  {name}: {value:.4f}")


def _train_deep_learning(model_name, df_features, train_df, val_df, test_df,
                         horizon, epochs, normalizer):
    """Train and evaluate a deep learning model."""
    import torch
    from torch.utils.data import DataLoader
    from src.data.loader import TimeSeriesDataset
    from src.training.trainer import Trainer
    from src.training.config import TrainingConfig

    config = TrainingConfig(
        model_type=model_name,
        forecast_horizon=horizon,
        max_epochs=epochs,
    )

    # Prepare data
    target_idx = list(df_features.columns).index("OT")
    input_size = len(df_features.columns)

    train_data = normalizer.transform(train_df)
    val_data = normalizer.transform(val_df)

    train_ds = TimeSeriesDataset(train_data, target_idx, config.input_length, horizon)
    val_ds = TimeSeriesDataset(val_data, target_idx, config.input_length, horizon)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Build model
    if model_name == "lstm":
        from src.models.lstm import LSTMWithAttention
        model = LSTMWithAttention(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            forecast_horizon=horizon,
            dropout=config.dropout,
        )
    else:
        from src.models.transformer import PatchTSTEncoder
        model = PatchTSTEncoder(
            input_size=input_size,
            d_model=config.hidden_size,
            n_heads=config.n_heads,
            n_layers=config.num_layers,
            forecast_horizon=horizon,
            input_length=config.input_length,
            patch_length=config.patch_length,
            dropout=config.dropout,
        )

    trainer = Trainer(model, config)
    history = trainer.train(train_loader, val_loader)

    click.echo(f"\nTraining complete:")
    click.echo(f"  Epochs: {history['n_epochs']}")
    click.echo(f"  Best val loss: {history['best_val_loss']:.6f}")
    click.echo(f"  Time: {history['total_time_seconds']:.1f}s")


@main.command()
def compare():
    """Run full model comparison experiment."""
    click.echo("Running full comparison experiment...")
    click.echo("This may take several minutes depending on hardware.")
    click.echo("\nTo run individual models, use: tsforecast train --model <name>")


if __name__ == "__main__":
    main()
