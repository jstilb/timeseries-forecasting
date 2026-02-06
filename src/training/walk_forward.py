"""Walk-forward validation evaluator.

Implements the gold-standard evaluation methodology for time series:
train on history, predict the future, advance the window, repeat.

This is fundamentally different from k-fold cross-validation because:
1. Training data always precedes test data (temporal causality)
2. No future information leaks into the training set
3. Results reflect real-world deployment performance

The evaluator supports both statistical models (fit/predict interface)
and deep learning models (PyTorch DataLoader interface).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.loader import TimeSeriesDataset
from src.data.normalize import FitOnTrainNormalizer
from src.data.splits import WalkForwardConfig, WalkForwardSplit
from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""

    fold_idx: int
    train_size: int
    test_size: int
    predictions: np.ndarray
    actuals: np.ndarray
    metrics: dict[str, float]
    elapsed_seconds: float


class WalkForwardEvaluator:
    """Walk-forward cross-validation for time series models.

    Evaluates a model using expanding or sliding window walk-forward
    validation. For each fold:
    1. Fit normalizer on training data only
    2. Fit/train the model on training data
    3. Generate predictions on the test window
    4. Compute metrics on original-scale values
    5. Advance the window and repeat

    Args:
        walk_forward_config: Walk-forward split configuration.
    """

    def __init__(self, walk_forward_config: WalkForwardConfig | None = None):
        self.config = walk_forward_config or WalkForwardConfig()
        self.splitter = WalkForwardSplit(self.config)

    def evaluate_statistical(
        self,
        y: np.ndarray,
        model_factory: Callable,
        horizon: int,
    ) -> list[FoldResult]:
        """Walk-forward evaluation for statistical models (ARIMA, Prophet, etc.).

        Args:
            y: Full target series, shape (n_samples,).
            model_factory: Callable that returns a fresh model instance.
            horizon: Forecast horizon.

        Returns:
            List of FoldResult for each fold.
        """
        results = []

        for fold_idx, (train_idx, test_idx) in enumerate(self.splitter.split(len(y))):
            start_time = time.time()

            y_train = y[train_idx]
            y_test = y[test_idx[:horizon]]  # Limit to forecast horizon

            # Fit and predict
            model = model_factory()
            model.fit(y_train)
            predictions = model.predict(len(y_test))

            # Compute metrics
            metrics = compute_all_metrics(y_test, predictions)

            elapsed = time.time() - start_time
            result = FoldResult(
                fold_idx=fold_idx,
                train_size=len(y_train),
                test_size=len(y_test),
                predictions=predictions,
                actuals=y_test,
                metrics=metrics,
                elapsed_seconds=elapsed,
            )
            results.append(result)

            logger.info(
                "Fold %d: train=%d, test=%d, MAE=%.4f, RMSE=%.4f (%.1fs)",
                fold_idx,
                len(y_train),
                len(y_test),
                metrics["mae"],
                metrics["rmse"],
                elapsed,
            )

        return results

    def evaluate_deep_learning(
        self,
        data: np.ndarray,
        target_idx: int,
        model_factory: Callable,
        trainer_factory: Callable,
        input_length: int = 96,
        forecast_horizon: int = 24,
        batch_size: int = 32,
    ) -> list[FoldResult]:
        """Walk-forward evaluation for deep learning models.

        For each fold:
        1. Split data temporally
        2. Fit normalizer on training portion only
        3. Create DataLoaders
        4. Train model from scratch
        5. Generate and evaluate predictions

        Args:
            data: Full dataset, shape (n_samples, n_features).
            target_idx: Column index of the target variable.
            model_factory: Callable returning a fresh model.
            trainer_factory: Callable(model) returning a Trainer instance.
            input_length: Input sequence length.
            forecast_horizon: Number of steps to predict.
            batch_size: DataLoader batch size.

        Returns:
            List of FoldResult for each fold.
        """
        results = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            self.splitter.split(len(data))
        ):
            start_time = time.time()

            train_data = data[train_idx]
            test_data = data[: test_idx[-1] + 1]  # Include history for test windows

            # Fit normalizer on training data only
            normalizer = FitOnTrainNormalizer(method="standard")
            normalizer.fit(train_data)

            train_normalized = normalizer.transform(train_data)
            test_normalized = normalizer.transform(test_data)

            # Create datasets
            train_dataset = TimeSeriesDataset(
                train_normalized,
                target_idx=target_idx,
                input_length=input_length,
                forecast_horizon=forecast_horizon,
            )

            # For test, we only use windows that start in the test period
            test_start = len(train_idx) - input_length  # Allow lookback into training
            test_dataset = TimeSeriesDataset(
                test_normalized[test_start:],
                target_idx=target_idx,
                input_length=input_length,
                forecast_horizon=forecast_horizon,
            )

            if len(train_dataset) == 0 or len(test_dataset) == 0:
                logger.warning("Fold %d: insufficient data, skipping", fold_idx)
                continue

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False
            )
            val_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )

            # Train model
            model = model_factory()
            trainer = trainer_factory(model)
            trainer.train(train_loader, val_loader)

            # Generate predictions
            pred_normalized = trainer.predict(val_loader)

            # Inverse transform predictions to original scale
            predictions = normalizer.inverse_transform_target(
                pred_normalized, target_idx
            )

            # Get actual values
            actuals_list = []
            for _, batch_y in val_loader:
                actuals_list.append(batch_y.numpy())
            actuals_normalized = np.concatenate(actuals_list, axis=0)
            actuals = normalizer.inverse_transform_target(
                actuals_normalized, target_idx
            )

            # Compute per-sample metrics (average across samples)
            all_metrics = []
            for i in range(min(len(predictions), len(actuals))):
                m = compute_all_metrics(actuals[i], predictions[i])
                all_metrics.append(m)

            # Average metrics across samples
            avg_metrics = {}
            for key in all_metrics[0]:
                values = [m[key] for m in all_metrics if not np.isnan(m[key])]
                avg_metrics[key] = np.mean(values) if values else float("nan")

            elapsed = time.time() - start_time
            result = FoldResult(
                fold_idx=fold_idx,
                train_size=len(train_idx),
                test_size=len(test_idx),
                predictions=predictions,
                actuals=actuals,
                metrics=avg_metrics,
                elapsed_seconds=elapsed,
            )
            results.append(result)

            logger.info(
                "Fold %d: train=%d, test_windows=%d, MAE=%.4f, RMSE=%.4f (%.1fs)",
                fold_idx,
                len(train_idx),
                len(predictions),
                avg_metrics.get("mae", float("nan")),
                avg_metrics.get("rmse", float("nan")),
                elapsed,
            )

        return results

    @staticmethod
    def aggregate_results(results: list[FoldResult]) -> dict[str, dict[str, float]]:
        """Aggregate metrics across all folds.

        Returns:
            Dictionary with 'mean' and 'std' for each metric.
        """
        if not results:
            return {}

        metric_names = list(results[0].metrics.keys())
        aggregated: dict[str, dict[str, float]] = {}

        for name in metric_names:
            values = [r.metrics[name] for r in results if not np.isnan(r.metrics[name])]
            aggregated[name] = {
                "mean": np.mean(values) if values else float("nan"),
                "std": np.std(values) if values else float("nan"),
                "min": np.min(values) if values else float("nan"),
                "max": np.max(values) if values else float("nan"),
            }

        return aggregated
