"""Forecasting evaluation metrics.

A comprehensive suite of metrics for time series forecast evaluation.
Each metric captures a different aspect of forecast quality:

- MAE/RMSE: absolute error magnitude (RMSE penalizes large errors more)
- MAPE/SMAPE: percentage errors (scale-independent comparison)
- MASE: relative to naive baseline (the scientifically correct scale-free metric)
- Directional accuracy: did we predict the direction of change correctly?

Note on MAPE: it is undefined when actuals contain zeros and becomes
extremely large for near-zero values. SMAPE and MASE are preferred
alternatives. We include MAPE because interviewers ask about it.
"""

from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error.

    The average absolute difference between predicted and actual values.
    Interpretable in the same units as the target variable.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.

    Returns:
        MAE value (lower is better, minimum 0).
    """
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error.

    Penalizes large errors more heavily than MAE due to squaring.
    Useful when large forecast errors are disproportionately costly.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.

    Returns:
        RMSE value (lower is better, minimum 0).
    """
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error.

    Scale-independent metric expressed as a percentage.

    WARNING: Undefined for zero actuals. We add epsilon to prevent division
    by zero, but results are unreliable when actuals are near zero.
    Prefer SMAPE or MASE for robust evaluation.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.
        epsilon: Small value to prevent division by zero.

    Returns:
        MAPE as a percentage (lower is better, minimum 0).
    """
    return float(np.mean(np.abs((actual - predicted) / (np.abs(actual) + epsilon))) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error.

    A symmetric version of MAPE that handles the case where actuals are zero
    more gracefully. Bounded between 0% and 200%.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.
        epsilon: Small value to prevent division by zero.

    Returns:
        SMAPE as a percentage (lower is better, range [0, 200]).
    """
    numerator = np.abs(actual - predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2 + epsilon
    return float(np.mean(numerator / denominator) * 100)


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    training_series: np.ndarray,
    seasonal_period: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    The scientifically preferred scale-free metric (Hyndman & Koehler, 2006).
    Compares forecast errors against the errors of a naive seasonal forecast.

    MASE < 1: model beats the naive baseline
    MASE = 1: model performs same as naive baseline
    MASE > 1: naive baseline is better (model is useless)

    Args:
        actual: Ground truth values for the test period.
        predicted: Predicted values.
        training_series: Training data used to compute the naive baseline error.
        seasonal_period: Seasonal period for the naive baseline (1 = non-seasonal).

    Returns:
        MASE value (lower is better, < 1 means beating naive).
    """
    # Compute naive baseline error on training data
    naive_errors = np.abs(
        training_series[seasonal_period:] - training_series[:-seasonal_period]
    )
    naive_mae = np.mean(naive_errors)

    if naive_mae < 1e-8:
        return float("inf")

    forecast_mae = np.mean(np.abs(actual - predicted))
    return float(forecast_mae / naive_mae)


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Directional Accuracy (DA).

    Measures the percentage of timesteps where the predicted direction
    of change matches the actual direction of change. Useful in trading
    and operations where the direction matters more than magnitude.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.

    Returns:
        DA as a percentage (higher is better, range [0, 100]).
        Returns NaN if fewer than 2 observations.
    """
    if len(actual) < 2:
        return float("nan")

    actual_direction = np.sign(np.diff(actual))
    predicted_direction = np.sign(np.diff(predicted))

    correct = np.sum(actual_direction == predicted_direction)
    return float(correct / len(actual_direction) * 100)


def prediction_interval_coverage(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Prediction Interval Coverage Probability (PICP).

    Measures the percentage of actual values that fall within the
    predicted confidence interval. For a 95% CI, ideal coverage is 95%.

    Args:
        actual: Ground truth values.
        lower: Lower bound of prediction interval.
        upper: Upper bound of prediction interval.

    Returns:
        Coverage as a percentage (closer to nominal level is better).
    """
    in_interval = (actual >= lower) & (actual <= upper)
    return float(np.mean(in_interval) * 100)


def compute_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    training_series: np.ndarray | None = None,
    seasonal_period: int = 24,
) -> dict[str, float]:
    """Compute all forecasting metrics.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.
        training_series: Training data for MASE computation (optional).
        seasonal_period: Seasonal period for MASE.

    Returns:
        Dictionary mapping metric names to values.
    """
    results = {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
        "smape": smape(actual, predicted),
        "directional_accuracy": directional_accuracy(actual, predicted),
    }

    if training_series is not None and len(training_series) > seasonal_period:
        results["mase"] = mase(actual, predicted, training_series, seasonal_period)
    else:
        results["mase"] = float("nan")

    return results
