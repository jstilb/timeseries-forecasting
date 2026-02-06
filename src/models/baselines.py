"""Naive baseline forecasters.

These simple methods provide the floor for model comparison. Any model that
cannot beat these baselines is adding complexity without value -- and that
happens more often than the deep learning literature would have you believe.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class NaiveForecaster:
    """Last-value (persistence) forecaster.

    Predicts that the future will equal the last observed value.
    This is the simplest possible baseline and is surprisingly hard
    to beat on many real-world datasets, especially at short horizons.
    """

    def __init__(self):
        self.last_value: float | None = None
        self.name = "Naive (Last Value)"

    def fit(self, y: np.ndarray) -> "NaiveForecaster":
        """Store the last observed value.

        Args:
            y: Historical target values, shape (n_samples,).

        Returns:
            self for method chaining.
        """
        self.last_value = float(y[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Predict by repeating the last observed value.

        Args:
            horizon: Number of future timesteps to predict.

        Returns:
            Array of shape (horizon,) with constant predictions.
        """
        if self.last_value is None:
            raise RuntimeError("Must call fit() before predict().")
        return np.full(horizon, self.last_value)


class SeasonalNaiveForecaster:
    """Seasonal naive forecaster.

    Predicts by repeating the values from exactly one seasonal cycle ago.
    For hourly data with daily seasonality, this means predicting tomorrow's
    3 PM value as today's 3 PM value.

    This baseline captures recurring patterns without any model fitting
    and is a strong benchmark for seasonal data.

    Args:
        seasonal_period: Length of one seasonal cycle (e.g., 24 for daily
            seasonality in hourly data, 168 for weekly).
    """

    def __init__(self, seasonal_period: int = 24):
        self.seasonal_period = seasonal_period
        self.seasonal_values: np.ndarray | None = None
        self.name = f"Seasonal Naive (period={seasonal_period})"

    def fit(self, y: np.ndarray) -> "SeasonalNaiveForecaster":
        """Store the last seasonal cycle of observations.

        Args:
            y: Historical target values, shape (n_samples,).

        Returns:
            self for method chaining.
        """
        if len(y) < self.seasonal_period:
            raise ValueError(
                f"Need at least {self.seasonal_period} observations, got {len(y)}"
            )
        self.seasonal_values = y[-self.seasonal_period:]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Predict by tiling the last seasonal cycle.

        Args:
            horizon: Number of future timesteps to predict.

        Returns:
            Array of shape (horizon,) with seasonal pattern repeated.
        """
        if self.seasonal_values is None:
            raise RuntimeError("Must call fit() before predict().")

        # Tile the seasonal values to cover the full horizon
        repeats = (horizon // self.seasonal_period) + 1
        tiled = np.tile(self.seasonal_values, repeats)
        return tiled[:horizon]


class DriftForecaster:
    """Linear drift forecaster.

    Extends the line between the first and last observation into the future.
    Captures simple linear trends that naive methods miss.
    """

    def __init__(self):
        self.last_value: float | None = None
        self.slope: float | None = None
        self.name = "Drift"

    def fit(self, y: np.ndarray) -> "DriftForecaster":
        """Compute the average drift (slope) from the training data.

        Args:
            y: Historical target values, shape (n_samples,).

        Returns:
            self for method chaining.
        """
        n = len(y)
        if n < 2:
            raise ValueError("Need at least 2 observations for drift.")
        self.last_value = float(y[-1])
        self.slope = (y[-1] - y[0]) / (n - 1)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Predict along the extrapolated drift line.

        Args:
            horizon: Number of future timesteps to predict.

        Returns:
            Array of shape (horizon,) with linearly extrapolated values.
        """
        if self.last_value is None or self.slope is None:
            raise RuntimeError("Must call fit() before predict().")
        steps = np.arange(1, horizon + 1)
        return self.last_value + self.slope * steps
