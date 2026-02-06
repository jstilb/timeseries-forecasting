"""Tests for forecasting evaluation metrics.

Validates metric correctness with known analytical solutions
and edge cases. Every metric should have:
1. A basic correctness test with known values
2. A perfect prediction test (should return 0 or 100)
3. An edge case test (zeros, constant series, etc.)
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    mae,
    rmse,
    mape,
    smape,
    mase,
    directional_accuracy,
    prediction_interval_coverage,
    compute_all_metrics,
)


class TestMAE:
    def test_basic(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.5, 2.5, 2.5, 3.5])
        assert mae(actual, predicted) == pytest.approx(0.5)

    def test_perfect_prediction(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert mae(actual, actual) == pytest.approx(0.0)

    def test_symmetric(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])
        # MAE is symmetric: MAE(a, p) == MAE(p, a) since |a-p| == |p-a|
        assert mae(actual, predicted) == mae(predicted, actual)

    def test_single_value(self):
        assert mae(np.array([5.0]), np.array([3.0])) == pytest.approx(2.0)


class TestRMSE:
    def test_basic(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 4.0])
        # MSE = (0 + 0 + 1) / 3 = 1/3, RMSE = sqrt(1/3)
        expected = np.sqrt(1.0 / 3.0)
        assert rmse(actual, predicted) == pytest.approx(expected)

    def test_perfect_prediction(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert rmse(actual, actual) == pytest.approx(0.0)

    def test_rmse_ge_mae(self):
        """RMSE is always >= MAE (Cauchy-Schwarz inequality)."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.5, 2.5, 2.5, 3.5, 6.0])
        assert rmse(actual, predicted) >= mae(actual, predicted)


class TestMAPE:
    def test_basic(self):
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 190.0])
        # |10/100| + |10/200| = 0.1 + 0.05 = 0.15, average = 0.075
        expected = 7.5  # percentage
        assert mape(actual, predicted) == pytest.approx(expected, abs=0.1)

    def test_perfect_prediction(self):
        actual = np.array([100.0, 200.0, 300.0])
        assert mape(actual, actual) == pytest.approx(0.0, abs=0.01)


class TestSMAPE:
    def test_basic(self):
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 190.0])
        # SMAPE should be between 0 and 200
        result = smape(actual, predicted)
        assert 0.0 <= result <= 200.0

    def test_perfect_prediction(self):
        actual = np.array([100.0, 200.0])
        assert smape(actual, actual) == pytest.approx(0.0, abs=0.01)

    def test_bounded(self):
        """SMAPE should always be between 0 and 200."""
        rng = np.random.default_rng(42)
        actual = rng.uniform(1, 100, 100)
        predicted = rng.uniform(1, 100, 100)
        result = smape(actual, predicted)
        assert 0.0 <= result <= 200.0


class TestMASE:
    def test_beats_naive(self):
        """MASE < 1 means the model beats the naive baseline."""
        training = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        actual = np.array([11.0, 12.0, 13.0])
        predicted = np.array([10.8, 12.1, 13.2])  # Close predictions
        result = mase(actual, predicted, training, seasonal_period=1)
        assert result < 1.0

    def test_worse_than_naive(self):
        """MASE > 1 means the naive baseline is better."""
        training = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([6.0, 7.0])
        predicted = np.array([10.0, 2.0])  # Bad predictions
        result = mase(actual, predicted, training, seasonal_period=1)
        assert result > 1.0

    def test_perfect_prediction(self):
        training = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([6.0, 7.0])
        result = mase(actual, actual, training, seasonal_period=1)
        assert result == pytest.approx(0.0)


class TestDirectionalAccuracy:
    def test_perfect_direction(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.0, 2.5, 3.5, 4.5])  # Same direction as actual
        assert directional_accuracy(actual, predicted) == pytest.approx(100.0)

    def test_wrong_direction(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([4.0, 3.0, 2.0, 1.0])  # Opposite direction
        assert directional_accuracy(actual, predicted) == pytest.approx(0.0)

    def test_single_observation(self):
        """Cannot compute directional accuracy with fewer than 2 values."""
        actual = np.array([1.0])
        predicted = np.array([2.0])
        assert np.isnan(directional_accuracy(actual, predicted))


class TestPredictionIntervalCoverage:
    def test_full_coverage(self):
        actual = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])
        assert prediction_interval_coverage(actual, lower, upper) == pytest.approx(100.0)

    def test_no_coverage(self):
        actual = np.array([10.0, 20.0, 30.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        assert prediction_interval_coverage(actual, lower, upper) == pytest.approx(0.0)


class TestComputeAllMetrics:
    def test_returns_all_metrics(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
        training = np.arange(100, dtype=float)

        result = compute_all_metrics(actual, predicted, training_series=training)

        assert "mae" in result
        assert "rmse" in result
        assert "mape" in result
        assert "smape" in result
        assert "mase" in result
        assert "directional_accuracy" in result

    def test_without_training_series(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.1, 2.2, 2.8])

        result = compute_all_metrics(actual, predicted)
        assert np.isnan(result["mase"])
        assert not np.isnan(result["mae"])
