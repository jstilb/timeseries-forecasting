"""Tests for fit-on-train normalization.

Critical for preventing data leakage: the normalizer must compute
statistics only from training data, never from validation or test data.
"""

import numpy as np
import pytest

from src.data.normalize import FitOnTrainNormalizer


class TestFitOnTrainNormalizer:
    def test_standard_normalization(self):
        train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalizer = FitOnTrainNormalizer(method="standard")
        result = normalizer.fit_transform(train)

        # After standard normalization, mean should be ~0 and std ~1
        assert np.abs(result.mean(axis=0)).max() < 1e-6
        assert np.abs(result.std(axis=0) - 1.0).max() < 0.1

    def test_minmax_normalization(self):
        train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalizer = FitOnTrainNormalizer(method="minmax")
        result = normalizer.fit_transform(train)

        # Min should be 0, max should be 1
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_inverse_transform_roundtrip(self):
        """fit -> transform -> inverse_transform should recover original values."""
        rng = np.random.default_rng(42)
        original = rng.normal(50, 10, (100, 5))

        normalizer = FitOnTrainNormalizer(method="standard")
        transformed = normalizer.fit_transform(original)
        recovered = normalizer.inverse_transform(transformed)

        np.testing.assert_allclose(recovered, original, atol=1e-6)

    def test_minmax_inverse_transform(self):
        rng = np.random.default_rng(42)
        original = rng.uniform(0, 100, (50, 3))

        normalizer = FitOnTrainNormalizer(method="minmax")
        transformed = normalizer.fit_transform(original)
        recovered = normalizer.inverse_transform(transformed)

        np.testing.assert_allclose(recovered, original, atol=1e-6)

    def test_fit_on_train_only(self):
        """Statistics should come from training data, not test data."""
        train = np.array([[0.0], [1.0], [2.0]])  # mean=1, std~0.816
        test = np.array([[100.0], [200.0]])  # Very different distribution

        normalizer = FitOnTrainNormalizer(method="standard")
        normalizer.fit(train)

        # Transform test using TRAIN statistics
        test_normalized = normalizer.transform(test)

        # Test values should be far from 0 (since they are far from train mean)
        assert np.abs(test_normalized[0, 0]) > 50

    def test_transform_before_fit_raises(self):
        normalizer = FitOnTrainNormalizer()
        with pytest.raises(RuntimeError, match="must be fit"):
            normalizer.transform(np.array([1.0, 2.0]))

    def test_inverse_transform_target(self):
        """Inverse transform of a single target column."""
        train = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])

        normalizer = FitOnTrainNormalizer(method="standard")
        normalizer.fit(train)

        # Normalize just the target column index 1
        target_normalized = normalizer.transform(train)[:, 1]
        target_recovered = normalizer.inverse_transform_target(target_normalized, target_idx=1)

        np.testing.assert_allclose(target_recovered, train[:, 1], atol=1e-6)

    def test_constant_column_handling(self):
        """Constant columns should not cause division by zero."""
        train = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])
        normalizer = FitOnTrainNormalizer(method="standard")
        result = normalizer.fit_transform(train)

        # Should not contain NaN or inf
        assert np.all(np.isfinite(result))

    def test_is_fitted_property(self):
        normalizer = FitOnTrainNormalizer()
        assert normalizer.is_fitted is False

        normalizer.fit(np.array([[1.0], [2.0]]))
        assert normalizer.is_fitted is True
