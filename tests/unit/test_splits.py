"""Tests for temporal splitting strategies.

Validates that:
1. Temporal splits maintain chronological order (no future leakage)
2. Split ratios produce correct sizes
3. Walk-forward splits never look ahead
4. All indices are covered exactly once per fold
"""

import numpy as np
import pandas as pd
import pytest

from src.data.splits import (
    TemporalSplitter,
    SplitConfig,
    WalkForwardSplit,
    WalkForwardConfig,
)


@pytest.fixture
def sample_df():
    """Create a DataFrame with 1000 hourly timestamps."""
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame({"value": rng.normal(0, 1, 1000)}, index=dates)


class TestTemporalSplitter:
    def test_ratios_sum_to_one(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            SplitConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_default_split_sizes(self, sample_df):
        splitter = TemporalSplitter()
        train, val, test = splitter.split(sample_df)

        assert len(train) == 700
        assert len(val) == 100
        assert len(test) == 200

    def test_temporal_ordering(self, sample_df):
        """Train must end before val begins, val before test."""
        splitter = TemporalSplitter()
        train, val, test = splitter.split(sample_df)

        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_no_overlap(self, sample_df):
        """No index should appear in more than one split."""
        splitter = TemporalSplitter()
        train, val, test = splitter.split(sample_df)

        train_set = set(train.index)
        val_set = set(val.index)
        test_set = set(test.index)

        assert len(train_set & val_set) == 0
        assert len(val_set & test_set) == 0
        assert len(train_set & test_set) == 0

    def test_full_coverage(self, sample_df):
        """All original indices should appear in exactly one split."""
        splitter = TemporalSplitter()
        train, val, test = splitter.split(sample_df)

        total = len(train) + len(val) + len(test)
        assert total == len(sample_df)

    def test_custom_ratios(self, sample_df):
        config = SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        splitter = TemporalSplitter(config)
        train, val, test = splitter.split(sample_df)

        assert len(train) == 800
        assert len(val) == 100

    def test_split_arrays(self):
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        splitter = TemporalSplitter()

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = splitter.split_arrays(X, y)

        assert len(X_train) == 70
        assert len(X_val) == 10
        assert len(X_test) == 20

        # Values should be in order
        assert y_train[-1] < y_val[0]
        assert y_val[-1] < y_test[0]


class TestWalkForwardSplit:
    def test_correct_number_of_folds(self):
        config = WalkForwardConfig(n_splits=5, min_train_size=100, test_size=50)
        splitter = WalkForwardSplit(config)

        folds = list(splitter.split(n_samples=1000))
        assert len(folds) == 5

    def test_train_before_test(self):
        """Training indices must always precede test indices."""
        config = WalkForwardConfig(n_splits=3, min_train_size=100, test_size=50)
        splitter = WalkForwardSplit(config)

        for train_idx, test_idx in splitter.split(n_samples=500):
            assert train_idx.max() < test_idx.min()

    def test_no_overlap_within_fold(self):
        config = WalkForwardConfig(n_splits=3, min_train_size=100, test_size=50)
        splitter = WalkForwardSplit(config)

        for train_idx, test_idx in splitter.split(n_samples=500):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_expanding_window_grows(self):
        """In expanding mode, training set should grow with each fold."""
        config = WalkForwardConfig(
            n_splits=3, min_train_size=100, test_size=50, strategy="expanding"
        )
        splitter = WalkForwardSplit(config)

        train_sizes = []
        for train_idx, _ in splitter.split(n_samples=500):
            train_sizes.append(len(train_idx))

        # Each fold should have >= previous training size
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_gap_between_train_and_test(self):
        """When gap > 0, there should be a buffer between train and test."""
        config = WalkForwardConfig(
            n_splits=3, min_train_size=100, test_size=50, gap=10
        )
        splitter = WalkForwardSplit(config)

        for train_idx, test_idx in splitter.split(n_samples=500):
            gap = test_idx.min() - train_idx.max()
            assert gap >= 10

    def test_insufficient_data_raises(self):
        """Should raise error when not enough data for requested splits."""
        config = WalkForwardConfig(
            n_splits=10, min_train_size=500, test_size=200
        )
        splitter = WalkForwardSplit(config)

        with pytest.raises(ValueError, match="Not enough data"):
            list(splitter.split(n_samples=100))
