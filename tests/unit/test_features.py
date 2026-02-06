"""Tests for the feature engineering pipeline.

Validates that:
1. Lag features are correctly computed (no future leakage)
2. Rolling statistics only look backward
3. Calendar features are deterministic and correct
4. Fourier features have correct periodicity
5. The full pipeline produces expected output shapes
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features import FeatureEngineer, FeatureConfig


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with hourly data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=200, freq="h")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "HUFL": rng.normal(0, 1, 200),
            "HULL": rng.normal(0, 1, 200),
            "OT": rng.normal(20, 5, 200),
        },
        index=dates,
    )
    return df


@pytest.fixture
def engineer():
    """Create a feature engineer with small config for testing."""
    config = FeatureConfig(
        lag_periods=[1, 2, 3],
        rolling_windows=[3, 6],
        rolling_stats=["mean", "std"],
        add_calendar=True,
        add_fourier=True,
        fourier_periods=[24],
        fourier_order=2,
    )
    return FeatureEngineer(target_col="OT", config=config)


class TestLagFeatures:
    def test_lag_values_correct(self, sample_df, engineer):
        result = engineer.create_lag_features(sample_df)

        # Lag 1 should equal the previous value
        for i in range(1, len(result)):
            assert result["OT_lag_1"].iloc[i] == result["OT"].iloc[i - 1]

    def test_lag_creates_nan_at_start(self, sample_df, engineer):
        result = engineer.create_lag_features(sample_df)
        # First row should have NaN for lag 1
        assert pd.isna(result["OT_lag_1"].iloc[0])

    def test_no_future_leakage(self, sample_df, engineer):
        """Lag features must never contain future information."""
        result = engineer.create_lag_features(sample_df)
        for lag in engineer.config.lag_periods:
            col = f"OT_lag_{lag}"
            for i in range(lag, len(result)):
                # Each lag value should equal the value from `lag` steps ago
                assert result[col].iloc[i] == result["OT"].iloc[i - lag]


class TestRollingFeatures:
    def test_rolling_mean_backward_only(self, sample_df, engineer):
        result = engineer.create_rolling_features(sample_df)

        # Manually compute rolling mean for verification
        for i in range(3, len(result)):
            window_vals = result["OT"].iloc[i - 3 : i].values  # Exclude current
            expected_mean = np.mean(window_vals)
            actual_mean = result["OT_roll_mean_3"].iloc[i]
            assert actual_mean == pytest.approx(expected_mean, abs=1e-6), (
                f"Rolling mean mismatch at index {i}"
            )

    def test_rolling_creates_expected_columns(self, sample_df, engineer):
        result = engineer.create_rolling_features(sample_df)
        expected_cols = [
            "OT_roll_mean_3", "OT_roll_std_3",
            "OT_roll_mean_6", "OT_roll_std_6",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


class TestCalendarFeatures:
    def test_hour_range(self, sample_df, engineer):
        result = engineer.create_calendar_features(sample_df)
        assert result["hour"].min() >= 0
        assert result["hour"].max() <= 23

    def test_day_of_week_range(self, sample_df, engineer):
        result = engineer.create_calendar_features(sample_df)
        assert result["day_of_week"].min() >= 0
        assert result["day_of_week"].max() <= 6

    def test_cyclical_encoding_range(self, sample_df, engineer):
        """Sine/cosine features should be in [-1, 1]."""
        result = engineer.create_calendar_features(sample_df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert result[col].min() >= -1.0
            assert result[col].max() <= 1.0

    def test_weekend_flag(self, sample_df, engineer):
        result = engineer.create_calendar_features(sample_df)
        # Verify weekend flag matches day_of_week
        for i in range(len(result)):
            dow = result["day_of_week"].iloc[i]
            is_wknd = result["is_weekend"].iloc[i]
            if dow >= 5:
                assert is_wknd == 1
            else:
                assert is_wknd == 0


class TestFourierFeatures:
    def test_fourier_range(self, sample_df, engineer):
        """Fourier features should be in [-1, 1]."""
        result = engineer.create_fourier_features(sample_df)
        fourier_cols = [c for c in result.columns if c.startswith("fourier_")]
        for col in fourier_cols:
            assert result[col].min() >= -1.0
            assert result[col].max() <= 1.0

    def test_fourier_count(self, sample_df, engineer):
        result = engineer.create_fourier_features(sample_df)
        fourier_cols = [c for c in result.columns if c.startswith("fourier_")]
        # 1 period * 2 orders * 2 (sin+cos) = 4
        expected = len(engineer.config.fourier_periods) * engineer.config.fourier_order * 2
        assert len(fourier_cols) == expected


class TestFullPipeline:
    def test_output_shape(self, sample_df, engineer):
        result = engineer.transform(sample_df, drop_na=True)
        # Should have more columns than input
        assert len(result.columns) > len(sample_df.columns)
        # Should have fewer rows (NaN from lags dropped)
        assert len(result) < len(sample_df)

    def test_no_nan_after_drop(self, sample_df, engineer):
        result = engineer.transform(sample_df, drop_na=True)
        assert result.isna().sum().sum() == 0

    def test_preserves_index(self, sample_df, engineer):
        result = engineer.transform(sample_df, drop_na=True)
        # Index should still be datetime
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_original_columns_preserved(self, sample_df, engineer):
        result = engineer.transform(sample_df, drop_na=True)
        for col in sample_df.columns:
            assert col in result.columns
