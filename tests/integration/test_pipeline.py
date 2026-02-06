"""Integration tests for the end-to-end forecasting pipeline.

Tests the complete flow from data loading through model training
and evaluation, using small synthetic datasets for speed.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.loader import TimeSeriesDataset
from src.data.features import FeatureEngineer, FeatureConfig
from src.data.splits import TemporalSplitter, WalkForwardSplit, WalkForwardConfig
from src.data.normalize import FitOnTrainNormalizer
from src.models.baselines import NaiveForecaster, SeasonalNaiveForecaster
from src.models.lstm import LSTMWithAttention
from src.models.transformer import PatchTSTEncoder
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.analysis import per_horizon_analysis, diebold_mariano_test


def _create_synthetic_series(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic time series with trend, seasonality, and noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="h")

    t = np.arange(n)
    trend = 0.001 * t
    daily = 5 * np.sin(2 * np.pi * t / 24)
    weekly = 2 * np.sin(2 * np.pi * t / 168)
    noise = rng.normal(0, 0.5, n)
    target = 20 + trend + daily + weekly + noise

    df = pd.DataFrame(
        {
            "feature_1": rng.normal(0, 1, n),
            "feature_2": rng.normal(0, 1, n),
            "feature_3": daily + rng.normal(0, 0.1, n),  # Correlated with target
            "OT": target,
        },
        index=dates,
    )
    return df


class TestEndToEndStatistical:
    """Test the complete pipeline for statistical models."""

    def test_naive_baseline_pipeline(self):
        df = _create_synthetic_series(500)
        splitter = TemporalSplitter()
        train_df, val_df, test_df = splitter.split(df)

        y_train = train_df["OT"].values
        y_test = test_df["OT"].values[:24]

        model = NaiveForecaster()
        model.fit(y_train)
        predictions = model.predict(24)

        metrics = compute_all_metrics(y_test, predictions, training_series=y_train)

        assert metrics["mae"] > 0
        assert metrics["rmse"] >= metrics["mae"]
        assert 0 <= metrics["smape"] <= 200

    def test_seasonal_naive_pipeline(self):
        df = _create_synthetic_series(500)
        splitter = TemporalSplitter()
        train_df, _, test_df = splitter.split(df)

        y_train = train_df["OT"].values
        y_test = test_df["OT"].values[:24]

        model = SeasonalNaiveForecaster(seasonal_period=24)
        model.fit(y_train)
        predictions = model.predict(24)

        metrics = compute_all_metrics(y_test, predictions, training_series=y_train)
        assert metrics["mae"] > 0

    def test_seasonal_naive_beats_naive(self):
        """On data with daily seasonality, seasonal naive should beat naive."""
        df = _create_synthetic_series(2000)
        splitter = TemporalSplitter()
        train_df, _, test_df = splitter.split(df)

        y_train = train_df["OT"].values
        y_test = test_df["OT"].values[:24]

        naive = NaiveForecaster().fit(y_train)
        seasonal = SeasonalNaiveForecaster(24).fit(y_train)

        naive_pred = naive.predict(24)
        seasonal_pred = seasonal.predict(24)

        naive_mae = compute_all_metrics(y_test, naive_pred)["mae"]
        seasonal_mae = compute_all_metrics(y_test, seasonal_pred)["mae"]

        # Seasonal naive should have lower error on seasonal data
        assert seasonal_mae < naive_mae


class TestEndToEndDeepLearning:
    """Test the complete pipeline for deep learning models."""

    @pytest.mark.slow
    def test_lstm_pipeline(self):
        df = _create_synthetic_series(500)

        # Feature engineering (minimal for speed)
        config = FeatureConfig(
            lag_periods=[1, 2],
            rolling_windows=[3],
            rolling_stats=["mean"],
            add_calendar=False,
            add_fourier=False,
        )
        engineer = FeatureEngineer(target_col="OT", config=config)
        df_features = engineer.transform(df)

        # Split
        splitter = TemporalSplitter()
        train_df, val_df, test_df = splitter.split(df_features)

        # Normalize
        normalizer = FitOnTrainNormalizer()
        train_norm = normalizer.fit_transform(train_df)
        val_norm = normalizer.transform(val_df)

        target_idx = list(df_features.columns).index("OT")
        n_features = len(df_features.columns)

        # Create datasets
        train_ds = TimeSeriesDataset(train_norm, target_idx, input_length=32, forecast_horizon=12)
        val_ds = TimeSeriesDataset(val_norm, target_idx, input_length=32, forecast_horizon=12)

        assert len(train_ds) > 0
        assert len(val_ds) > 0

        # Build model
        model = LSTMWithAttention(
            input_size=n_features,
            hidden_size=16,
            num_layers=1,
            forecast_horizon=12,
        )

        # Quick training (2 epochs)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = DataLoader(train_ds, batch_size=16, shuffle=False)

        model.train()
        for epoch in range(2):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred, _ = model(batch_x)
                loss = torch.nn.functional.mse_loss(pred, batch_y)
                loss.backward()
                optimizer.step()

        # Verify predictions have correct shape
        model.eval()
        with torch.no_grad():
            sample_x, sample_y = train_ds[0]
            pred = model.predict(sample_x.unsqueeze(0))
            assert pred.shape == (1, 12)

    @pytest.mark.slow
    def test_transformer_pipeline(self):
        df = _create_synthetic_series(500)

        config = FeatureConfig(
            lag_periods=[1, 2],
            rolling_windows=[3],
            rolling_stats=["mean"],
            add_calendar=False,
            add_fourier=False,
        )
        engineer = FeatureEngineer(target_col="OT", config=config)
        df_features = engineer.transform(df)

        splitter = TemporalSplitter()
        train_df, val_df, _ = splitter.split(df_features)

        normalizer = FitOnTrainNormalizer()
        train_norm = normalizer.fit_transform(train_df)

        target_idx = list(df_features.columns).index("OT")
        n_features = len(df_features.columns)

        train_ds = TimeSeriesDataset(train_norm, target_idx, input_length=32, forecast_horizon=12)
        assert len(train_ds) > 0

        model = PatchTSTEncoder(
            input_size=n_features,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            patch_length=8,
            input_length=32,
            forecast_horizon=12,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = DataLoader(train_ds, batch_size=16, shuffle=False)

        model.train()
        for epoch in range(2):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = torch.nn.functional.mse_loss(pred, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            sample_x, _ = train_ds[0]
            pred = model.predict(sample_x.unsqueeze(0))
            assert pred.shape == (1, 12)


class TestDieboldMarianoIntegration:
    """Test statistical significance testing in the pipeline."""

    def test_dm_test_with_models(self):
        df = _create_synthetic_series(500)
        splitter = TemporalSplitter()
        train_df, _, test_df = splitter.split(df)

        y_train = train_df["OT"].values
        y_test = test_df["OT"].values[:100]

        # Two different forecasters
        naive = NaiveForecaster().fit(y_train)
        seasonal = SeasonalNaiveForecaster(24).fit(y_train)

        pred1 = naive.predict(100)
        pred2 = seasonal.predict(100)

        result = diebold_mariano_test(y_test, pred1, pred2)

        assert "dm_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert 0 <= result["p_value"] <= 1


class TestPerHorizonIntegration:
    def test_horizon_analysis(self):
        rng = np.random.default_rng(42)
        actuals = rng.normal(0, 1, (10, 24))
        predictions = actuals + rng.normal(0, 0.1, (10, 24))

        horizon_errors = per_horizon_analysis(actuals, predictions)

        assert len(horizon_errors) == 24
        assert all(v >= 0 for v in horizon_errors.values())
