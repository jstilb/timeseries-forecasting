"""Tests for deep learning model architectures.

Validates model shapes, gradient flow, and basic training behavior
using small synthetic data. These tests run on CPU and complete in seconds.
"""

import numpy as np
import pytest
import torch

from src.models.lstm import LSTMWithAttention, TemporalAttention
from src.models.transformer import PatchTSTEncoder, PatchEmbedding
from src.models.baselines import NaiveForecaster, SeasonalNaiveForecaster, DriftForecaster


class TestNaiveForecaster:
    def test_prediction_is_last_value(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model = NaiveForecaster()
        model.fit(y)
        pred = model.predict(3)
        np.testing.assert_array_equal(pred, [5.0, 5.0, 5.0])

    def test_predict_before_fit_raises(self):
        model = NaiveForecaster()
        with pytest.raises(RuntimeError):
            model.predict(5)


class TestSeasonalNaiveForecaster:
    def test_seasonal_pattern_repeats(self):
        # 24-hour pattern
        pattern = np.sin(np.linspace(0, 2 * np.pi, 24))
        y = np.tile(pattern, 3)  # 3 full cycles

        model = SeasonalNaiveForecaster(seasonal_period=24)
        model.fit(y)
        pred = model.predict(24)

        np.testing.assert_array_almost_equal(pred, pattern)

    def test_insufficient_data_raises(self):
        model = SeasonalNaiveForecaster(seasonal_period=24)
        with pytest.raises(ValueError):
            model.fit(np.array([1.0, 2.0, 3.0]))


class TestDriftForecaster:
    def test_linear_trend(self):
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        model = DriftForecaster()
        model.fit(y)
        pred = model.predict(3)
        np.testing.assert_array_almost_equal(pred, [5.0, 6.0, 7.0])


class TestTemporalAttention:
    def test_output_shape(self):
        batch, seq_len, hidden = 4, 32, 64
        attention = TemporalAttention(hidden, attention_size=32)
        x = torch.randn(batch, seq_len, hidden)
        context, weights = attention(x)

        assert context.shape == (batch, hidden)
        assert weights.shape == (batch, seq_len)

    def test_weights_sum_to_one(self):
        attention = TemporalAttention(64, 32)
        x = torch.randn(2, 16, 64)
        _, weights = attention(x)

        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(2), atol=1e-5, rtol=1e-5)


class TestLSTMWithAttention:
    @pytest.fixture
    def model(self):
        return LSTMWithAttention(
            input_size=7,
            hidden_size=32,
            num_layers=1,
            forecast_horizon=24,
            dropout=0.0,
        )

    def test_forward_shape(self, model):
        x = torch.randn(4, 96, 7)
        forecast, attn_weights = model(x)

        assert forecast.shape == (4, 24)
        assert attn_weights.shape == (4, 96)

    def test_predict_shape(self, model):
        x = torch.randn(2, 96, 7)
        pred = model.predict(x)
        assert pred.shape == (2, 24)

    def test_gradient_flow(self, model):
        """Verify gradients flow through the entire model."""
        x = torch.randn(2, 96, 7)
        target = torch.randn(2, 24)

        forecast, _ = model(x)
        loss = torch.nn.functional.mse_loss(forecast, target)
        loss.backward()

        # Check that all parameters received gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_training_loss_decreases(self, model):
        """Basic sanity check: loss should decrease over several steps."""
        x = torch.randn(8, 96, 7)
        target = torch.randn(8, 24)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            forecast, _ = model(x)
            loss = torch.nn.functional.mse_loss(forecast, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease (final < initial)
        assert losses[-1] < losses[0]


class TestPatchEmbedding:
    def test_output_shape(self):
        embed = PatchEmbedding(input_size=7, patch_length=16, d_model=64)
        x = torch.randn(4, 96, 7)
        result = embed(x)

        # 96 / 16 = 6 patches
        assert result.shape == (4, 6, 64)

    def test_different_patch_sizes(self):
        for patch_len in [8, 16, 32]:
            embed = PatchEmbedding(input_size=7, patch_length=patch_len, d_model=64)
            x = torch.randn(2, 96, 7)
            result = embed(x)
            expected_patches = (96 - patch_len) // patch_len + 1
            assert result.shape[1] == expected_patches


class TestPatchTSTEncoder:
    @pytest.fixture
    def model(self):
        return PatchTSTEncoder(
            input_size=7,
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            patch_length=16,
            input_length=96,
            forecast_horizon=24,
            dropout=0.0,
        )

    def test_forward_shape(self, model):
        x = torch.randn(4, 96, 7)
        forecast = model(x)
        assert forecast.shape == (4, 24)

    def test_predict_shape(self, model):
        x = torch.randn(2, 96, 7)
        pred = model.predict(x)
        assert pred.shape == (2, 24)

    def test_gradient_flow(self, model):
        x = torch.randn(2, 96, 7)
        target = torch.randn(2, 24)

        forecast = model(x)
        loss = torch.nn.functional.mse_loss(forecast, target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_training_loss_decreases(self, model):
        x = torch.randn(8, 96, 7)
        target = torch.randn(8, 24)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            forecast = model(x)
            loss = torch.nn.functional.mse_loss(forecast, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0]
