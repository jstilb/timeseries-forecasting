"""LSTM with attention mechanism for time series forecasting.

Implements a multi-layer LSTM encoder with a temporal attention mechanism
that learns to weight different timesteps in the input window when
generating forecasts. This is more interpretable than a vanilla LSTM
because the attention weights show which historical timesteps the model
considers most relevant.

Architecture:
    Input -> LSTM Encoder -> Temporal Attention -> FC Projection -> Forecast
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Scaled dot-product temporal attention over LSTM hidden states.

    Learns a query vector that attends over all encoder timesteps,
    producing a weighted context vector that summarizes the input
    sequence based on relevance to forecasting.

    Args:
        hidden_size: Dimensionality of LSTM hidden states.
        attention_size: Dimensionality of the attention key space.
    """

    def __init__(self, hidden_size: int, attention_size: int = 64):
        super().__init__()
        self.query = nn.Linear(hidden_size, attention_size, bias=False)
        self.key = nn.Linear(hidden_size, attention_size, bias=False)
        self.scale = math.sqrt(attention_size)

    def forward(
        self, encoder_outputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted context vector.

        Args:
            encoder_outputs: LSTM outputs, shape (batch, seq_len, hidden_size).
            mask: Optional boolean mask, shape (batch, seq_len). True = masked.

        Returns:
            context: Weighted sum, shape (batch, hidden_size).
            weights: Attention weights, shape (batch, seq_len).
        """
        # Use last hidden state as query
        query = self.query(encoder_outputs[:, -1:, :])  # (batch, 1, attn_size)
        keys = self.key(encoder_outputs)  # (batch, seq_len, attn_size)

        # Scaled dot-product attention
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale  # (batch, 1, seq_len)
        scores = scores.squeeze(1)  # (batch, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # Weighted sum of encoder outputs
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden)
        context = context.squeeze(1)  # (batch, hidden_size)

        return context, weights


class LSTMWithAttention(nn.Module):
    """LSTM encoder with temporal attention for multi-step forecasting.

    The model processes multi-variate input sequences through a stacked
    LSTM, applies temporal attention to weight historical timesteps,
    then projects the attention context to a multi-step forecast.

    This architecture is particularly effective when:
    - The input contains multiple correlated features
    - Certain historical timesteps are more predictive than others
    - The forecast horizon is relatively short (< 48 steps)

    For longer horizons, the Transformer model typically performs better
    because attention can capture longer-range dependencies.

    Args:
        input_size: Number of input features per timestep.
        hidden_size: LSTM hidden state dimensionality.
        num_layers: Number of stacked LSTM layers.
        forecast_horizon: Number of future timesteps to predict.
        dropout: Dropout probability between LSTM layers.
        attention_size: Dimensionality of the attention projection.
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 2,
        forecast_horizon: int = 24,
        dropout: float = 0.1,
        attention_size: int = 64,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.name = "LSTM-Attention"

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Temporal attention
        self.attention = TemporalAttention(hidden_size, attention_size)

        # Output projection: attention context -> forecast
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_horizon),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform for stability."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode input sequence and generate forecast.

        Args:
            x: Input tensor, shape (batch, input_length, input_size).

        Returns:
            forecast: Predicted values, shape (batch, forecast_horizon).
            attention_weights: Temporal attention weights, shape (batch, input_length).
        """
        batch_size = x.size(0)

        # Project input features to hidden dimension
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_size)

        # Temporal attention over encoder states
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden_size)

        # Project to forecast
        forecast = self.output_proj(context)  # (batch, forecast_horizon)

        return forecast, attn_weights

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Generate forecast without attention weights (for inference).

        Args:
            x: Input tensor, shape (batch, input_length, input_size).

        Returns:
            Predicted values, shape (batch, forecast_horizon).
        """
        self.eval()
        with torch.no_grad():
            forecast, _ = self.forward(x)
        return forecast
