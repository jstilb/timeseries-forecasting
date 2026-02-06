"""Transformer encoder for time series forecasting (PatchTST-inspired).

Implements a Transformer-based architecture inspired by PatchTST (Nie et al., 2023).
The key insight from PatchTST is that patching (grouping consecutive timesteps)
reduces the effective sequence length, making self-attention more efficient and
enabling the model to capture both local patterns (within patches) and global
dependencies (between patches).

Architecture:
    Input -> Patching -> Positional Encoding -> Transformer Encoder -> Flatten -> FC -> Forecast

Reference:
    Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (2023)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Convert time series into patch embeddings.

    Groups consecutive timesteps into non-overlapping patches, then projects
    each patch to the model dimension. This reduces the sequence length by
    a factor of patch_length, making self-attention O(n^2) much cheaper.

    Args:
        input_size: Number of input features per timestep.
        patch_length: Number of consecutive timesteps per patch.
        d_model: Output embedding dimensionality.
        stride: Stride between patches (default = patch_length for non-overlapping).
    """

    def __init__(
        self,
        input_size: int,
        patch_length: int = 16,
        d_model: int = 128,
        stride: int | None = None,
    ):
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride or patch_length
        self.projection = nn.Linear(input_size * patch_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create patch embeddings from input sequence.

        Args:
            x: Input tensor, shape (batch, seq_len, input_size).

        Returns:
            Patch embeddings, shape (batch, n_patches, d_model).
        """
        batch_size, seq_len, n_features = x.shape

        # Unfold into patches: (batch, n_patches, patch_length, n_features)
        patches = x.unfold(1, self.patch_length, self.stride)  # (batch, n_patches, features, patch_len)
        patches = patches.permute(0, 1, 3, 2)  # (batch, n_patches, patch_len, features)
        n_patches = patches.size(1)

        # Flatten each patch and project
        patches = patches.reshape(batch_size, n_patches, -1)  # (batch, n_patches, patch_len * features)
        embeddings = self.projection(patches)  # (batch, n_patches, d_model)

        return embeddings


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer.

    Adds position information to patch embeddings so the Transformer
    can distinguish between patches at different temporal positions.

    Args:
        d_model: Embedding dimensionality.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            Position-encoded tensor, same shape as input.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PatchTSTEncoder(nn.Module):
    """Transformer encoder for time series, inspired by PatchTST.

    Processes multi-variate time series through:
    1. Patching: Group consecutive timesteps to reduce sequence length
    2. Embedding: Project patches to model dimension
    3. Positional encoding: Add temporal position information
    4. Transformer encoder: Self-attention over patches
    5. Projection: Map encoded representations to forecast

    Compared to LSTM:
    - Better at capturing long-range dependencies
    - More parallelizable (no sequential hidden state)
    - Typically better for longer forecast horizons
    - But requires more data to train effectively

    Args:
        input_size: Number of input features per timestep.
        d_model: Transformer model dimensionality.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer encoder layers.
        d_ff: Feed-forward hidden dimensionality.
        patch_length: Number of timesteps per patch.
        input_length: Total input sequence length.
        forecast_horizon: Number of future timesteps to predict.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_size: int = 7,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        patch_length: int = 16,
        input_length: int = 96,
        forecast_horizon: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.name = "PatchTST"

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            input_size=input_size,
            patch_length=patch_length,
            d_model=d_model,
        )

        # Calculate number of patches
        self.n_patches = (input_length - patch_length) // patch_length + 1

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.n_patches + 1, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)

        # Output projection: flatten all patch representations and project to forecast
        self.output_proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.n_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, forecast_horizon),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the PatchTST encoder.

        Args:
            x: Input tensor, shape (batch, input_length, input_size).

        Returns:
            Forecast tensor, shape (batch, forecast_horizon).
        """
        # Create patch embeddings
        patches = self.patch_embedding(x)  # (batch, n_patches, d_model)

        # Add positional encoding
        patches = self.pos_encoding(patches)

        # Transformer encoder
        encoded = self.transformer_encoder(patches)  # (batch, n_patches, d_model)

        # Layer norm
        encoded = self.layer_norm(encoded)

        # Project to forecast
        forecast = self.output_proj(encoded)  # (batch, forecast_horizon)

        return forecast

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Generate forecast (inference mode).

        Args:
            x: Input tensor, shape (batch, input_length, input_size).

        Returns:
            Predicted values, shape (batch, forecast_horizon).
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
