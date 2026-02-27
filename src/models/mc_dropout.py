"""
Monte Carlo Dropout — Uncertainty Quantification
ISC-5784: MC Dropout producing prediction intervals (upper/lower bounds)

Implements:
- enable_dropout(): activates dropout at inference time
- mc_dropout_predict(): runs N forward passes and returns mean + CI
- MCDropoutLSTM: LSTM with dropout at inference time
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple


def enable_dropout(model: nn.Module) -> None:
    """
    Enable dropout at inference time for Monte Carlo Dropout.
    
    Standard PyTorch sets dropout to eval mode (disabled) during model.eval().
    This function selectively re-enables dropout layers while keeping
    BatchNorm in eval mode (crucial — BN should NOT use batch statistics at test time).
    
    Args:
        model: PyTorch model with dropout layers
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    num_samples: int = 100,
    prediction_length: int = 24,
) -> Dict[str, np.ndarray]:
    """
    Run Monte Carlo Dropout inference.
    
    Produces probabilistic predictions by running num_samples stochastic
    forward passes with dropout enabled, then aggregates the distribution.
    
    Args:
        model: PyTorch model (must have dropout layers)
        x: Input tensor of shape [1, context_length, features]
        num_samples: Number of MC samples (stochastic forward passes)
        prediction_length: Number of steps to forecast
    
    Returns:
        Dict with keys:
          - 'mean': point estimate (shape: [prediction_length])
          - 'std': standard deviation (shape: [prediction_length])  
          - 'lower': 10th percentile — lower prediction bound
          - 'upper': 90th percentile — upper prediction bound
          - 'samples': all sample paths (shape: [num_samples, prediction_length])
    """
    model.eval()
    enable_dropout(model)  # Re-enable dropout for stochastic inference
    
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            samples.append(pred.squeeze().cpu().numpy())
    
    samples = np.array(samples)  # [num_samples, prediction_length]
    
    return {
        "mean": samples.mean(axis=0),
        "std": samples.std(axis=0),
        "lower": np.percentile(samples, 10, axis=0),
        "upper": np.percentile(samples, 90, axis=0),
        "samples": samples,
    }


class MCDropoutLSTM(nn.Module):
    """
    LSTM with Monte Carlo Dropout for uncertainty quantification.
    
    Dropout is applied:
    - After input projection
    - Between LSTM layers (via LSTM dropout)
    - Before output projection
    
    This enables MC Dropout at inference: running with dropout active
    gives a distribution over predictions rather than a point estimate.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        prediction_length: int = 24,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        self.dropout_rate = dropout_rate

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(p=dropout_rate)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        self.output_dropout = nn.Dropout(p=dropout_rate)
        self.output_proj = nn.Linear(hidden_size, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_size]
        
        Returns:
            predictions: [batch, prediction_length]
        """
        # Project input features
        x = self.input_proj(x)  # [B, T, H]
        x = self.input_dropout(x)

        # LSTM
        lstm_out, _ = self.lstm(x)  # [B, T, H]
        last_hidden = lstm_out[:, -1, :]  # Use last time step

        # Output projection
        out = self.output_dropout(last_hidden)
        predictions = self.output_proj(out)  # [B, prediction_length]
        return predictions

    def predict_with_uncertainty(
        self, x: torch.Tensor, num_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Convenience method: run MC Dropout and return uncertainty estimates.
        
        Returns dict with 'mean', 'std', 'lower', 'upper', 'samples'.
        """
        return mc_dropout_predict(self, x, num_samples=num_samples)


def run_mc_dropout_demo() -> Dict[str, np.ndarray]:
    """
    Demonstration of MC Dropout uncertainty quantification.
    
    Creates a simple time series, trains a minimal MCDropoutLSTM,
    and shows prediction intervals.
    
    Returns the uncertainty dict for inspection.
    """
    import torch.optim as optim

    print("Monte Carlo Dropout Uncertainty Quantification Demo")
    print("=" * 55)

    # Generate synthetic time series
    rng = np.random.default_rng(42)
    t = np.arange(500)
    series = (
        np.sin(2 * np.pi * t / 24)
        + 0.3 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 0.1, 500)
    ).astype(np.float32)

    context_length = 48
    prediction_length = 24
    input_size = 1

    # Prepare data
    X = torch.tensor(series[:300].reshape(-1, context_length, input_size), dtype=torch.float32)
    y = torch.tensor(series[context_length:300 + prediction_length].reshape(-1, prediction_length), dtype=torch.float32)

    # Build model
    model = MCDropoutLSTM(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout_rate=0.2,
        prediction_length=prediction_length,
    )

    # Quick training (5 epochs for demo)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(5):
        optimizer.zero_grad()
        pred = model(X[:4])
        loss = criterion(pred, y[:4])
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch + 1}/5, Loss: {loss.item():.4f}")

    # MC Dropout inference
    x_test = torch.tensor(series[300:300 + context_length].reshape(1, context_length, input_size))
    uncertainty = model.predict_with_uncertainty(x_test, num_samples=100)

    print(f"\nMC Dropout Results (100 samples):")
    print(f"  Point estimate (mean): {uncertainty['mean'][:6].round(3)}")
    print(f"  Std (uncertainty):     {uncertainty['std'][:6].round(3)}")
    print(f"  Lower bound (10th %%): {uncertainty['lower'][:6].round(3)}")
    print(f"  Upper bound (90th %%): {uncertainty['upper'][:6].round(3)}")
    print(f"  Mean CI width:         {np.mean(uncertainty['upper'] - uncertainty['lower']):.4f}")

    return uncertainty


if __name__ == "__main__":
    result = run_mc_dropout_demo()
    print("\nAll MC Dropout prediction intervals generated successfully.")
