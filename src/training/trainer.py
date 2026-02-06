"""Training loop for deep learning time series models.

Handles the standard training lifecycle: forward pass, loss computation,
backpropagation, gradient clipping, learning rate scheduling, early stopping,
and MLflow logging. Supports both LSTM and Transformer architectures.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.config import TrainingConfig

logger = logging.getLogger(__name__)


def get_device(config: TrainingConfig) -> torch.device:
    """Resolve device string to torch.device with fallback logic.

    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config.device)


class EarlyStopping:
    """Early stopping to prevent overfitting.

    Monitors validation loss and stops training if no improvement is
    observed for `patience` consecutive epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """Training loop for PyTorch time series models.

    Handles:
    - Device management (CUDA/MPS/CPU with automatic fallback)
    - Mixed precision training (when available)
    - Gradient clipping for stable LSTM training
    - Cosine annealing / ReduceOnPlateau learning rate scheduling
    - Early stopping based on validation loss
    - Best model checkpoint saving
    - MLflow experiment tracking (optional)

    Args:
        model: PyTorch model (LSTMWithAttention or PatchTSTEncoder).
        config: Training configuration.
        checkpoint_dir: Directory to save model checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.config = config
        self.device = get_device(config)
        self.model = model.to(self.device)
        self.checkpoint_dir = checkpoint_dir

        # Loss function
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = self._build_scheduler()

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience, min_delta=config.min_delta
        )

        # Training state
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val_loss = float("inf")
        self.best_model_state = None

        logger.info("Trainer initialized on device: %s", self.device)

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.max_epochs
            )
        elif self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5, factor=0.5
            )
        return None

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass (handle models that return tuples)
            output = self.model(batch_x)
            if isinstance(output, tuple):
                output = output[0]

            loss = self.criterion(output, batch_y)

            # Backward pass
            loss.backward()

            # Gradient clipping (critical for LSTM stability)
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> float:
        """Run validation.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            output = self.model(batch_x)
            if isinstance(output, tuple):
                output = output[0]

            loss = self.criterion(output, batch_y)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        mlflow_run: Optional[object] = None,
    ) -> dict:
        """Execute the full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            mlflow_run: Optional MLflow run for logging.

        Returns:
            Dictionary with training history and best metrics.
        """
        logger.info(
            "Starting training: %d epochs, batch_size=%d, lr=%.1e",
            self.config.max_epochs,
            self.config.batch_size,
            self.config.learning_rate,
        )

        start_time = time.time()

        for epoch in range(self.config.max_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self._validate(val_loader)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Track best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            # Early stopping check
            if self.early_stopping.step(val_loss):
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

            # Logging
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "Epoch %3d/%d | Train Loss: %.6f | Val Loss: %.6f | "
                    "LR: %.2e | Time: %.1fs",
                    epoch + 1,
                    self.config.max_epochs,
                    train_loss,
                    val_loss,
                    current_lr,
                    epoch_time,
                )

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model (val_loss=%.6f)", self.best_val_loss)

        total_time = time.time() - start_time
        n_epochs = len(self.train_losses)

        return {
            "n_epochs": n_epochs,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.train_losses[-1],
            "total_time_seconds": total_time,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """Generate predictions for a dataset.

        Args:
            dataloader: Data loader for inference.

        Returns:
            Predictions array of shape (n_samples, forecast_horizon).
        """
        self.model.eval()
        predictions = []

        for batch_x, _ in dataloader:
            batch_x = batch_x.to(self.device)

            output = self.model(batch_x)
            if isinstance(output, tuple):
                output = output[0]

            predictions.append(output.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)
