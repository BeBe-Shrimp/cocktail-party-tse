"""Training callbacks: checkpointing, early stopping, logging."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointCallback:
    """Manages model checkpoint saving during training.

    Features:
    - Save top-k checkpoints based on a monitored metric
    - Save periodic checkpoints (every N epochs)
    - Save latest checkpoint for training resumption
    - Save optimizer and scheduler state for exact resumption

    Args:
        checkpoint_dir: Directory to save checkpoints.
        save_top_k: Number of best checkpoints to keep.
        monitor: Metric name to monitor for best checkpoint selection.
        mode: ``"min"`` or ``"max"`` — whether lower or higher monitor is better.
        save_last: Whether to always save the latest checkpoint.
        save_period: Save every N epochs (0 = disabled).
        filename_prefix: Prefix for checkpoint filenames.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        save_top_k: int = 3,
        monitor: str = "val_loss",
        mode: str = "min",
        save_last: bool = True,
        save_period: int = 0,
        filename_prefix: str = "auditory_tse",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.save_period = save_period
        self.filename_prefix = filename_prefix

        # Track best checkpoints
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_checkpoints: List[Dict[str, Any]] = []

        # Metric history
        self.history_path = self.checkpoint_dir / "checkpoint_history.json"
        self.history: List[Dict[str, Any]] = []
        if self.history_path.exists():
            with open(self.history_path, "r") as f:
                self.history = json.load(f)
            # Restore best score
            if self.history:
                scores = [entry.get(monitor, float("inf") if mode == "min" else float("-inf")) for entry in self.history]
                self.best_score = min(scores) if mode == "min" else max(scores)

    def is_better(self, current: float) -> bool:
        """Check if current score is better than best.

        Args:
            current: Current metric value.

        Returns:
            True if current is better than the best recorded score.
        """
        if self.mode == "min":
            return current < self.best_score
        else:
            return current > self.best_score

    def on_epoch_end(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Called at the end of each epoch. Saves checkpoints as needed.

        Args:
            epoch: Current epoch (0-indexed).
            model: The model to save.
            optimizer: Optimizer state to include.
            scheduler: Scheduler state to include.
            metrics: Dictionary of metric values for this epoch.

        Returns:
            True if this is a new best checkpoint.
        """
        metrics = metrics or {}
        current_score = metrics.get(self.monitor)

        is_best = False
        if current_score is not None and self.is_better(current_score):
            is_best = True
            self.best_score = current_score

        # Save checkpoint data
        checkpoint_data = self._build_checkpoint(epoch, model, optimizer, scheduler, metrics)

        # Determine save path
        save_path = self._get_save_path(epoch, is_best)

        # Save
        torch.save(checkpoint_data, save_path)
        logger.info(f"Checkpoint saved: {save_path} | {self.monitor}={current_score}")

        # Update history
        history_entry = {
            "epoch": epoch,
            "path": str(save_path),
            "is_best": is_best,
        }
        history_entry.update(metrics)
        self.history.append(history_entry)

        # Manage top-k checkpoints
        if is_best:
            self.best_checkpoints.append(history_entry)
            # Keep only top-k
            self.best_checkpoints.sort(
                key=lambda x: x.get(self.monitor, 0),
                reverse=(self.mode == "max"),
            )
            self.best_checkpoints = self.best_checkpoints[:self.save_top_k]
            # Remove checkpoints no longer in top-k
            kept_paths = {entry["path"] for entry in self.best_checkpoints}
            for f in self.checkpoint_dir.glob(f"{self.filename_prefix}_best_epoch*.ckpt"):
                if str(f) not in kept_paths:
                    f.unlink()
                    logger.debug(f"Removed old best checkpoint: {f}")

        # Save latest
        if self.save_last:
            latest_path = self.checkpoint_dir / f"{self.filename_prefix}_latest.ckpt"
            torch.save(checkpoint_data, latest_path)

        # Periodic save
        if self.save_period > 0 and (epoch + 1) % self.save_period == 0:
            periodic_path = self.checkpoint_dir / f"{self.filename_prefix}_epoch{epoch+1:04d}.ckpt"
            torch.save(checkpoint_data, periodic_path)

        # Save history
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return is_best

    def _build_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """Build checkpoint dictionary."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        return checkpoint

    def _get_save_path(self, epoch: int, is_best: bool) -> Path:
        """Generate checkpoint file path.

        Args:
            epoch: Current epoch.
            is_best: Whether this is a new best checkpoint.

        Returns:
            Checkpoint file path.
        """
        if is_best:
            return self.checkpoint_dir / f"{self.filename_prefix}_best_epoch{epoch+1:04d}.ckpt"
        return self.checkpoint_dir / f"{self.filename_prefix}_epoch{epoch+1:04d}.ckpt"

    def load_best(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Load the best checkpoint.

        Args:
            model: Model to load weights into.

        Returns:
            Checkpoint data dictionary.

        Raises:
            FileNotFoundError: If no best checkpoint is found.
        """
        if not self.best_checkpoints:
            # Try loading latest
            latest = self.checkpoint_dir / f"{self.filename_prefix}_latest.ckpt"
            if latest.exists():
                checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                return checkpoint
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")

        best_entry = self.best_checkpoints[0]
        checkpoint = torch.load(best_entry["path"], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint


class EarlyStopping:
    """Early stopping to prevent overfitting.

    Monitors a validation metric and stops training when it stops improving.

    Args:
        monitor: Metric to monitor.
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: ``"min"`` or ``"max"``.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, metrics: Dict[str, float]) -> bool:
        """Check if training should stop.

        Args:
            metrics: Current epoch metrics.

        Returns:
            True if training should stop.
        """
        current = metrics.get(self.monitor)
        if current is None:
            return False

        if self.mode == "min":
            improved = current < self.best_score - self.min_delta
        else:
            improved = current > self.best_score + self.min_delta

        if improved:
            self.best_score = current
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(
                f"Early stopping triggered after {self.counter} epochs "
                f"without improvement in {self.monitor}."
            )
            return True

        return False
