"""Trainer — orchestrates the training loop with mixed precision and logging.

Handles:
- Training loop with gradient accumulation and AMP
- Validation loop with metric computation
- TensorBoard logging
- Checkpoint management
- Learning rate scheduling
- Progress display via tqdm
"""

import contextlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.training.losses import si_snr_loss
from src.training.optimizer import create_optimizer, create_scheduler
from src.training.callbacks import CheckpointCallback, EarlyStopping

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator for Auditory-TSE.

    Manages the complete training lifecycle: forward pass, loss computation,
    backpropagation with mixed precision, validation, checkpointing, and logging.

    Args:
        model: The AuditoryTSE model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration dictionary with keys:
            - ``learning_rate``: Initial learning rate (default 1e-3).
            - ``weight_decay``: Weight decay (default 1e-5).
            - ``epochs``: Total training epochs (default 100).
            - ``gradient_clip_val``: Max gradient norm (default 5.0).
            - ``accumulate_grad_batches``: Gradient accumulation steps (default 1).
            - ``use_amp``: Use automatic mixed precision (default True).
            - ``log_every_n_steps``: Logging interval in steps (default 100).
            - ``val_check_interval``: Validate every N epochs (default 1).
            - ``optimizer_type``: Optimizer type (default "adamw").
            - ``scheduler_type``: Scheduler type (default "cosine_warmup").
            - ``warmup_epochs``: Warmup epochs (default 10).
        output_dir: Directory for checkpoints and logs.
        device: Device to train on (auto-detected if None).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # Output directory
        self.output_dir = Path(output_dir or "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # Training configuration
        self.learning_rate = self.config.get("learning_rate", 1e-3)
        self.weight_decay = self.config.get("weight_decay", 1e-5)
        self.epochs = self.config.get("epochs", 100)
        self.gradient_clip_val = self.config.get("gradient_clip_val", 5.0)
        self.accumulate_grad_batches = self.config.get("accumulate_grad_batches", 1)
        self.use_amp = self.config.get("use_amp", True) and self.device.type == "cuda"
        self.log_every_n_steps = self.config.get("log_every_n_steps", 100)
        self.val_check_interval = self.config.get("val_check_interval", 1)

        # Optimizer
        self.optimizer = create_optimizer(
            self.model.parameters(),
            optimizer_type=self.config.get("optimizer_type", "adamw"),
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        steps_per_epoch = len(self.train_loader) // self.accumulate_grad_batches
        self.scheduler: Optional[Any] = None
        if self.config.get("scheduler_type") is not None:
            self.scheduler = create_scheduler(
                self.optimizer,
                scheduler_type=self.config.get("scheduler_type", "cosine_warmup"),
                num_epochs=self.epochs,
                warmup_epochs=self.config.get("warmup_epochs", 10),
                steps_per_epoch=steps_per_epoch,
            )

        # Callbacks
        self.checkpoint_callback = CheckpointCallback(
            checkpoint_dir=self.output_dir / "checkpoints",
            save_top_k=self.config.get("save_top_k", 3),
            monitor=self.config.get("monitor", "val_loss"),
            mode=self.config.get("monitor_mode", "min"),
            save_period=self.config.get("save_every_n_epochs", 0),
        )

        self.early_stopping: Optional[EarlyStopping] = None
        if self.config.get("early_stopping_patience", 0) > 0:
            self.early_stopping = EarlyStopping(
                monitor=self.config.get("monitor", "val_loss"),
                patience=self.config.get("early_stopping_patience", 20),
                mode=self.config.get("monitor_mode", "min"),
            )

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        logger.info(f"Trainer initialized | device={self.device} | epochs={self.epochs}")
        logger.info(f"Trainer | optimizer={type(self.optimizer).__name__} | lr={self.learning_rate}")
        logger.info(f"Trainer | AMP={self.use_amp} | grad_accum={self.accumulate_grad_batches}")

    def fit(self) -> Dict[str, float]:
        """Run the complete training loop.

        Returns:
            Dictionary of final training metrics.
        """
        logger.info(f"Starting training for {self.epochs} epochs...")
        start_time = time.time()

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Training
            train_metrics = self._train_epoch(epoch)

            # Validation
            val_metrics: Dict[str, float] = {}
            if (
                self.val_loader is not None
                and (epoch + 1) % self.val_check_interval == 0
            ):
                val_metrics = self._validate_epoch(epoch)

            # Scheduler step (epoch-based)
            if self.scheduler is not None and hasattr(self.scheduler, "step"):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    monitor_val = val_metrics.get("val_loss", train_metrics.get("loss", 0))
                    self.scheduler.step(monitor_val)
                else:
                    self.scheduler.step()

            # Log epoch-level metrics
            all_metrics = {**train_metrics, **val_metrics}
            self._log_epoch(epoch, all_metrics)

            # Checkpointing
            self.checkpoint_callback.on_epoch_end(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics=all_metrics,
            )

            # Early stopping
            if self.early_stopping is not None and val_metrics:
                if self.early_stopping(val_metrics):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            # Print progress
            loss_str = f"loss={train_metrics.get('loss', 0):.4f}"
            val_str = f"val_loss={val_metrics.get('val_loss', 0):.4f}" if val_metrics else ""
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch {epoch+1}/{self.epochs} | {loss_str} {val_str} | lr={lr:.2e}")

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 60:.1f} minutes")

        self.writer.close()
        return {
            "best_val_loss": self.checkpoint_callback.best_score,
            "epochs_completed": self.current_epoch + 1,
        }

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run a single training epoch.

        Args:
            epoch: Current epoch index.

        Returns:
            Dictionary with average training metrics.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        from tqdm import tqdm

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            loss = self._training_step(batch, batch_idx)

            # Gradient accumulation
            loss = loss / self.accumulate_grad_batches
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_val
                )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

                # Step-based scheduler
                if self.scheduler is not None and hasattr(self.scheduler, "step"):
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        self.scheduler.step()

            # Track metrics
            total_loss += loss.item() * self.accumulate_grad_batches
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item() * self.accumulate_grad_batches:.4f}"})

            # Step-level logging
            if self.global_step % self.log_every_n_steps == 0:
                self.writer.add_scalar("train/step_loss", loss.item(), self.global_step)
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/lr", lr, self.global_step)

        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss}

    def _training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Single training step: forward + loss.

        Args:
            batch: Data batch from the DataLoader.
            batch_idx: Batch index.

        Returns:
            Scalar loss value.
        """
        mixture = batch["mixture"].to(self.device, non_blocking=True)
        enrollment = batch["enrollment"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        audio_mask = batch.get("audio_mask")

        # Mixed precision forward
        with torch.amp.autocast("cuda") if self.use_amp else contextlib.nullcontext():
            output = self.model(mixture, enrollment)
            estimated = output["waveform"].squeeze(1)  # (B, T)

            # Ensure same length
            min_len = min(estimated.shape[-1], target.shape[-1])
            estimated = estimated[..., :min_len]
            target = target[..., :min_len]

            if audio_mask is not None:
                audio_mask = audio_mask.to(self.device, non_blocking=True)
                audio_mask = audio_mask[..., :min_len]

            loss = si_snr_loss(estimated, target, audio_mask)

        return loss

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Run validation.

        Args:
            epoch: Current epoch.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        from tqdm import tqdm

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
        for batch in pbar:
            mixture = batch["mixture"].to(self.device, non_blocking=True)
            enrollment = batch["enrollment"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            audio_mask = batch.get("audio_mask")

            with torch.amp.autocast("cuda") if self.use_amp else contextlib.nullcontext():
                output = self.model(mixture, enrollment)
                estimated = output["waveform"].squeeze(1)

                min_len = min(estimated.shape[-1], target.shape[-1])
                estimated = estimated[..., :min_len]
                target = target[..., :min_len]
                if audio_mask is not None:
                    audio_mask = audio_mask.to(self.device, non_blocking=True)
                    audio_mask = audio_mask[..., :min_len]

                loss = si_snr_loss(estimated, target, audio_mask)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        return {"val_loss": avg_loss}

    def _log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log epoch-level metrics to TensorBoard.

        Args:
            epoch: Current epoch.
            metrics: Dictionary of metric name -> value.
        """
        for name, value in metrics.items():
            self.writer.add_scalar(f"epoch/{name}", value, epoch)

    def resume_from_checkpoint(self, checkpoint_path: Path) -> int:
        """Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            The epoch to resume from.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        resume_epoch = checkpoint.get("epoch", -1) + 1
        self.current_epoch = resume_epoch

        logger.info(f"Resumed from {checkpoint_path} at epoch {resume_epoch}")
        return resume_epoch
