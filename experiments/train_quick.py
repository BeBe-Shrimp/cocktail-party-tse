#!/usr/bin/env python
"""Direct training script — no Hydra required (Python 3.14 compatibility).

Usage:
    python experiments/train_quick.py
    python experiments/train_quick.py --epochs 20 --batch-size 4
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.auditory_tse import AuditoryTSE
from src.data.custom_tse import CustomTSEDataset
from src.data.collate import Collator
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Quick training for Auditory-TSE")
    parser.add_argument("--data-dir", type=str, default="./data/synthetic_tse")
    parser.add_argument("--output", type=str, default="./outputs/quick_train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # --- Model (tiny for fast CPU training) ---
    model = AuditoryTSE(
        encoder_channels=64,
        speaker_embedding_dim=64,
        separation_type="conv_tasnet",
        bottleneck_channels=32,
        num_tcn_blocks=3,
        tcn_repeats=1,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created: {n_params:,} parameters")

    # --- Data (1.5s segments for faster training) ---
    train_dataset = CustomTSEDataset(data_dir=data_dir, split="train", segment=1.5)
    val_dataset = CustomTSEDataset(data_dir=data_dir, split="val", segment=1.5, return_full=True)
    collator = Collator()

    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, collate_fn=collator, num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        drop_last=False, collate_fn=collator, num_workers=2,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "warmup_epochs": 3,
            "early_stopping_patience": 8,
            "use_amp": False,
            "scheduler_type": "cosine_warmup",
            "monitor": "val_loss",
            "monitor_mode": "min",
        },
        output_dir=output_dir,
        device=torch.device(args.device),
    )

    # --- Train ---
    logger.info("Starting training...")
    results = trainer.fit()
    logger.info(f"Training done! Best val_loss: {results['best_val_loss']:.4f}")

    # Show checkpoint location
    ckpt_dir = output_dir / "checkpoints"
    for ckpt in sorted(ckpt_dir.glob("*.ckpt")):
        size_mb = ckpt.stat().st_size / 1024 / 1024
        logger.info(f"  Checkpoint: {ckpt.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
