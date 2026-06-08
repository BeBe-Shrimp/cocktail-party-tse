#!/usr/bin/env python
"""Training entry point for Auditory-TSE.

Launches training with Hydra configuration management.
Supports command-line overrides for hyperparameter sweeps.

Usage:
    # Default training
    python experiments/train.py

    # With model override
    python experiments/train.py model=sepformer

    # With hyperparameter overrides
    python experiments/train.py training.learning_rate=5e-4 training.epochs=200

    # Multi-run (hyperparameter sweep)
    python experiments/train.py -m training.learning_rate=1e-3,5e-4,1e-4
"""

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.auditory_tse import AuditoryTSE
from src.data.dataset import TSEDataset
from src.data.librimix import LibriMixDataset
from src.data.wham import WHAMDataset
from src.data.collate import Collator
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model(cfg: DictConfig) -> AuditoryTSE:
    """Create the AuditoryTSE model from config.

    Args:
        cfg: Hydra configuration.

    Returns:
        Instantiated model.
    """
    model_cfg = cfg.model

    # Collect separation kwargs
    separation_kwargs: Dict[str, Any] = {}
    if model_cfg.separation_type == "conv_tasnet":
        separation_kwargs.update({
            "bottleneck_channels": model_cfg.bottleneck_channels,
            "num_tcn_blocks": model_cfg.num_tcn_blocks,
            "tcn_repeats": model_cfg.tcn_repeats,
            "causal": model_cfg.causal,
        })
    elif model_cfg.separation_type == "sepformer":
        separation_kwargs.update({
            "embed_dim": model_cfg.sepformer_embed_dim,
            "num_heads": model_cfg.sepformer_num_heads,
            "ffn_dim": model_cfg.sepformer_ffn_dim,
            "num_layers": model_cfg.sepformer_num_layers,
            "chunk_size": model_cfg.sepformer_chunk_size,
            "dropout": model_cfg.sepformer_dropout,
        })
    elif model_cfg.separation_type == "cross_attn":
        separation_kwargs.update({
            "num_heads": model_cfg.cross_attn_num_heads,
            "num_layers": model_cfg.cross_attn_num_layers,
            "dropout": model_cfg.cross_attn_dropout,
        })

    model = AuditoryTSE(
        encoder_kernel_size=model_cfg.encoder_kernel_size,
        encoder_stride=model_cfg.encoder_stride,
        encoder_channels=model_cfg.encoder_channels,
        speaker_embedding_dim=model_cfg.speaker_embedding_dim,
        separation_type=model_cfg.separation_type,
        sample_rate=model_cfg.sample_rate,
        use_dual_path_encoder=model_cfg.get("use_dual_path_encoder", False),
        **separation_kwargs,
    )

    return model


def create_dataloaders(cfg: DictConfig) -> tuple:
    """Create training and validation DataLoaders.

    Args:
        cfg: Hydra configuration.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    data_cfg = cfg.data
    data_dir = Path(data_cfg.data_dir)

    dataset_class = {
        "librimix": LibriMixDataset,
        "wham": WHAMDataset,
    }.get(data_cfg.name)

    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {data_cfg.name}")

    # Build dataset kwargs
    dataset_kwargs = {
        "data_dir": data_dir,
        "sample_rate": data_cfg.sample_rate,
        "segment": data_cfg.segment,
        "enrollment_duration": data_cfg.enrollment_duration,
    }

    # Add dataset-specific kwargs
    if data_cfg.name == "librimix":
        dataset_kwargs.update({
            "n_src": data_cfg.get("n_src", 2),
            "version": data_cfg.get("version", "min"),
            "task": data_cfg.get("task", "sep_clean"),
        })
    elif data_cfg.name == "wham":
        dataset_kwargs.update({
            "mode": data_cfg.get("mode", "min"),
        })

    # Create datasets
    train_dataset = dataset_class(split="train", **dataset_kwargs)
    val_dataset = dataset_class(split="dev", return_full=True, **dataset_kwargs)

    collator = Collator()

    dataloader_kwargs = {
        "batch_size": data_cfg.batch_size,
        "num_workers": data_cfg.get("num_workers", 4),
        "pin_memory": data_cfg.get("pin_memory", True),
        "collate_fn": collator,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=data_cfg.get("shuffle", True),
        drop_last=data_cfg.get("drop_last", True),
        **dataloader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **{k: v for k, v in dataloader_kwargs.items() if k != "shuffle"},
    )

    logger.info(
        f"DataLoaders created | train={len(train_dataset)} samples "
        f"({len(train_loader)} batches) | val={len(val_dataset)} samples "
        f"({len(val_loader)} batches)"
    )

    return train_loader, val_loader


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entry point.

    Args:
        cfg: Hydra configuration (auto-populated).
    """
    # Print config
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Set seed
    set_seed(cfg.seed)

    # Create model
    model = create_model(cfg)
    logger.info(f"Model created | params={sum(p.numel() for p in model.parameters()):,}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(cfg)

    # Build training config dict
    training_config = dict(cfg.training)
    training_config["epochs"] = cfg.training.epochs

    # Create trainer
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        output_dir=output_dir,
    )

    # Train
    results = trainer.fit()
    logger.info(f"Training complete | best_val_loss={results['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
