#!/usr/bin/env python
"""Evaluation entry point — runs a trained model on the test set.

Usage:
    python experiments/evaluate.py checkpoint=/path/to/model.ckpt

    # With GPU override
    python experiments/evaluate.py checkpoint=./best.ckpt device=cuda:0

    # Save separated audio
    python experiments/evaluate.py checkpoint=./best.ckpt save_audio=true
"""

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.auditory_tse import AuditoryTSE
from src.data import Collator, LibriMixDataset, WHAMDataset
from src.evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device) -> AuditoryTSE:
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to .ckpt file.
        device: Target device.

    Returns:
        Loaded model in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Try to infer model config from checkpoint
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Determine encoder_channels from state dict
    encoder_weight = state_dict.get("encoder.conv.weight")
    if encoder_weight is not None:
        encoder_channels = encoder_weight.shape[0]
        encoder_kernel_size = encoder_weight.shape[2]
        # Infer stride from decoder
        decoder_weight = state_dict.get("decoder.deconv.weight")
        encoder_stride = decoder_weight.shape[2] if decoder_weight is not None else encoder_kernel_size // 2
    else:
        encoder_channels = 512
        encoder_kernel_size = 16
        encoder_stride = 8

    # Determine speaker_embedding_dim
    speaker_weight = state_dict.get("speaker_encoder.fc.weight")
    speaker_embedding_dim = speaker_weight.shape[0] if speaker_weight is not None else 256

    model = AuditoryTSE(
        encoder_kernel_size=encoder_kernel_size,
        encoder_stride=encoder_stride,
        encoder_channels=encoder_channels,
        speaker_embedding_dim=speaker_embedding_dim,
        sample_rate=16000,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded | params={sum(p.numel() for p in model.parameters()):,}")
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation entry point."""
    logger.info("Evaluation Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    checkpoint_path = cfg.get("checkpoint", "checkpoints/best.ckpt")
    model = load_model(checkpoint_path, device)

    # Create test dataset
    data_cfg = cfg.data
    data_dir = Path(data_cfg.data_dir)

    dataset_class = {"librimix": LibriMixDataset, "wham": WHAMDataset}.get(data_cfg.name)
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {data_cfg.name}")

    test_dataset = dataset_class(
        data_dir=data_dir,
        split="test",
        sample_rate=data_cfg.sample_rate,
        segment=None,  # Full-length for evaluation
        enrollment_duration=data_cfg.enrollment_duration,
        return_full=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Single sample for consistent metrics
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=Collator(),
    )

    logger.info(f"Test dataset | {len(test_dataset)} samples")

    # Create evaluator
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        save_audio=cfg.get("save_audio", False),
    )

    # Run evaluation
    results = evaluator.evaluate(
        compute_pesq_stoi=cfg.get("compute_pesq_stoi", True),
        verbose=True,
    )

    # Save results
    import json

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
