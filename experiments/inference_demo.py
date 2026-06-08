#!/usr/bin/env python
"""Interactive demo script for Auditory-TSE.

Performs target speaker extraction on user-provided audio files.
The mixture should contain multiple speakers; the enrollment should
be a clean recording of the target speaker.

Usage:
    # Basic usage
    python experiments/inference_demo.py \\
        --input mixture.wav \\
        --enrollment target_speaker.wav \\
        --output separated.wav

    # With specific model checkpoint
    python experiments/inference_demo.py \\
        --input mixture.wav \\
        --enrollment target.wav \\
        --output separated.wav \\
        --checkpoint checkpoints/best.ckpt

    # Generate synthetic test data
    python experiments/inference_demo.py --generate-samples
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.pipeline import InferencePipeline

logger = logging.getLogger(__name__)


def generate_synthetic_samples(output_dir: Path) -> None:
    """Generate synthetic test audio files for demo purposes.

    Creates:
    - A 2-speaker mixture (sine waves at different frequencies)
    - A target speaker enrollment (sine wave at the target frequency)

    Args:
        output_dir: Directory to save generated audio.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 16000
    duration = 3.0
    t = np.arange(int(sample_rate * duration)) / sample_rate

    # Target speaker: 220 Hz (A3) + harmonics
    target = (
        0.5 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.15 * np.sin(2 * np.pi * 660 * t)
    )

    # Interfering speaker: 330 Hz (E4) + harmonics (different voice)
    interference = (
        0.5 * np.sin(2 * np.pi * 330 * t)
        + 0.3 * np.sin(2 * np.pi * 660 * t)
        + 0.15 * np.sin(2 * np.pi * 990 * t)
    )

    # Mixture: target + interference
    mixture = target + interference

    # Enrollment: clean target at different content (294 Hz = D4)
    enrollment_dur = 2.0
    t_enr = np.arange(int(sample_rate * enrollment_dur)) / sample_rate
    enrollment = (
        0.5 * np.sin(2 * np.pi * 294 * t_enr)
        + 0.3 * np.sin(2 * np.pi * 588 * t_enr)
        + 0.15 * np.sin(2 * np.pi * 882 * t_enr)
    )

    # Normalize
    mixture = mixture / np.abs(mixture).max() * 0.9
    target = target / np.abs(target).max() * 0.9
    enrollment = enrollment / np.abs(enrollment).max() * 0.9

    # Save
    try:
        import soundfile as sf

        sf.write(str(output_dir / "demo_mixture.wav"), mixture, sample_rate)
        sf.write(str(output_dir / "demo_enrollment.wav"), enrollment, sample_rate)
        sf.write(str(output_dir / "demo_target_groundtruth.wav"), target, sample_rate)

        logger.info(f"Synthetic samples generated in {output_dir}:")
        logger.info(f"  demo_mixture.wav      — 2-speaker mixture")
        logger.info(f"  demo_enrollment.wav   — Target speaker reference")
        logger.info(f"  demo_target_groundtruth.wav — Ground truth (for comparison)")
    except ImportError:
        # Fallback: save as raw numpy
        np.savez(
            str(output_dir / "demo_samples.npz"),
            mixture=mixture,
            enrollment=enrollment,
            target=target,
            sample_rate=sample_rate,
        )
        logger.info(f"Synthetic samples saved as demo_samples.npz in {output_dir}")


def main() -> None:
    """Main entry point for the inference demo."""
    parser = argparse.ArgumentParser(
        description="Auditory-TSE: Target Speaker Extraction Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Separate a target speaker from a mixture
  python experiments/inference_demo.py -i mix.wav -e target.wav -o out.wav

  # Use a specific model checkpoint
  python experiments/inference_demo.py -i mix.wav -e target.wav -o out.wav -c best.ckpt

  # Generate synthetic test samples
  python experiments/inference_demo.py --generate-samples
        """,
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Path to input mixture audio file (multi-speaker)",
    )
    parser.add_argument(
        "-e", "--enrollment",
        type=str,
        help="Path to target speaker enrollment audio",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="separated_output.wav",
        help="Path to save the separated audio (default: separated_output.wav)",
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (if omitted, uses a randomly initialized model for demo)",
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate synthetic test audio files and exit",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate (default: 16000)",
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=10.0,
        help="Chunk duration in seconds for long audio (default: 10.0)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Generate synthetic samples mode
    if args.generate_samples:
        output_dir = Path(args.output).parent if args.output != "separated_output.wav" else Path("demo_samples")
        generate_synthetic_samples(output_dir)
        return

    # Validate input
    if not args.input or not args.enrollment:
        parser.error("--input and --enrollment are required for inference. "
                      "Use --generate-samples to create test data.")

    input_path = Path(args.input)
    enrollment_path = Path(args.enrollment)

    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")
    if not enrollment_path.exists():
        parser.error(f"Enrollment file not found: {enrollment_path}")

    # Load model
    if args.checkpoint:
        logger.info(f"Loading model from {args.checkpoint}")
        pipeline = InferencePipeline.from_checkpoint(
            args.checkpoint,
            sample_rate=args.sample_rate,
        )
    else:
        logger.warning(
            "No checkpoint provided. Using a randomly initialized model. "
            "Results will be meaningless without a trained model."
        )
        # Create a default model for demonstration
        from src.models.auditory_tse import AuditoryTSE

        model = AuditoryTSE(sample_rate=args.sample_rate)
        pipeline = InferencePipeline(
            model=model,
            sample_rate=args.sample_rate,
            chunk_duration_seconds=args.chunk_size,
        )

    # Run inference
    logger.info(f"Separating target speaker from {input_path}...")
    logger.info(f"Enrollment: {enrollment_path}")

    separated = pipeline.run(
        mixture_path=input_path,
        enrollment_path=enrollment_path,
        output_path=args.output,
    )

    duration = len(separated) / args.sample_rate
    logger.info(f"Done! Separated audio: {args.output} ({duration:.1f}s)")


if __name__ == "__main__":
    main()
