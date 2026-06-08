#!/usr/bin/env python
"""Download LibriSpeech and generate LibriMix-style training data for TSE.

This script:
1. Downloads LibriSpeech train-clean-100 via torchaudio
2. Creates 2-speaker mixtures with enrollment utterances for TSE training
3. Saves everything as .wav files with a metadata CSV

Run: python experiments/prepare_data.py --data-dir ./data --n-mixtures 1000
"""

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SAMPLE_RATE = 16000


def download_librispeech(data_dir: Path, subset: str = "train-clean-100") -> Path:
    """Download a LibriSpeech subset via torchaudio."""
    subset_dir = data_dir / "LibriSpeech" / subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_files = list(subset_dir.rglob("*.flac"))
    if len(existing_files) > 1000:
        logger.info(f"LibriSpeech {subset} already downloaded: {len(existing_files)} .flac files")
        return subset_dir

    logger.info(f"Downloading LibriSpeech {subset} (this may take a while, ~6GB)...")
    try:
        dataset = torchaudio.datasets.LIBRISPEECH(
            root=str(data_dir / "LibriSpeech"),
            url=subset,
            download=True,
        )
        # Trigger download by accessing first item
        _ = dataset[0]
        logger.info(f"Download complete. Found {len(dataset)} utterances.")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Manual download: https://www.openslr.org/12")
        logger.info(f"Extract to: {data_dir / 'LibriSpeech' / subset}")
        raise

    return subset_dir


def build_speaker_map(librispeech_dir: Path) -> Dict[str, List[Path]]:
    """Build {speaker_id: [utterance_paths]} from LibriSpeech directory."""
    speaker_map: Dict[str, List[Path]] = {}
    for audio_file in librispeech_dir.rglob("*.flac"):
        # LibriSpeech path: .../speaker_id/chapter_id/file.flac
        speaker_id = audio_file.parent.parent.name
        if speaker_id not in speaker_map:
            speaker_map[speaker_id] = []
        speaker_map[speaker_id].append(audio_file)

    logger.info(f"Found {len(speaker_map)} speakers, {sum(len(v) for v in speaker_map.values())} utterances")
    return speaker_map


def load_audio(path: Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load and resample audio to target sample rate."""
    waveform, sr = torchaudio.load(str(path))
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)  # (T,)


def create_mixture(
    target_path: Path,
    interference_path: Path,
    segment_duration: float = 4.0,
    enrollment_duration: float = 3.0,
    sample_rate: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Create a 2-speaker mixture with enrollment.

    Args:
        target_path: Path to target speaker utterance (used in mixture).
        interference_path: Path to interference speaker utterance.
        segment_duration: Duration of the mixture segment.
        enrollment_duration: Duration of enrollment audio.
        sample_rate: Sample rate.

    Returns:
        (mixture, target, enrollment, target_speaker_id, interference_speaker_id)
    """
    target_audio = load_audio(target_path, sample_rate)
    interference_audio = load_audio(interference_path, sample_rate)
    segment_samples = int(segment_duration * sample_rate)
    enroll_samples = int(enrollment_duration * sample_rate)

    # Trim to segment length
    if target_audio.shape[-1] < segment_samples:
        # Pad if too short
        target_audio = torch.nn.functional.pad(target_audio, (0, segment_samples - target_audio.shape[-1]))
    else:
        start = random.randint(0, target_audio.shape[-1] - segment_samples)
        target_audio = target_audio[start : start + segment_samples]

    if interference_audio.shape[-1] < segment_samples:
        interference_audio = torch.nn.functional.pad(
            interference_audio, (0, segment_samples - interference_audio.shape[-1])
        )
    else:
        start = random.randint(0, interference_audio.shape[-1] - segment_samples)
        interference_audio = interference_audio[start : start + segment_samples]

    # Normalize each speaker to similar loudness, then mix 1:1
    target_rms = torch.sqrt(torch.mean(target_audio ** 2) + 1e-8)
    interference_rms = torch.sqrt(torch.mean(interference_audio ** 2) + 1e-8)

    # Random relative gain: target at 0 dB, interference at -5 to +5 dB
    relative_gain = 10 ** (random.uniform(-5, 5) / 20.0)
    mixture = target_audio + interference_audio * (target_rms / (interference_rms + 1e-8)) * relative_gain

    # Normalize mixture to prevent clipping
    peak = mixture.abs().max()
    if peak > 0.99:
        mixture = mixture / peak * 0.95
        target_audio = target_audio / peak * 0.95

    # Create enrollment: a different random segment from target utterance
    full_target = load_audio(target_path, sample_rate)
    if full_target.shape[-1] < enroll_samples:
        full_target = torch.nn.functional.pad(full_target, (0, enroll_samples - full_target.shape[-1]))
    enroll_start = random.randint(0, full_target.shape[-1] - enroll_samples)
    enrollment = full_target[enroll_start : enroll_start + enroll_samples]

    # Speaker IDs
    target_spk = target_path.parent.parent.name
    intf_spk = interference_path.parent.parent.name

    return (
        mixture.numpy().astype(np.float32),
        target_audio.numpy().astype(np.float32),
        enrollment.numpy().astype(np.float32),
        target_spk,
        intf_spk,
    )


def generate_tse_data(
    librispeech_dir: Path,
    output_dir: Path,
    n_mixtures: int = 1000,
    segment_duration: float = 4.0,
    enrollment_duration: float = 3.0,
    num_speakers: int = 20,
    val_split: float = 0.1,
    test_split: float = 0.1,
) -> None:
    """Generate TSE training data from LibriSpeech.

    Args:
        librispeech_dir: Path to LibriSpeech subset.
        output_dir: Output directory for generated data.
        n_mixtures: Number of mixtures to generate.
        segment_duration: Duration of each mixture in seconds.
        enrollment_duration: Duration of enrollment audio.
        num_speakers: Number of distinct speakers to use.
        val_split: Fraction for validation.
        test_split: Fraction for testing.
    """
    speaker_map = build_speaker_map(librispeech_dir)

    # Select speakers with enough utterances
    qualified = {
        spk: paths
        for spk, paths in speaker_map.items()
        if len(paths) >= 5  # Need multiple utterances for enrollment + mixture
    }
    logger.info(f"Speakers with >= 5 utterances: {len(qualified)}")

    selected_speakers = random.sample(list(qualified.keys()), min(num_speakers, len(qualified)))
    logger.info(f"Selected {len(selected_speakers)} speakers")

    # Generate mixtures
    splits = {"train": [], "val": [], "test": []}

    for i in range(n_mixtures):
        # Pick two different speakers
        target_spk, intf_spk = random.sample(selected_speakers, 2)

        # Pick random utterances from each
        target_path = random.choice(qualified[target_spk])
        intf_path = random.choice(qualified[intf_spk])

        mixture, target, enrollment, t_id, i_id = create_mixture(
            target_path,
            intf_path,
            segment_duration=segment_duration,
            enrollment_duration=enrollment_duration,
        )

        # Determine split
        r = random.random()
        if r < test_split:
            split = "test"
        elif r < test_split + val_split:
            split = "val"
        else:
            split = "train"

        splits[split].append({
            "mixture": mixture,
            "target": target,
            "enrollment": enrollment,
            "target_speaker": t_id,
            "interference_speaker": i_id,
            "index": i,
        })

        if (i + 1) % 200 == 0:
            logger.info(f"Generated {i + 1}/{n_mixtures} mixtures...")

    # Save to disk
    import soundfile as sf

    for split_name, examples in splits.items():
        if not examples:
            continue

        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        mix_dir = split_dir / "mix"
        tgt_dir = split_dir / "target"
        enr_dir = split_dir / "enrollment"
        for d in [mix_dir, tgt_dir, enr_dir]:
            d.mkdir(exist_ok=True)

        metadata_rows = []
        for ex in examples:
            idx = ex["index"]

            # Save audio
            mix_path = mix_dir / f"mix_{idx:05d}.wav"
            tgt_path = tgt_dir / f"target_{idx:05d}.wav"
            enr_path = enr_dir / f"enroll_{idx:05d}.wav"

            sf.write(str(mix_path), ex["mixture"], SAMPLE_RATE)
            sf.write(str(tgt_path), ex["target"], SAMPLE_RATE)
            sf.write(str(enr_path), ex["enrollment"], SAMPLE_RATE)

            metadata_rows.append({
                "mixture_path": str(mix_path),
                "target_path": str(tgt_path),
                "enrollment_path": str(enr_path),
                "target_speaker": ex["target_speaker"],
                "interference_speaker": ex["interference_speaker"],
                "duration": segment_duration,
            })

        # Save metadata CSV
        csv_path = split_dir / "metadata.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
            writer.writeheader()
            writer.writerows(metadata_rows)

        logger.info(f"Saved {len(examples)} {split_name} examples to {split_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare LibriMix-style TSE data")
    parser.add_argument("--data-dir", type=str, default="./data", help="Root data directory")
    parser.add_argument("--n-mixtures", type=int, default=1000, help="Number of mixtures to generate")
    parser.add_argument("--num-speakers", type=int, default=20, help="Number of distinct speakers")
    parser.add_argument("--segment-duration", type=float, default=4.0, help="Mixture duration (seconds)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download LibriSpeech
    librispeech_dir = download_librispeech(data_dir, subset="train-clean-100")

    # Step 2: Generate TSE mixtures
    output_dir = data_dir / "tse_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_tse_data(
        librispeech_dir=librispeech_dir,
        output_dir=output_dir,
        n_mixtures=args.n_mixtures,
        segment_duration=args.segment_duration,
        num_speakers=args.num_speakers,
    )

    logger.info(f"Data preparation complete! Data saved to {output_dir}")
    logger.info("Next: python experiments/train.py data=tse_data")


if __name__ == "__main__":
    main()
