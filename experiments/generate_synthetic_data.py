#!/usr/bin/env python
"""Generate synthetic TSE training data — no download required.

Creates realistic-enough training data using:
- Harmonic series with formant-like shaping ("vowel-like" sounds)
- Different fundamental frequency ranges per speaker (simulating male/female/child)
- Random pauses and amplitude modulation (simulating natural speech rhythm)
- Background noise

This is enough to train a TSE model end-to-end and verify the pipeline works.
After training, delete the data/ folder — keep only the checkpoint.
"""

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SAMPLE_RATE = 16000


def generate_speaker_voice(
    duration: float,
    sample_rate: int,
    f0_range: Tuple[float, float],
    formant_freqs: List[float],
    vibrato_rate: float = 5.0,
) -> np.ndarray:
    """Generate a synthetic "voice" using harmonic series with formant shaping.

    Args:
        duration: Duration in seconds.
        sample_rate: Sample rate.
        f0_range: (min, max) fundamental frequency range.
        formant_freqs: Center frequencies of formants (Hz).
        vibrato_rate: Vibrato frequency.

    Returns:
        Audio array (num_samples,).
    """
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # Smoothly varying F0 (pitch contour)
    f0 = np.linspace(
        random.uniform(*f0_range),
        random.uniform(*f0_range),
        n_samples,
    )
    # Add vibrato
    f0 += 5 * np.sin(2 * np.pi * vibrato_rate * t + random.random() * np.pi)
    f0 = np.clip(f0, f0_range[0], f0_range[1])

    # Phase accumulation
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate

    # Harmonic series with formant shaping
    voice = np.zeros(n_samples, dtype=np.float32)
    for h in range(1, 9):  # 8 harmonics
        amplitude = 1.0 / h  # Natural -6dB/octave rolloff
        # Formant shaping: boost near formant frequencies
        for ff in formant_freqs:
            # Gaussian bump around each formant
            distance = (h * f0.mean() - ff) / (ff * 0.15)
            amplitude *= 1 + 3 * np.exp(-0.5 * distance ** 2)

        voice += amplitude * np.sin(h * phase)

    # Amplitude modulation (simulating syllable rhythm)
    syllable_rate = random.uniform(2.5, 5.0)  # syllables per second
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t + random.random() * np.pi)
    envelope = np.clip(envelope, 0.1, 1.0)

    # Random pauses
    n_pauses = random.randint(0, int(duration))
    for _ in range(n_pauses):
        pause_start = random.randint(0, n_samples - sample_rate // 4)
        pause_len = random.randint(sample_rate // 20, sample_rate // 5)
        pause_end = min(pause_start + pause_len, n_samples)
        envelope[pause_start:pause_end] *= random.uniform(0.05, 0.2)

    voice = voice * envelope

    # Normalize to [-1, 1]
    peak = np.abs(voice).max()
    if peak > 0:
        voice = voice / peak * 0.9

    return voice.astype(np.float32)


# Speaker profiles: (gender_label, f0_range, formants)
SPEAKER_PROFILES = {
    0: ("male_low", (85, 120), [500, 1500, 2500]),
    1: ("male_mid", (100, 150), [550, 1600, 2600]),
    2: ("male_high", (120, 180), [600, 1700, 2700]),
    3: ("female_low", (160, 220), [700, 1800, 2800]),
    4: ("female_mid", (180, 260), [750, 1900, 2900]),
    5: ("female_high", (200, 300), [800, 2000, 3000]),
    6: ("child", (250, 400), [900, 2200, 3200]),
    7: ("male_deep", (75, 110), [450, 1400, 2400]),
    8: ("female_bright", (220, 320), [850, 2100, 3100]),
    9: ("teen_male", (130, 200), [600, 1650, 2650]),
}


def generate_mixture(
    target_speaker_id: int,
    interference_speaker_id: int,
    duration: float,
    enrollment_duration: float,
    sample_rate: int = SAMPLE_RATE,
) -> dict:
    """Generate a mixture + enrollment + target for TSE.

    Args:
        target_speaker_id: ID of the target speaker.
        interference_speaker_id: ID of the interference speaker.
        duration: Mixture duration in seconds.
        enrollment_duration: Enrollment duration.
        sample_rate: Sample rate.

    Returns:
        Dict with keys: mixture, target, enrollment, target_spk, intf_spk.
    """
    _, f0_tgt, fmt_tgt = SPEAKER_PROFILES[target_speaker_id]
    _, f0_intf, fmt_intf = SPEAKER_PROFILES[interference_speaker_id]

    # Generate target speaker voice
    target = generate_speaker_voice(duration, sample_rate, f0_tgt, fmt_tgt)

    # Generate interference speaker voice
    interference = generate_speaker_voice(duration, sample_rate, f0_intf, fmt_intf)

    # Mix: target at 0dB, interference at random relative level
    snr = random.uniform(-5, 10)  # dB
    intf_rms = np.sqrt(np.mean(interference ** 2) + 1e-8)
    tgt_rms = np.sqrt(np.mean(target ** 2) + 1e-8)
    scale = tgt_rms / intf_rms * 10 ** (-snr / 20)
    mixture = target + interference * scale

    # Add light background noise
    noise_level = random.uniform(0.001, 0.01)
    noise = np.random.randn(len(mixture)).astype(np.float32)
    mixture = mixture + noise_level * noise

    # Normalize mixture
    peak = np.abs(mixture).max()
    if peak > 0.99:
        mixture = mixture / peak * 0.95
        target = target / peak * 0.95

    # Generate enrollment: different content, same speaker
    enrollment = generate_speaker_voice(enrollment_duration, sample_rate, f0_tgt, fmt_tgt)

    return {
        "mixture": mixture.astype(np.float32),
        "target": target.astype(np.float32),
        "enrollment": enrollment.astype(np.float32),
        "target_speaker": target_speaker_id,
        "interference_speaker": interference_speaker_id,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic TSE training data")
    parser.add_argument("--output", type=str, default="./data/synthetic_tse", help="Output directory")
    parser.add_argument("--n-train", type=int, default=2000, help="Training examples")
    parser.add_argument("--n-val", type=int, default=200, help="Validation examples")
    parser.add_argument("--n-test", type=int, default=100, help="Test examples")
    parser.add_argument("--duration", type=float, default=4.0, help="Mixture duration (seconds)")
    parser.add_argument("--enrollment-duration", type=float, default=3.0, help="Enrollment duration")
    parser.add_argument("--num-speakers", type=int, default=10, help="Number of distinct speakers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    num_speakers = min(args.num_speakers, len(SPEAKER_PROFILES))
    speaker_ids = list(range(num_speakers))

    # Generate splits
    splits = {
        "train": (args.n_train, True),
        "val": (args.n_val, False),
        "test": (args.n_test, False),
    }

    total = 0
    for split_name, (n_examples, is_train) in splits.items():
        split_dir = output_dir / split_name
        for sub in ["mix", "target", "enrollment"]:
            (split_dir / sub).mkdir(parents=True, exist_ok=True)

        metadata = []
        for i in range(n_examples):
            # Pick two different speakers
            tgt_spk, intf_spk = random.sample(speaker_ids, 2)

            data = generate_mixture(
                tgt_spk, intf_spk,
                duration=args.duration,
                enrollment_duration=args.enrollment_duration,
            )

            idx = total + i

            # Save audio
            sf.write(str(split_dir / "mix" / f"mix_{idx:05d}.wav"), data["mixture"], SAMPLE_RATE)
            sf.write(str(split_dir / "target" / f"target_{idx:05d}.wav"), data["target"], SAMPLE_RATE)
            sf.write(str(split_dir / "enrollment" / f"enroll_{idx:05d}.wav"), data["enrollment"], SAMPLE_RATE)

            metadata.append({
                "mixture_path": str(split_dir / "mix" / f"mix_{idx:05d}.wav"),
                "target_path": str(split_dir / "target" / f"target_{idx:05d}.wav"),
                "enrollment_path": str(split_dir / "enrollment" / f"enroll_{idx:05d}.wav"),
                "target_speaker": str(data["target_speaker"]),
                "interference_speaker": str(data["interference_speaker"]),
                "duration": args.duration,
            })

        # Save metadata CSV
        csv_path = split_dir / "metadata.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
            writer.writeheader()
            writer.writerows(metadata)

        logger.info(f"Saved {n_examples} {split_name} examples to {split_dir}")
        total += n_examples

    logger.info(f"Total: {total} examples generated")
    logger.info(f"Data ready at: {output_dir}")
    logger.info(f"\nNow run: python experiments/train.py data=synthetic_tse")


if __name__ == "__main__":
    main()
