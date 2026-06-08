"""Speech separation evaluation metrics.

Standard metrics for evaluating speech separation quality:
- SI-SDRi: Scale-Invariant SDR improvement (dB)
- PESQ: Perceptual Evaluation of Speech Quality
- STOI: Short-Time Objective Intelligibility
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.training.losses import si_snr

logger = logging.getLogger(__name__)


def si_sdri(
    estimate: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute Scale-Invariant SDR improvement (SI-SDRi).

    SI-SDRi measures how much the estimated signal improves over the
    original mixture in terms of SI-SDR. Higher is better.
    SI-SDRi = SI-SDR(estimate, target) - SI-SDR(mixture, target)

    Args:
        estimate: Estimated separated signal ``(T,)``.
        target: Ground truth target ``(T,)``.
        mixture: Original mixture ``(T,)``.
        eps: Numerical stability epsilon.

    Returns:
        SI-SDRi in dB.
    """
    # Ensure 1D
    estimate = estimate.reshape(-1)
    target = target.reshape(-1)
    mixture = mixture.reshape(-1)

    # Align lengths
    min_len = min(estimate.shape[-1], target.shape[-1], mixture.shape[-1])
    estimate = estimate[:min_len]
    target = target[:min_len]
    mixture = mixture[:min_len]

    si_sdr_est = si_snr(estimate.unsqueeze(0), target.unsqueeze(0), eps).item()
    si_sdr_mix = si_snr(mixture.unsqueeze(0), target.unsqueeze(0), eps).item()

    return si_sdr_est - si_sdr_mix


def pesq_score(
    estimate: np.ndarray,
    target: np.ndarray,
    sample_rate: int = 16000,
    mode: str = "wb",
) -> Optional[float]:
    """Compute PESQ (Perceptual Evaluation of Speech Quality).

    PESQ is an ITU-T standard (P.862) for objective speech quality assessment.
    Scores range from -0.5 to 4.5, with higher values indicating better quality.

    Args:
        estimate: Estimated signal (1D numpy array).
        target: Reference signal (1D numpy array).
        sample_rate: Sample rate (8000 for narrowband, 16000 for wideband).
        mode: ``"wb"`` for wideband or ``"nb"`` for narrowband.

    Returns:
        PESQ score, or None if the library is not installed.
    """
    try:
        from pesq import pesq

        # Ensure same length
        min_len = min(len(estimate), len(target))
        estimate = estimate[:min_len]
        target = target[:min_len]

        score = pesq(sample_rate, target, estimate, mode)
        return float(score)
    except ImportError:
        logger.warning("pesq library not installed. Install with: pip install pesq")
        return None
    except Exception as e:
        logger.debug(f"PESQ computation failed: {e}")
        return None


def stoi_score(
    estimate: np.ndarray,
    target: np.ndarray,
    sample_rate: int = 16000,
) -> Optional[float]:
    """Compute STOI (Short-Time Objective Intelligibility).

    STOI measures speech intelligibility. Scores range from 0 to 1,
    with higher values indicating better intelligibility.
    Typically, STOI > 0.75 indicates good intelligibility.

    Args:
        estimate: Estimated signal (1D numpy array).
        target: Reference signal (1D numpy array).
        sample_rate: Sample rate.

    Returns:
        STOI score, or None if pystoi is not installed.
    """
    try:
        from pystoi import stoi

        min_len = min(len(estimate), len(target))
        estimate = estimate[:min_len]
        target = target[:min_len]

        score = stoi(target, estimate, sample_rate, extended=False)
        return float(score)
    except ImportError:
        logger.warning("pystoi library not installed. Install with: pip install pystoi")
        return None
    except Exception as e:
        logger.debug(f"STOI computation failed: {e}")
        return None


def compute_metrics(
    estimated: torch.Tensor,
    target: torch.Tensor,
    mixture: Optional[torch.Tensor] = None,
    sample_rate: int = 16000,
    compute_all: bool = True,
) -> Dict[str, float]:
    """Compute all evaluation metrics for a separated utterance.

    Args:
        estimated: Estimated waveform ``(T,)``.
        target: Ground truth waveform ``(T,)``.
        mixture: Original mixture ``(T,)`` for SI-SDRi (optional).
        sample_rate: Audio sample rate.
        compute_all: If True, compute PESQ and STOI (slower).

    Returns:
        Dictionary of metric name -> value.
    """
    metrics: Dict[str, float] = {}

    # SI-SDR (no mixture needed)
    si_sdr_val = si_snr(estimated.unsqueeze(0), target.unsqueeze(0)).item()
    metrics["si_sdr"] = si_sdr_val

    # SI-SDRi (needs mixture)
    if mixture is not None:
        metrics["si_sdri"] = si_sdri(estimated, target, mixture)

    # PESQ and STOI (slower, numpy-based)
    if compute_all:
        est_np = estimated.cpu().numpy()
        tgt_np = target.cpu().numpy()

        pesq_val = pesq_score(est_np, tgt_np, sample_rate)
        if pesq_val is not None:
            metrics["pesq"] = pesq_val

        stoi_val = stoi_score(est_np, tgt_np, sample_rate)
        if stoi_val is not None:
            metrics["stoi"] = stoi_val

    return metrics
