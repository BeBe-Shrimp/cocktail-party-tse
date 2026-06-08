"""Loss functions for speech separation.

Key loss: Scale-Invariant Signal-to-Noise Ratio (SI-SNR), the standard
evaluation and training loss for time-domain speech separation.

Also includes: SI-SDR (very similar to SI-SNR), and composite losses.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2 normalize along a dimension.

    Args:
        x: Input tensor.
        dim: Dimension to normalize along.
        eps: Epsilon for numerical stability.

    Returns:
        L2-normalized tensor.
    """
    return x / (torch.norm(x, dim=dim, keepdim=True) + eps)


def si_snr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR).

    SI-SNR measures the signal fidelity independent of scaling.
    A positive SI-SNR means the estimate is closer to the target than
    to silence, with higher values indicating better separation.

    Formula:
        s_target = <estimate, target> * target / ||target||^2
        e_noise = estimate - s_target
        SI_SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)

    Args:
        estimate: Estimated signal ``(..., T)``.
        target: Ground truth signal ``(..., T)``.
        eps: Small value for numerical stability.

    Returns:
        SI-SNR in decibels. Shape: ``(...)`` (one value per batch item)
    """
    # Ensure same shape
    assert estimate.shape == target.shape, f"{estimate.shape} != {target.shape}"

    # Zero-mean normalization along time dimension
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)

    # Project estimate onto target
    target_norm = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    projection = torch.sum(estimate * target, dim=-1, keepdim=True) / target_norm
    s_target = projection * target
    e_noise = estimate - s_target

    # Compute SI-SNR
    s_power = torch.sum(s_target ** 2, dim=-1) + eps
    e_power = torch.sum(e_noise ** 2, dim=-1) + eps

    si_snr_val = 10 * torch.log10(s_power / e_power)
    return si_snr_val


def si_snr_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """SI-SNR loss: negative mean SI-SNR (to be minimized).

    Args:
        estimate: Estimated waveform ``(B, T)``.
        target: Ground truth waveform ``(B, T)``.
        mask: Boolean mask ``(B, T)`` for valid (non-padded) samples.
            Zero-padded regions are excluded from loss computation.
        eps: Numerical stability epsilon.

    Returns:
        Scalar loss value (negative mean SI-SNR).
    """
    if mask is not None:
        # Compute SI-SNR only on valid regions
        # For padded batches, we compute per-sample SI-SNR weighted by valid length
        si_snr_values = []
        for i in range(estimate.shape[0]):
            valid_len = mask[i].sum().item()
            if valid_len > 0:
                # Compute SI-SNR on valid (non-padded) region
                e_valid = estimate[i, :valid_len]
                t_valid = target[i, :valid_len]
                val = si_snr(e_valid, t_valid, eps)
                si_snr_values.append(val)
        if not si_snr_values:
            return torch.tensor(0.0, device=estimate.device, requires_grad=True)
        si_snr_vals = torch.stack(si_snr_values)
        return -torch.mean(si_snr_vals)
    else:
        return -torch.mean(si_snr(estimate, target, eps))


class SiSNRLoss(nn.Module):
    """SI-SNR loss as a ``torch.nn.Module``.

    Wraps ``si_snr_loss`` for use in training loops and Hydra configs.

    Args:
        eps: Numerical stability epsilon.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute SI-SNR loss.

        Args:
            estimate: Estimated waveform ``(B, T)``.
            target: Target waveform ``(B, T)``.
            mask: Valid-sample mask ``(B, T)``.

        Returns:
            Scalar loss.
        """
        return si_snr_loss(estimate, target, mask, self.eps)


class CompositeLoss(nn.Module):
    """Composite loss combining SI-SNR with auxiliary losses.

    SI-SNR is the primary metric but can be supplemented with:
    - L1 loss on waveform: promotes sample-level accuracy
    - STFT magnitude loss: promotes spectral consistency
    - Speaker consistency loss: encourages the output to match target speaker

    Args:
        si_snr_weight: Weight for SI-SNR loss.
        l1_weight: Weight for L1 waveform loss (0 to disable).
        stft_weight: Weight for STFT magnitude loss (0 to disable).
    """

    def __init__(
        self,
        si_snr_weight: float = 1.0,
        l1_weight: float = 0.0,
        stft_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.si_snr_weight = si_snr_weight
        self.l1_weight = l1_weight
        self.stft_weight = stft_weight

        self.si_snr_fn = SiSNRLoss()

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute composite loss.

        Args:
            estimate: Estimated waveform ``(B, T)``.
            target: Target waveform ``(B, T)``.
            mask: Valid-sample mask ``(B, T)``.

        Returns:
            Tuple of ``(total_loss, loss_components_dict)``.
        """
        components: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=estimate.device)

        # SI-SNR loss (primary)
        loss_si_snr = self.si_snr_fn(estimate, target, mask)
        components["si_snr"] = loss_si_snr
        total = total + self.si_snr_weight * loss_si_snr

        # L1 waveform loss
        if self.l1_weight > 0:
            if mask is not None:
                loss_l1 = F.l1_loss(estimate * mask.float(), target * mask.float())
            else:
                loss_l1 = F.l1_loss(estimate, target)
            components["l1"] = loss_l1
            total = total + self.l1_weight * loss_l1

        # STFT magnitude loss
        if self.stft_weight > 0:
            loss_stft = self._stft_magnitude_loss(estimate, target)
            components["stft"] = loss_stft
            total = total + self.stft_weight * loss_stft

        return total, components

    def _stft_magnitude_loss(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        n_fft: int = 512,
        hop_length: int = 128,
    ) -> torch.Tensor:
        """STFT magnitude loss: L1 distance in the spectral domain.

        Args:
            estimate: Estimated waveform ``(B, T)``.
            target: Target waveform ``(B, T)``.
            n_fft: FFT size.
            hop_length: Hop length.

        Returns:
            Scalar STFT magnitude loss.
        """
        eps = 1e-8

        window = torch.hann_window(n_fft, device=estimate.device)

        # Compute STFT magnitudes
        est_spec = torch.stft(
            estimate.reshape(-1, estimate.shape[-1]),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        ).abs()

        tgt_spec = torch.stft(
            target.reshape(-1, target.shape[-1]),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        ).abs()

        # L1 loss on magnitudes (log-compressed for perceptual relevance)
        loss = F.l1_loss(
            torch.log(est_spec + eps),
            torch.log(tgt_spec + eps),
        )
        return loss
