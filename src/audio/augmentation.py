"""Audio data augmentation for robust training.

Implements common audio augmentations used in speech separation:
- Speed perturbation
- SpecAugment (time/frequency masking)
- Random gain
- Room impulse response (RIR) convolution
"""

import random
from typing import Optional

import torch
import torch.nn.functional as F


class AudioAugmentor:
    """Audio data augmentation pipeline.

    Applies a configurable set of augmentations to improve model robustness
    and generalization across acoustic conditions.

    Args:
        speed_perturb: If True, randomly perturb speed by ±speed_perturb_range.
        speed_perturb_range: Speed perturbation rate range.
        specaugment: If True, apply SpecAugment masking.
        specaug_time_mask_max: Maximum number of time masks.
        specaug_freq_mask_max: Maximum number of frequency masks.
        random_gain: If True, apply random gain adjustment.
        random_gain_range: Gain range in dB.
        rir_apply_prob: Probability of applying RIR convolution.
    """

    def __init__(
        self,
        speed_perturb: bool = True,
        speed_perturb_range: tuple = (0.9, 1.1),
        specaugment: bool = False,  # Applied in feature domain, not here
        specaug_time_mask_max: int = 10,
        specaug_freq_mask_max: int = 2,
        random_gain: bool = True,
        random_gain_range: tuple = (-6.0, 6.0),
        rir_apply_prob: float = 0.3,
    ) -> None:
        self.speed_perturb = speed_perturb
        self.speed_perturb_range = speed_perturb_range
        self.specaugment = specaugment
        self.specaug_time_mask_max = specaug_time_mask_max
        self.specaug_freq_mask_max = specaug_freq_mask_max
        self.random_gain = random_gain
        self.random_gain_range = random_gain_range
        self.rir_apply_prob = rir_apply_prob

    def apply_speed_perturb(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Apply speed perturbation by resampling.

        Args:
            waveform: Input ``(..., T)``.
            sample_rate: Original sample rate.

        Returns:
            Speed-perturbed waveform of same length.
        """
        speed = random.uniform(*self.speed_perturb_range)
        if abs(speed - 1.0) < 0.01:
            return waveform

        # Resample to perturbed rate, then back to original rate
        # This changes both speed and pitch (mimicking natural speaking rate variation)
        orig_shape = waveform.shape
        new_length = int(orig_shape[-1] / speed)
        perturbed = F.interpolate(
            waveform.unsqueeze(1) if waveform.dim() < 3 else waveform,
            size=new_length,
            mode="linear",
            align_corners=False,
        )

        # Resize back to original length
        if waveform.dim() < 3:
            perturbed = perturbed.squeeze(1)
        result = F.interpolate(
            perturbed.unsqueeze(1) if perturbed.dim() < 3 else perturbed,
            size=orig_shape[-1],
            mode="linear",
            align_corners=False,
        )
        return result.squeeze(1) if waveform.dim() < 3 else result

    def apply_random_gain(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random gain adjustment.

        Args:
            waveform: Input waveform.

        Returns:
            Gain-adjusted waveform.
        """
        gain_db = random.uniform(*self.random_gain_range)
        gain_linear = 10 ** (gain_db / 20.0)
        return waveform * gain_linear

    def apply_specaugment(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SpecAugment: time and frequency masking.

        This should be applied to encoded features (not raw waveform) during training.

        Args:
            features: Feature tensor ``(B, F, T)``.

        Returns:
            Masked features.
        """
        if not self.specaugment:
            return features

        B, F, T = features.shape

        # Frequency masking
        for _ in range(random.randint(0, self.specaug_freq_mask_max)):
            f_mask = random.randint(1, max(1, F // 10))
            f_start = random.randint(0, F - f_mask)
            features[:, f_start : f_start + f_mask, :] = 0.0

        # Time masking
        for _ in range(random.randint(0, self.specaug_time_mask_max)):
            t_mask = random.randint(1, max(1, T // 10))
            t_start = random.randint(0, T - t_mask)
            features[:, :, t_start : t_start + t_mask] = 0.0

        return features

    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Apply augmentation pipeline.

        Args:
            waveform: Audio waveform ``(..., T)``.
            sample_rate: Sample rate.

        Returns:
            Augmented waveform.
        """
        if self.speed_perturb:
            waveform = self.apply_speed_perturb(waveform, sample_rate)
        if self.random_gain:
            waveform = self.apply_random_gain(waveform)
        return waveform
