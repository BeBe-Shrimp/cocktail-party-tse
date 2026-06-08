"""Audio signal transformations: STFT, iSTFT, resampling, normalization."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def STFT(
    waveform: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    center: bool = True,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """Compute Short-Time Fourier Transform.

    Args:
        waveform: Input ``(B, T)`` or ``(B, C, T)``.
        n_fft: FFT size.
        hop_length: Hop length between successive frames.
        win_length: Window length (defaults to n_fft).
        window: Window tensor (defaults to Hann window).
        center: If True, pad signal so frames are centered.
        pad_mode: Padding mode.

    Returns:
        Complex spectrogram ``(B, C, n_fft//2+1, num_frames)``.
    """
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length)

    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(1)

    B, C, T = waveform.shape
    waveform_2d = waveform.reshape(B * C, T)

    spec = torch.stft(
        waveform_2d,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window.to(waveform.device),
        center=center,
        pad_mode=pad_mode,
        return_complex=True,
    )

    _, n_freq, n_frames = spec.shape
    spec = spec.reshape(B, C, n_freq, n_frames)
    return spec


def iSTFT(
    spectrogram: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    center: bool = True,
    length: Optional[int] = None,
) -> torch.Tensor:
    """Compute inverse Short-Time Fourier Transform.

    Args:
        spectrogram: Complex spectrogram ``(B, C, n_fft//2+1, num_frames)``.
        n_fft: FFT size.
        hop_length: Hop length.
        win_length: Window length.
        window: Window tensor.
        center: If True, the STFT was computed with center=True.
        length: Target output length in samples.

    Returns:
        Reconstructed waveform ``(B, C, T)``.
    """
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length)

    B, C, _, _ = spectrogram.shape
    spec_2d = spectrogram.reshape(B * C, *spectrogram.shape[2:])

    waveform = torch.istft(
        spec_2d,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window.to(spectrogram.device),
        center=center,
        length=length,
        return_complex=False,
    )

    waveform = waveform.reshape(B, C, waveform.shape[-1])
    return waveform


def resample(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
) -> torch.Tensor:
    """Resample audio to a new sample rate.

    Args:
        waveform: Audio tensor ``(B, T)`` or ``(B, C, T)``.
        orig_freq: Original sample rate.
        new_freq: Target sample rate.

    Returns:
        Resampled audio.
    """
    if orig_freq == new_freq:
        return waveform

    squeeze = False
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
        squeeze = True
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(1)

    resampled = torchaudio_resample(waveform, orig_freq, new_freq)

    if squeeze:
        resampled = resampled.squeeze(0).squeeze(0)
    return resampled


def torchaudio_resample(
    waveform: torch.Tensor, orig_freq: int, new_freq: int
) -> torch.Tensor:
    """Use torchaudio's resampler (high-quality Kaiser windowed sinc).

    Wrapped separately to handle import gracefully.
    """
    try:
        import torchaudio

        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=new_freq
        ).to(waveform.device)

        B, C, T = waveform.shape
        waveform_2d = waveform.reshape(B * C, T)
        resampled_2d = resampler(waveform_2d)
        _, T_new = resampled_2d.shape
        return resampled_2d.reshape(B, C, T_new)

    except ImportError:
        # Fallback: simple linear interpolation via PyTorch
        ratio = new_freq / orig_freq
        new_length = int(waveform.shape[-1] * ratio)
        return F.interpolate(waveform, size=new_length, mode="linear", align_corners=False)


def normalize_audio(
    waveform: torch.Tensor, target_dbfs: float = -25.0, eps: float = 1e-8
) -> torch.Tensor:
    """Normalize audio to a target dBFS level.

    Args:
        waveform: Audio tensor of any shape (normalized along last dim).
        target_dbfs: Target level in dB full scale.
        eps: Small value to avoid log(0).

    Returns:
        Level-normalized audio.
    """
    rms = torch.sqrt(torch.mean(waveform ** 2, dim=-1, keepdim=True) + eps)
    target_rms = 10 ** (target_dbfs / 20.0)
    scalar = target_rms / (rms + eps)
    return waveform * scalar
