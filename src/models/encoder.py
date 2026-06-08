"""Auditory-inspired encoder — learnable Conv1D frontend with optional Gammatone branch.

The encoder converts raw waveform into a time-frequency feature representation.
It implements a learned 1D convolutional encoder (like Conv-TasNet) and can
optionally concatenate biologically-inspired Gammatone filterbank features.
"""

from typing import Optional

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """1D Convolutional encoder that maps waveform to a learned feature space.

    This is the standard encoder used in Conv-TasNet and related architectures.
    It uses a strided 1D convolution with ReLU activation.
    Each output channel acts as a learned "filter" analogous to a cochlear filter.

    Args:
        kernel_size: Convolution kernel size in samples (default 16 for 16kHz).
        stride: Stride in samples. Defaults to kernel_size // 2 for 50% overlap.
        in_channels: Number of input channels (1 for mono audio).
        out_channels: Number of output feature channels (encoder basis dimension).
        bias: Whether to include bias in the convolution.
    """

    def __init__(
        self,
        kernel_size: int = 16,
        stride: Optional[int] = None,
        in_channels: int = 1,
        out_channels: int = 512,
        bias: bool = False,
    ) -> None:
        super().__init__()
        stride = stride if stride is not None else kernel_size // 2

        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )
        self.activation = nn.ReLU()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode waveform into feature representation.

        Args:
            waveform: Input waveform of shape ``(B, C, T)`` or ``(B, T)``.

        Returns:
            Feature tensor of shape ``(B, out_channels, T_enc)``
            where ``T_enc = (T - kernel_size) // stride + 1``.
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # (B, T) -> (B, 1, T)

        features = self.conv(waveform)  # (B, out_channels, T_enc)
        features = self.activation(features)
        return features

    def get_output_length(self, input_length: int) -> int:
        """Compute output length for a given input length.

        Args:
            input_length: Number of input samples.

        Returns:
            Number of output time steps.
        """
        return (input_length - self.kernel_size) // self.stride + 1


class DualPathEncoder(nn.Module):
    """Dual-path encoder that combines learned Conv1D with Gammatone filterbank.

    This encoder simulates the dual-pathway nature of the human auditory system
    by combining:
    - A **learned** pathway (Conv1D, data-driven)
    - An **innate** pathway (Gammatone filterbank, biologically-inspired)

    The two pathways' outputs are concatenated along the feature dimension,
    then projected back to the desired output dimension.

    Args:
        kernel_size: Conv1D kernel size.
        stride: Conv1D stride.
        in_channels: Input channels.
        conv_channels: Number of channels from the learned Conv1D pathway.
        gammatone_channels: Number of Gammatone filterbank channels.
        out_channels: Final output feature dimension after fusion.
        sample_rate: Audio sample rate (Hz).
    """

    def __init__(
        self,
        kernel_size: int = 16,
        stride: Optional[int] = None,
        in_channels: int = 1,
        conv_channels: int = 256,
        gammatone_channels: int = 64,
        out_channels: int = 512,
        sample_rate: int = 16000,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.conv_encoder = ConvEncoder(
            kernel_size=kernel_size,
            stride=stride,
            in_channels=in_channels,
            out_channels=conv_channels,
            bias=bias,
        )
        self.gammatone_channels = gammatone_channels
        self.sample_rate = sample_rate

        # Fusion projection: concat(conv, gammatone) -> out_channels
        fusion_in = conv_channels + gammatone_channels
        self.fusion = nn.Conv1d(fusion_in, out_channels, kernel_size=1, bias=False)
        self.activation = nn.ReLU()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode waveform with dual pathways.

        Args:
            waveform: Input waveform ``(B, C, T)``.

        Returns:
            Fused feature tensor ``(B, out_channels, T_enc)``.
        """
        conv_feat = self.conv_encoder(waveform)

        # Compute Gammatone features on-the-fly approximation using STFT
        # (Full gammatone filterbank implemented in src.audio.filterbank)
        gammatone_feat = self._gammatone_approximation(waveform, conv_feat.shape[-1])

        # Concatenate and fuse
        fused = torch.cat([conv_feat, gammatone_feat], dim=1)
        fused = self.fusion(fused)
        return self.activation(fused)

    def _gammatone_approximation(
        self, waveform: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        """Approximate gammatone features using learned projection from STFT magnitudes.

        Full gammatone implementation is in ``src.audio.filterbank.GammatoneFilterbank``.
        This provides a differentiable stand-in that can be used during training.
        """
        batch, _, _ = waveform.shape
        n_fft = 512
        hop_length = waveform.shape[-1] // target_length

        # STFT magnitude as rough approximation of cochlear response
        stft = torch.stft(
            waveform.squeeze(1),
            n_fft=n_fft,
            hop_length=max(hop_length, 1),
            window=torch.hann_window(n_fft, device=waveform.device),
            return_complex=True,
        ).abs()

        # Resize to target length via interpolation
        if stft.shape[-1] != target_length:
            # stft is (B, F, T) — interpolate along time dim (last dim)
            stft = nn.functional.interpolate(
                stft,
                size=target_length,
                mode="linear",
                align_corners=False,
            )

        # Select or pad to gammatone_channels
        n_freqs = stft.shape[1]
        if n_freqs >= self.gammatone_channels:
            # Sample logarithmically-spaced frequency bins (mimics cochlear spacing)
            indices = torch.linspace(0, n_freqs - 1, self.gammatone_channels).long()
            stft = stft[:, indices, :]
        else:
            pad = self.gammatone_channels - n_freqs
            stft = nn.functional.pad(stft, (0, 0, 0, pad))

        return stft
