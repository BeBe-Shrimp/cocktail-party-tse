"""Audio decoder — transposed convolution that reconstructs waveform from features.

The decoder is symmetric to the encoder and performs the inverse operation:
it maps from the learned feature space back to audio waveform using
a transposed 1D convolution (deconvolution).
"""

from typing import Optional

import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    """Transposed 1D convolutional decoder for waveform reconstruction.

    Performs the inverse of ``ConvEncoder``: maps from feature space
    ``(B, F, T_enc)`` back to waveform ``(B, 1, T)`` using a transposed
    convolution (also known as deconvolution or fractionally-strided convolution).

    Args:
        kernel_size: Convolution kernel size (must match encoder kernel_size).
        stride: Stride (must match encoder stride).
        in_channels: Input feature dimension (encoder out_channels).
        out_channels: Number of output audio channels (1 for mono).
        bias: Whether to include bias.
    """

    def __init__(
        self,
        kernel_size: int = 16,
        stride: int = 8,
        in_channels: int = 512,
        out_channels: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to waveform.

        Args:
            features: Feature tensor ``(B, in_channels, T_enc)``.

        Returns:
            Reconstructed waveform ``(B, out_channels, T)``.
        """
        return self.deconv(features)

    def get_output_length(self, input_length: int) -> int:
        """Compute output waveform length for a given feature length.

        Args:
            input_length: Number of input feature time steps.

        Returns:
            Number of output audio samples.
        """
        return (input_length - 1) * self.stride + self.kernel_size
