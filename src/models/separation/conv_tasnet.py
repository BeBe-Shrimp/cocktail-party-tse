"""Conv-TasNet: Convolutional Time-domain Audio Separation Network.

Temporal Convolutional Network (TCN) based separation backbone, adapted for
target speaker extraction with FiLM-based speaker conditioning.

Reference: Luo & Mesgarani (2019) — *Conv-TasNet: Surpassing Ideal Time-Frequency
Magnitude Masking for Speech Separation*

Enhancement: Added FiLM layers at each TCN block for speaker-conditioned separation,
enabling the network to focus on the target speaker specified by the enrollment audio.
"""

from typing import Optional

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Causal 1D convolution ensuring no leakage from future time steps.

    Useful for real-time / streaming applications where low latency is required.
    Uses left-padding only.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Kernel size.
        dilation: Dilation factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with causal padding."""
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Single TCN block: dilated 1x1-conv → depthwise dilated conv → 1x1-conv.

    Uses residual connection with group normalization and PReLU activation.
    Speaker conditioning is applied via FiLM after the depthwise convolution.

    Args:
        in_channels: Input channels.
        out_channels: Output channels (bottleneck).
        kernel_size: Depthwise conv kernel size.
        dilation: Dilation rate (exponentially growing across blocks).
        speaker_dim: Speaker embedding dimension for FiLM conditioning.
        causal: Whether to use causal convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        speaker_dim: int = 256,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv_cls = CausalConv1d if causal else nn.Conv1d

        # Bottleneck: 1x1 conv
        self.bottleneck = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bottleneck_norm = nn.GroupNorm(1, out_channels)

        # Depthwise dilated conv
        padding = ((kernel_size - 1) * dilation) // 2
        if causal:
            self.depthwise: nn.Module = conv_cls(
                out_channels, out_channels, kernel_size, dilation=dilation
            )
        else:
            self.depthwise = nn.Conv1d(
                out_channels, out_channels, kernel_size,
                dilation=dilation, padding=padding,
            )
        self.depthwise_norm = nn.GroupNorm(1, out_channels)

        # Expansion: 1x1 conv back to in_channels
        self.expand = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.expand_norm = nn.GroupNorm(1, in_channels)

        # Speaker conditioning (FiLM)
        self.film_gamma = nn.Linear(speaker_dim, out_channels)
        self.film_beta = nn.Linear(speaker_dim, out_channels)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        self.prelu = nn.PReLU()
        self.output_prelu = nn.PReLU()

        # Residual projection (if channels differ)
        self.residual_conv = None
        if in_channels != in_channels:
            self.residual_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with speaker conditioning.

        Args:
            x: Input features ``(B, C, T)``.
            speaker_emb: Speaker embedding ``(B, D)``.

        Returns:
            Processed features ``(B, C, T)``.
        """
        residual = x

        # Bottleneck
        out = self.bottleneck(x)
        out = self.bottleneck_norm(out)
        out = self.prelu(out)

        # FiLM conditioning at bottleneck
        gamma = self.film_gamma(speaker_emb).unsqueeze(-1) + 1.0  # centered at 1
        beta = self.film_beta(speaker_emb).unsqueeze(-1)  # centered at 0
        out = gamma * out + beta

        # Depthwise dilated conv
        out = self.depthwise(out)
        out = self.depthwise_norm(out)
        out = self.prelu(out)

        # Expansion
        out = self.expand(out)
        out = self.expand_norm(out)

        # Residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        out = out + residual
        out = self.output_prelu(out)

        return out


class TCNStack(nn.Module):
    """Stack of TCN blocks with exponentially increasing dilation.

    Repeats a pattern of dilations [1, 2, 4, 8, ..., 2^(num_blocks-1)]
    for `num_repeats` times, enabling the network to model dependencies
    at multiple time scales.

    Args:
        num_blocks: Number of TCN blocks per repeat.
        in_channels: Input channels.
        bottleneck_channels: Bottleneck channels.
        kernel_size: Depthwise conv kernel size.
        num_repeats: Number of times to repeat the dilation pattern.
        speaker_dim: Speaker embedding dimension.
        causal: Use causal convolutions (for streaming).
    """

    def __init__(
        self,
        num_blocks: int = 8,
        in_channels: int = 512,
        bottleneck_channels: int = 256,
        kernel_size: int = 3,
        num_repeats: int = 3,
        speaker_dim: int = 256,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()

        for r in range(num_repeats):
            for b in range(num_blocks):
                dilation = 2 ** b
                self.blocks.append(
                    TCNBlock(
                        in_channels=in_channels,
                        out_channels=bottleneck_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        speaker_dim=speaker_dim,
                        causal=causal,
                    )
                )

    def forward(self, x: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """Process features through all TCN blocks.

        Args:
            x: Input ``(B, C, T)``.
            speaker_emb: Speaker embedding ``(B, D)``.

        Returns:
            Processed features ``(B, C, T)``.
        """
        for block in self.blocks:
            x = block(x, speaker_emb)
        return x


class ConvTasNet(nn.Module):
    """Conv-TasNet separation network with speaker conditioning.

    This is a time-domain speech separation network that operates on
    learned features (from a Conv1D encoder). The TCN backbone processes
    features with speaker-conditioned dilated convolutions to estimate
    a target speaker mask.

    Args:
        feature_dim: Feature dimension (encoder output channels).
        bottleneck_channels: TCN bottleneck channels.
        num_tcn_blocks: TCN blocks per repeat cycle.
        tcn_repeats: Number of dilation pattern repeats.
        kernel_size: Depthwise conv kernel size in TCN.
        speaker_dim: Speaker embedding dimension.
        causal: Use causal convolutions.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        bottleneck_channels: int = 256,
        num_tcn_blocks: int = 8,
        tcn_repeats: int = 3,
        kernel_size: int = 3,
        speaker_dim: int = 256,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        # Input normalization
        self.input_norm = nn.GroupNorm(1, feature_dim)

        # 1x1 conv before TCN
        self.input_conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)

        # TCN stack with speaker conditioning
        self.tcn = TCNStack(
            num_blocks=num_tcn_blocks,
            in_channels=feature_dim,
            bottleneck_channels=bottleneck_channels,
            kernel_size=kernel_size,
            num_repeats=tcn_repeats,
            speaker_dim=speaker_dim,
            causal=causal,
        )

        # Mask estimation: 1x1 conv -> ReLU -> 1x1 conv -> Sigmoid
        self.mask_conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """Estimate target speaker mask.

        Args:
            features: Encoded mixture features ``(B, F, T)``.
            speaker_emb: Speaker embedding ``(B, D)``.

        Returns:
            Soft mask ``(B, F, T)`` in [0, 1].
        """
        x = self.input_norm(features)
        x = self.input_conv(x)
        x = self.tcn(x, speaker_emb)
        mask = self.mask_conv(x)
        return mask
