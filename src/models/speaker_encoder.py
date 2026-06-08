"""Speaker encoder — extracts speaker identity embedding from enrollment audio.

Implements ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation
in TDNN based speaker verification), a state-of-the-art architecture for extracting
robust speaker embeddings (d-vectors / x-vectors) used as the "top-down attention"
cue in the target speaker extraction system.

Reference: Desplanques et al. (2020)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    """Squeeze-and-Excitation module with channel attention.

    Recalibrates channel-wise feature responses by explicitly modeling
    inter-dependencies between channels.

    Args:
        channels: Number of input/output channels.
        bottleneck: Bottleneck reduction ratio for the SE block.
    """

    def __init__(self, channels: int, bottleneck: int = 128) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention.

        Args:
            x: Input ``(B, C, T)``.

        Returns:
            Channel-attended tensor ``(B, C, T)``.
        """
        scale = self.se(x)
        return x * scale


class SEBottleneck2D(nn.Module):
    """SE-Res2Net bottleneck block with 1D dilated convolutions.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        scale: Res2Net scale (number of groups in feature splitting).
        dilation: Dilation rate for the temporal convolution.
        kernel_size: Kernel size for temporal convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 8,
        dilation: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.out_channels = out_channels

        width = out_channels // scale

        # First 1x1 conv: reduce channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # Multi-scale processing: each branch has increasing dilation
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(scale):
            dilation_i = dilation ** (i + 1)
            pad = ((kernel_size - 1) * dilation_i) // 2
            self.convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation_i,
                    padding=pad,
                )
            )
            self.bns.append(nn.BatchNorm1d(width))

        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second 1x1 conv: expand channels
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEModule(out_channels)
        self.relu = nn.ReLU()

        # Residual connection projection if needed
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input ``(B, C_in, T)``.

        Returns:
            Output ``(B, C_out, T)``.
        """
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        # Res2Net multi-scale split
        spx = torch.split(out, self.out_channels // self.scale, dim=1)
        spx_sum = []
        for i in range(self.scale):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](sp)
            sp = self.relu(sp)
            spx_sum.append(sp)
            if i == 0:
                sp = sp
        out = torch.cat(spx_sum, dim=1)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = out + residual
        out = self.relu(out)
        return out


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling layer.

    Computes weighted mean and standard deviation of frame-level features,
    where the weights are learned via an attention mechanism.

    Args:
        channels: Number of input channels.
        attention_channels: Bottleneck channels in attention head.
    """

    def __init__(self, channels: int, attention_channels: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, attention_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(attention_channels, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attentive statistics.

        Args:
            x: Frame-level features ``(B, C, T)``.

        Returns:
            Utterance-level representation ``(B, 2*C)``.
        """
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        concat = torch.cat([x, mean.repeat(1, 1, x.shape[2]), std.repeat(1, 1, x.shape[2])], dim=1)

        attn_weights = self.attention(concat)  # (B, C, T)

        weighted_mean = torch.sum(attn_weights * x, dim=2)  # (B, C)
        weighted_var = torch.sum(attn_weights * (x - weighted_mean.unsqueeze(2)) ** 2, dim=2)
        weighted_std = torch.sqrt(weighted_var.clamp(min=1e-8))

        pooled = torch.cat([weighted_mean, weighted_std], dim=1)  # (B, 2*C)
        return pooled


class ECAPATDNN(nn.Module):
    """ECAPA-TDNN speaker encoder.

    Encodes a variable-length enrollment utterance into a fixed-dimensional
    speaker embedding (d-vector) that captures the speaker's vocal identity.
    This embedding serves as the "top-down attention" cue for the separation network.

    Args:
        channels: Base number of channels (scaled per block).
        embedding_size: Output speaker embedding dimension.
        scale: Res2Net scale factor.
        sample_rate: Audio sample rate (informational, not used in computation).
    """

    def __init__(
        self,
        channels: int = 512,
        embedding_size: int = 256,
        scale: int = 8,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.sample_rate = sample_rate

        # Initial feature extraction
        self.conv1 = nn.Conv1d(1, channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

        # SE-Res2Net blocks with increasing dilation
        self.layer1 = SEBottleneck2D(channels, channels, scale=scale, dilation=2)
        self.layer2 = SEBottleneck2D(channels, channels, scale=scale, dilation=3)
        self.layer3 = SEBottleneck2D(channels, channels, scale=scale, dilation=4)

        # MFA (Multi-layer Feature Aggregation): concat outputs of all layers
        self.layer4 = nn.Conv1d(channels * 3, channels * 3, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(channels * 3)

        # Attentive Statistics Pooling
        self.asp = AttentiveStatisticsPooling(channels * 3)
        # LayerNorm for utterance-level features (works with any spatial size)
        self.asp_ln = nn.LayerNorm(channels * 3 * 2)

        # Final embedding layer
        self.fc = nn.Conv1d(channels * 3 * 2, embedding_size, kernel_size=1)
        self.bn_fc = nn.LayerNorm(embedding_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from enrollment audio.

        Args:
            waveform: Enrollment waveform ``(B, T)`` or ``(B, 1, T)``.
                      Typical duration: 2-5 seconds at 16kHz.

        Returns:
            L2-normalized speaker embedding ``(B, embedding_size)``.
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # (B, T) -> (B, 1, T)

        # Frame-level feature extraction
        x = self.conv1(waveform)
        x = self.relu(x)
        x = self.bn1(x)

        # SE-Res2Net blocks with skip connections for MFA
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        # Multi-layer feature aggregation
        x = torch.cat([out1, out2, out3], dim=1)
        x = self.layer4(x)
        x = self.relu(x)
        x = self.bn4(x)

        # Attentive statistics pooling -> utterance-level
        x = self.asp(x)  # (B, C_asp*2)
        x = self.asp_ln(x)  # LayerNorm over feature dim
        x = x.unsqueeze(2)  # (B, C_asp*2, 1)

        # Final embedding
        x = self.fc(x)  # (B, embedding_size, 1)
        x = x.squeeze(2)  # (B, embedding_size)
        x = self.bn_fc(x)  # LayerNorm over feature dim

        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        return x
