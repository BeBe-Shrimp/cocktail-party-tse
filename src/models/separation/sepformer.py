"""SepFormer: Separator Transformer for speech separation.

Dual-path transformer architecture that processes audio features along both
the temporal (time) and spectral (frequency) dimensions with self-attention
and cross-attention to speaker embeddings.

Reference: Subakan et al. (2021) — *Attention is All You Need in Speech Separation*

Key innovation over Conv-TasNet: replaces dilated convolutions with multi-head
self-attention, achieving stronger long-range dependency modeling.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module.

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of parallel attention heads.
        dropout: Attention dropout probability.
    """

    def __init__(
        self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention forward.

        Args:
            x: Input ``(B, T, C)``.

        Returns:
            Attended output ``(B, T, C)``.
        """
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, nH, T, T)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, nH, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        return out


class TransformerBlock(nn.Module):
    """Transformer block: Self-Attention → FFN, with pre-norm and residuals.

    Args:
        embed_dim: Feature dimension.
        num_heads: Attention heads.
        ffn_dim: Feed-forward network hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input ``(B, T, C)``.

        Returns:
            Output ``(B, T, C)``.
        """
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = self.dropout1(x) + residual

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x) + residual

        return x


class DualPathProcessing(nn.Module):
    """Dual-path processing: alternates intra-chunk and inter-chunk attention.

    Splits the sequence into overlapping chunks, applies:
    1. **Intra-chunk** attention: models dependencies within each chunk (local)
    2. **Inter-chunk** attention: models dependencies across chunks (global)

    This is the core innovation of SepFormer — it achieves linear complexity
    in sequence length (compared to quadratic for vanilla transformers) while
    maintaining long-range modeling capability.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        ffn_dim: FFN hidden dimension.
        num_layers: Number of transformer layers per path.
        chunk_size: Size of each chunk in frames.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        num_layers: int = 2,
        chunk_size: int = 250,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size

        # Intra-chunk transformers (process within each chunk)
        self.intra_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Inter-chunk transformers (process across chunks)
        self.inter_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.intra_norm = nn.LayerNorm(embed_dim)
        self.inter_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dual-path processing.

        Args:
            x: Input ``(B, T, C)``.

        Returns:
            Processed ``(B, T, C)``.
        """
        B, T, C = x.shape

        # Pad to multiple of chunk_size
        pad_len = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = nn.functional.pad(x, (0, 0, 0, pad_len))
        B, T_padded, C = x.shape
        num_chunks = T_padded // self.chunk_size

        # Reshape to chunks: (B, num_chunks, chunk_size, C)
        # First, process each chunk independently (intra-chunk)
        # Then, process across chunks (inter-chunk)
        # Repeat for `num_layers` times

        for intra_block, inter_block in zip(self.intra_blocks, self.inter_blocks):
            # --- Intra-chunk ---
            # (B*num_chunks, chunk_size, C)
            x = x.reshape(B * num_chunks, self.chunk_size, C)
            x = intra_block(x)
            x = self.intra_norm(x)

            # --- Inter-chunk ---
            # (B*chunk_size, num_chunks, C)
            x = x.reshape(B, num_chunks, self.chunk_size, C)
            x = x.transpose(1, 2).reshape(B * self.chunk_size, num_chunks, C)
            x = inter_block(x)
            x = self.inter_norm(x)

            # Back to original shape: (B, T_padded, C)
            x = x.reshape(B, self.chunk_size, num_chunks, C)
            x = x.transpose(1, 2).reshape(B, T_padded, C)

        # Remove padding
        if pad_len > 0:
            x = x[:, :T, :]

        return x


class SpeakerConditionedDualPath(nn.Module):
    """Dual-path processing with speaker conditioning via FiLM.

    Same as DualPathProcessing but injects speaker information at each
    transformer block through FiLM layers, enabling the model to focus
    processing on target-speaker-relevant features.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        ffn_dim: FFN hidden dimension.
        speaker_dim: Speaker embedding dimension.
        num_layers: Number of dual-path layers.
        chunk_size: Chunk size in frames.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        speaker_dim: int = 256,
        num_layers: int = 2,
        chunk_size: int = 250,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.num_layers = num_layers

        self.intra_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.inter_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # FiLM layers for speaker conditioning
        self.intra_film_gamma = nn.ModuleList([
            nn.Linear(speaker_dim, embed_dim) for _ in range(num_layers)
        ])
        self.intra_film_beta = nn.ModuleList([
            nn.Linear(speaker_dim, embed_dim) for _ in range(num_layers)
        ])
        self.inter_film_gamma = nn.ModuleList([
            nn.Linear(speaker_dim, embed_dim) for _ in range(num_layers)
        ])
        self.inter_film_beta = nn.ModuleList([
            nn.Linear(speaker_dim, embed_dim) for _ in range(num_layers)
        ])

        # Initialize FiLM near identity
        for gamma in self.intra_film_gamma + self.inter_film_gamma:
            nn.init.zeros_(gamma.weight)
            nn.init.zeros_(gamma.bias)
        for beta in self.intra_film_beta + self.inter_film_beta:
            nn.init.zeros_(beta.weight)
            nn.init.zeros_(beta.bias)

        self.intra_norm = nn.LayerNorm(embed_dim)
        self.inter_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """Dual-path processing with speaker conditioning.

        Args:
            x: Input ``(B, T, C)``.
            speaker_emb: Speaker embedding ``(B, D)``.

        Returns:
            Processed ``(B, T, C)``.
        """
        B, T, C = x.shape

        # Pad to multiple of chunk_size
        pad_len = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = nn.functional.pad(x, (0, 0, 0, pad_len))
        B, T_padded, C = x.shape
        num_chunks = T_padded // self.chunk_size

        for i in range(self.num_layers):
            # --- Intra-chunk ---
            x = x.reshape(B * num_chunks, self.chunk_size, C)
            x = self.intra_blocks[i](x)
            # FiLM: (B*num_chunks, chunk_size, C)
            gamma_i = self.intra_film_gamma[i](speaker_emb) + 1.0  # (B, C)
            beta_i = self.intra_film_beta[i](speaker_emb)  # (B, C)
            # Expand to match chunk shape
            gamma_i = gamma_i.unsqueeze(1).unsqueeze(1).expand(B, num_chunks, self.chunk_size, C)
            beta_i = beta_i.unsqueeze(1).unsqueeze(1).expand(B, num_chunks, self.chunk_size, C)
            x = x.reshape(B, num_chunks, self.chunk_size, C)
            x = gamma_i * x + beta_i
            x = x.reshape(B * num_chunks, self.chunk_size, C)
            x = self.intra_norm(x)

            # --- Inter-chunk ---
            x = x.reshape(B, num_chunks, self.chunk_size, C)
            x = x.transpose(1, 2).reshape(B * self.chunk_size, num_chunks, C)
            x = self.inter_blocks[i](x)
            # FiLM
            gamma_j = self.inter_film_gamma[i](speaker_emb) + 1.0
            beta_j = self.inter_film_beta[i](speaker_emb)
            gamma_j = gamma_j.unsqueeze(1).unsqueeze(1).expand(B, self.chunk_size, num_chunks, C)
            beta_j = beta_j.unsqueeze(1).unsqueeze(1).expand(B, self.chunk_size, num_chunks, C)
            x = x.reshape(B, self.chunk_size, num_chunks, C)
            x = gamma_j * x + beta_j
            x = x.reshape(B * self.chunk_size, num_chunks, C)
            x = self.inter_norm(x)

            # Back to (B, T_padded, C)
            x = x.reshape(B, self.chunk_size, num_chunks, C)
            x = x.transpose(1, 2).reshape(B, T_padded, C)

        if pad_len > 0:
            x = x[:, :T, :]

        return x


class SepFormer(nn.Module):
    """SepFormer separation network with speaker conditioning.

    Dual-path transformer that processes features with alternating
    intra-chunk and inter-chunk self-attention, conditioned on the
    target speaker embedding via FiLM layers.

    Args:
        feature_dim: Input feature dimension.
        embed_dim: Transformer embedding dimension.
        num_heads: Number of attention heads.
        ffn_dim: FFN hidden dimension.
        speaker_dim: Speaker embedding dimension.
        num_layers: Number of dual-path layers.
        chunk_size: Chunk size in frames for dual-path processing.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        speaker_dim: int = 256,
        num_layers: int = 2,
        chunk_size: int = 250,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Input projection: feature_dim -> embed_dim
        self.input_proj = nn.Conv1d(feature_dim, embed_dim, kernel_size=1)
        self.input_norm = nn.LayerNorm(embed_dim)

        # Dual-path processing with speaker conditioning
        self.dual_path = SpeakerConditionedDualPath(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            speaker_dim=speaker_dim,
            num_layers=num_layers,
            chunk_size=chunk_size,
            dropout=dropout,
        )

        # Mask estimation
        self.mask_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, feature_dim),
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
        # Project: (B, F, T) -> (B, C, T) -> (B, T, C)
        x = self.input_proj(features)  # (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.input_norm(x)

        # Dual-path processing
        x = self.dual_path(x, speaker_emb)  # (B, T, C)

        # Mask estimation: (B, T, C) -> (B, T, F) -> (B, F, T)
        mask = self.mask_conv(x)  # (B, T, F)
        mask = mask.transpose(1, 2)  # (B, F, T)

        return mask
