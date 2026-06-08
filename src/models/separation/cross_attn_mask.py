"""Cross-Attention Mask Estimator — the core "top-down attention" module.

This module implements the attention mechanism that simulates human auditory
top-down attention: the target speaker's identity (speaker embedding) serves
as a query that attends to the mixture features (key/value), producing a mask
that isolates the target speaker's voice.

The design is inspired by how the human brain uses high-level cognitive
information ("who am I listening to?") to selectively enhance the neural
processing of the target speaker's speech stream in the auditory cortex.
"""

from typing import Optional

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer for speaker conditioning.

    Given a speaker embedding, FiLM produces scale (gamma) and shift (beta)
    parameters that modulate the acoustic features. This is a simple but
    effective form of "top-down" conditioning — the high-level speaker identity
    modulates the low-level acoustic processing.

    Reference: Perez et al. (2018) — *FiLM: Visual Reasoning with a General
    Conditioning Layer* (adapted here for speaker conditioning in audio).

    Args:
        feature_dim: Dimensionality of the acoustic features to modulate.
        speaker_dim: Dimensionality of the speaker embedding.
    """

    def __init__(self, feature_dim: int, speaker_dim: int) -> None:
        super().__init__()
        self.gamma_fc = nn.Linear(speaker_dim, feature_dim)
        self.beta_fc = nn.Linear(speaker_dim, feature_dim)

        # Initialize near-identity: gamma ~ 1, beta ~ 0
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, features: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            features: Acoustic features ``(B, C, T)``.
            speaker_emb: Speaker embedding ``(B, D)``.

        Returns:
            Conditioned features ``(B, C, T)``.
        """
        gamma = self.gamma_fc(speaker_emb).unsqueeze(-1) + 1.0  # (B, C, 1), centered at 1
        beta = self.beta_fc(speaker_emb).unsqueeze(-1)  # (B, C, 1), centered at 0
        return gamma * features + beta


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block for speaker-guided mask estimation.

    The speaker embedding (query) attends to the mixture features (key/value)
    through multi-head cross-attention, producing a context vector that
    captures the target speaker's acoustic properties in the mixture.
    A gating mechanism then combines this context with the original features.

    Args:
        embed_dim: Feature dimension for attention computation.
        num_heads: Number of attention heads.
        speaker_dim: Speaker embedding dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        speaker_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Project speaker embedding to query
        self.q_proj = nn.Linear(speaker_dim, embed_dim)

        # Project mixture features to key and value
        self.k_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.v_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

        # Output projection
        self.out_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, features: torch.Tensor, speaker_emb: torch.Tensor
    ) -> torch.Tensor:
        """Apply cross-attention between speaker embedding and mixture features.

        Args:
            features: Mixture features ``(B, C, T)``.
            speaker_emb: Speaker embedding ``(B, D)``.

        Returns:
            Attention-modulated features ``(B, C, T)``.
        """
        B, C, T = features.shape

        # Query from speaker embedding: (B, embed_dim) -> (B, num_heads, 1, head_dim)
        q = self.q_proj(speaker_emb)  # (B, embed_dim)
        q = q.view(B, self.num_heads, 1, self.head_dim) * self.scale

        # Key, Value from mixture features: (B, C, T) -> (B, num_heads, head_dim, T)
        k = self.k_proj(features).view(B, self.num_heads, self.head_dim, T)
        v = self.v_proj(features).view(B, self.num_heads, self.head_dim, T)

        # Attention scores: (B, num_heads, 1, T)
        attn_scores = torch.matmul(q, k)  # Soft attention over time
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: (B, num_heads, head_dim, T) per query position
        context = torch.matmul(attn_weights, v.transpose(-2, -1))  # ? wait
        # Actually: attn_weights @ v = (B, num_heads, 1, T) @ (B, num_heads, T, head_dim)
        # Let me recompute carefully
        v_permuted = v.transpose(-2, -1)  # (B, num_heads, T, head_dim)
        context = torch.matmul(attn_weights, v_permuted)  # (B, num_heads, 1, head_dim)
        context = context.squeeze(2)  # (B, num_heads, head_dim)
        context = context.reshape(B, -1)  # (B, C)

        # Expand context to match features: (B, C, T)
        context = context.unsqueeze(-1).expand(-1, -1, T)  # (B, C, T)

        # Gate: combine original features with attention context
        gate_input = torch.cat([features, context], dim=1)  # (B, 2*C, T)
        gate_values = self.gate(gate_input)  # (B, C, T)

        # Gated combination
        output = gate_values * features + (1 - gate_values) * context
        output = self.out_proj(output)

        # Layer norm (apply over channel dim)
        output = self.norm(output.transpose(1, 2)).transpose(1, 2)

        return output


class CrossAttentionMaskEstimator(nn.Module):
    """Cross-attention based mask estimator for target speaker extraction.

    This module takes mixture acoustic features and a target speaker embedding,
    and produces a soft mask that isolates the target speaker's voice in the
    feature domain.

    The workflow:
    1. Project mixture features through FiLM conditioning layers
       (speaker-guided feature modulation — simulating top-down attention)
    2. Apply multi-head cross-attention: speaker embedding attends to
       mixture features to find target-speaker-relevant information
    3. Estimate a soft mask via 1x1 convolution + sigmoid

    Args:
        feature_dim: Input feature dimension.
        speaker_dim: Speaker embedding dimension.
        num_heads: Number of cross-attention heads.
        num_layers: Number of FiLM + cross-attention layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        speaker_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.speaker_dim = speaker_dim

        # Input projection to match dimensions if needed
        self.input_proj = None
        if feature_dim != speaker_dim:
            self.input_proj = nn.Conv1d(feature_dim, speaker_dim, kernel_size=1)

        # The dimension of features flowing through the FiLM + cross-attention layers
        working_dim = speaker_dim if self.input_proj is not None else feature_dim

        # Stack of FiLM + Cross-Attention layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "film": FiLMLayer(working_dim, speaker_dim),
                        "cross_attn": CrossAttentionBlock(
                            embed_dim=working_dim,
                            num_heads=num_heads,
                            speaker_dim=speaker_dim,
                            dropout=dropout,
                        ),
                    }
                )
            )

        # Mask estimation: 1x1 conv -> sigmoid
        self.mask_conv = nn.Sequential(
            nn.Conv1d(working_dim, working_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(working_dim, feature_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, features: torch.Tensor, speaker_emb: torch.Tensor
    ) -> torch.Tensor:
        """Estimate target speaker mask from mixture features and speaker embedding.

        Args:
            features: Mixture features ``(B, F, T)`` from the auditory encoder.
            speaker_emb: Speaker embedding ``(B, D_spk)`` from the speaker encoder.

        Returns:
            Soft mask ``(B, F, T)`` in range [0, 1].
            Multiply element-wise with ``features`` to isolate target speaker.
        """
        # Project input if needed
        if self.input_proj is not None:
            x = self.input_proj(features)
        else:
            x = features

        # Apply FiLM + Cross-Attention layers
        for layer in self.layers:
            residual = x
            x = layer["film"](x, speaker_emb)  # Speaker-conditioned modulation
            x = layer["cross_attn"](x, speaker_emb)  # Cross-attention
            x = self.dropout(x)
            x = x + residual  # Residual connection

        # Estimate mask
        mask = self.mask_conv(x)  # (B, F, T)
        return mask
