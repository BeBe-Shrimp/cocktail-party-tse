"""Auditory-TSE: Auditory-inspired Target Speaker Extraction model.

This is the main model that assembles all components into an end-to-end
target speaker extraction system inspired by human auditory attention:

1. **Auditory Encoder**: Encodes the mixture waveform into a learned
   time-frequency representation (with optional Gammatone filterbank branch).

2. **Speaker Encoder**: Extracts a speaker identity embedding from the
   enrollment (reference) audio of the target speaker.

3. **Separation Network**: Estimates a soft mask conditioned on the speaker
   embedding. The mask isolates the target speaker in the feature domain.

4. **Decoder**: Reconstructs the separated waveform from masked features.

The entire pipeline is end-to-end differentiable, enabling joint optimization
of the encoder, speaker encoder, separation network, and decoder.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.encoder import ConvEncoder, DualPathEncoder
from src.models.decoder import ConvDecoder
from src.models.speaker_encoder import ECAPATDNN
from src.models.separation.conv_tasnet import ConvTasNet
from src.models.separation.sepformer import SepFormer
from src.models.separation.cross_attn_mask import CrossAttentionMaskEstimator

logger = logging.getLogger(__name__)


class AuditoryTSE(nn.Module):
    """Auditory-inspired Target Speaker Extraction model.

    Given a multi-speaker mixture waveform and an enrollment waveform
    of the target speaker, extracts only the target speaker's voice.

    This model simulates key aspects of human auditory attention:
    - **Cochlear-like encoding**: Learnable filters + optional Gammatone bank
    - **Speaker identity extraction**: ECAPA-TDNN for robust speaker embedding
    - **Top-down attention**: Speaker embedding modulates feature processing
      via FiLM and/or cross-attention, analogous to how prior knowledge of the
      target speaker's voice guides selective auditory attention

    Args:
        encoder_kernel_size: Encoder/decoder kernel size in samples.
        encoder_stride: Stride for encoder/decoder.
        encoder_channels: Number of encoder output channels (feature dim).
        speaker_embedding_dim: Speaker embedding dimension.
        separation_type: Type of separation network.
            Options: ``"conv_tasnet"``, ``"sepformer"``, ``"cross_attn"``.
        sample_rate: Audio sample rate (Hz).
        use_dual_path_encoder: Whether to use the dual-path (learned + Gammatone) encoder.
        **kwargs: Additional keyword arguments passed to the separation network.
    """

    def __init__(
        self,
        encoder_kernel_size: int = 16,
        encoder_stride: int = 8,
        encoder_channels: int = 512,
        speaker_embedding_dim: int = 256,
        separation_type: str = "conv_tasnet",
        sample_rate: int = 16000,
        use_dual_path_encoder: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.encoder_channels = encoder_channels
        self.separation_type = separation_type

        # --- 1. Auditory Encoder ---
        if use_dual_path_encoder:
            self.encoder: nn.Module = DualPathEncoder(
                kernel_size=encoder_kernel_size,
                stride=encoder_stride,
                in_channels=1,
                conv_channels=encoder_channels // 2,
                gammatone_channels=64,
                out_channels=encoder_channels,
                sample_rate=sample_rate,
            )
        else:
            self.encoder = ConvEncoder(
                kernel_size=encoder_kernel_size,
                stride=encoder_stride,
                in_channels=1,
                out_channels=encoder_channels,
            )

        # --- 2. Speaker Encoder ---
        self.speaker_encoder = ECAPATDNN(
            channels=512,
            embedding_size=speaker_embedding_dim,
            sample_rate=sample_rate,
        )

        # --- 3. Separation Network ---
        self.separation = self._build_separation(separation_type, **kwargs)

        # --- 4. Decoder ---
        self.decoder = ConvDecoder(
            kernel_size=encoder_kernel_size,
            stride=encoder_stride,
            in_channels=encoder_channels,
            out_channels=1,
        )

        self._log_model_info()

    def _build_separation(self, separation_type: str, **kwargs: Any) -> nn.Module:
        """Build the separation network based on the specified type.

        Args:
            separation_type: Type of separation network.
            **kwargs: Additional arguments for the network.

        Returns:
            Instantiated separation module.

        Raises:
            ValueError: If separation_type is unsupported.
        """
        feature_dim = self.encoder_channels
        speaker_dim = self.speaker_encoder.embedding_size

        if separation_type == "conv_tasnet":
            return ConvTasNet(
                feature_dim=feature_dim,
                bottleneck_channels=kwargs.get("bottleneck_channels", 256),
                num_tcn_blocks=kwargs.get("num_tcn_blocks", 8),
                tcn_repeats=kwargs.get("tcn_repeats", 3),
                speaker_dim=speaker_dim,
                causal=kwargs.get("causal", False),
            )
        elif separation_type == "sepformer":
            return SepFormer(
                feature_dim=feature_dim,
                embed_dim=kwargs.get("embed_dim", 256),
                num_heads=kwargs.get("num_heads", 8),
                ffn_dim=kwargs.get("ffn_dim", 1024),
                speaker_dim=speaker_dim,
                num_layers=kwargs.get("num_layers", 2),
                chunk_size=kwargs.get("chunk_size", 250),
                dropout=kwargs.get("dropout", 0.1),
            )
        elif separation_type == "cross_attn":
            return CrossAttentionMaskEstimator(
                feature_dim=feature_dim,
                speaker_dim=speaker_dim,
                num_heads=kwargs.get("num_heads", 8),
                num_layers=kwargs.get("num_layers", 4),
                dropout=kwargs.get("dropout", 0.1),
            )
        else:
            raise ValueError(
                f"Unknown separation type: {separation_type}. "
                f"Choose from: 'conv_tasnet', 'sepformer', 'cross_attn'."
            )

    def _log_model_info(self) -> None:
        """Log model configuration and parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"AuditoryTSE | Separation: {self.separation_type}")
        logger.info(f"AuditoryTSE | Total params: {total_params:,}")
        logger.info(f"AuditoryTSE | Trainable params: {trainable_params:,}")

    def forward(
        self,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
        return_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Extract the target speaker's voice from a mixture.

        Args:
            mixture: Multi-speaker mixture waveform ``(B, T)`` or ``(B, 1, T)``.
            enrollment: Target speaker enrollment waveform ``(B, T_ref)`` or ``(B, 1, T_ref)``.
            return_mask: If True, also return the estimated mask.

        Returns:
            Dictionary containing:
                - ``"waveform"``: Separated waveform ``(B, 1, T_out)``.
                - ``"mask"`` (optional): Estimated mask ``(B, F, T_enc)``.
                - ``"speaker_embedding"``: Speaker embedding ``(B, D_spk)``.
                - ``"features"``: Encoded mixture features ``(B, F, T_enc)``.
        """
        # Ensure correct shape
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)  # (B, T) -> (B, 1, T)
        if enrollment.dim() == 2:
            enrollment = enrollment.unsqueeze(1)

        # 1. Encode mixture -> features
        features = self.encoder(mixture)  # (B, F, T_enc)

        # 2. Extract speaker embedding from enrollment
        speaker_emb = self.speaker_encoder(enrollment)  # (B, D_spk)

        # 3. Estimate target speaker mask
        mask = self.separation(features, speaker_emb)  # (B, F, T_enc)

        # 4. Apply mask
        masked_features = features * mask  # (B, F, T_enc)

        # 5. Decode to waveform
        separated = self.decoder(masked_features)  # (B, 1, T_out)

        output: Dict[str, torch.Tensor] = {
            "waveform": separated,
            "speaker_embedding": speaker_emb,
            "features": features,
        }
        if return_mask:
            output["mask"] = mask

        return output

    def separate(
        self,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience method: extract target speaker waveform.

        Args:
            mixture: Mixture waveform ``(B, T)``.
            enrollment: Enrollment waveform ``(B, T_ref)``.

        Returns:
            Separated waveform ``(B, 1, T_out)``.
        """
        output = self.forward(mixture, enrollment)
        return output["waveform"]

    def get_speaker_embedding(self, enrollment: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from enrollment audio only.

        Useful for pre-computing and caching speaker embeddings.

        Args:
            enrollment: Enrollment waveform ``(B, T)`` or ``(B, 1, T)``.

        Returns:
            Speaker embedding ``(B, embedding_dim)``.
        """
        if enrollment.dim() == 2:
            enrollment = enrollment.unsqueeze(1)
        return self.speaker_encoder(enrollment)

    def separate_with_embedding(
        self,
        mixture: torch.Tensor,
        speaker_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate using a pre-computed speaker embedding.

        This enables caching speaker embeddings for efficiency when the same
        target speaker is used across multiple mixtures.

        Args:
            mixture: Mixture waveform ``(B, T)``.
            speaker_emb: Pre-computed speaker embedding ``(B, D_spk)``.

        Returns:
            Tuple of (separated_waveform ``(B, 1, T_out)``, mask ``(B, F, T_enc)``).
        """
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)

        features = self.encoder(mixture)
        mask = self.separation(features, speaker_emb)
        masked_features = features * mask
        separated = self.decoder(masked_features)
        return separated, mask

    @torch.no_grad()
    def inference(
        self,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
        chunk_size_seconds: float = 10.0,
        overlap_seconds: float = 1.0,
    ) -> torch.Tensor:
        """Inference with chunk-based processing for long audio.

        Processes long mixtures in overlapping chunks to handle
        variable-length inputs efficiently.

        Args:
            mixture: Mixture waveform ``(B, T)``.
            enrollment: Enrollment waveform ``(B, T_ref)``.
            chunk_size_seconds: Duration of each processing chunk.
            overlap_seconds: Overlap between consecutive chunks.

        Returns:
            Separated waveform ``(B, 1, T_out)``.
        """
        chunk_size = int(chunk_size_seconds * self.sample_rate)
        overlap = int(overlap_seconds * self.sample_rate)
        hop = chunk_size - overlap

        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)

        B, C, total_length = mixture.shape

        # Pre-compute speaker embedding once
        speaker_emb = self.get_speaker_embedding(enrollment)

        # Process chunks
        output = torch.zeros(B, C, total_length, device=mixture.device)
        count = torch.zeros(B, C, total_length, device=mixture.device)

        window = torch.hann_window(chunk_size, device=mixture.device)
        window = window.view(1, 1, -1)

        for start in range(0, total_length, hop):
            end = min(start + chunk_size, total_length)
            chunk = mixture[:, :, start:end]

            # Pad if needed
            if chunk.shape[-1] < chunk_size:
                pad_len = chunk_size - chunk.shape[-1]
                chunk = nn.functional.pad(chunk, (0, pad_len))

            # Process
            chunk_separated, _ = self.separate_with_embedding(chunk, speaker_emb)

            # Crop to actual length
            actual_len = end - start
            chunk_separated = chunk_separated[:, :, :actual_len]

            # Overlap-add with Hanning window
            window_chunk = window[:, :, :actual_len]
            output[:, :, start:end] += chunk_separated * window_chunk
            count[:, :, start:end] += window_chunk

        # Normalize by overlap count
        output = output / count.clamp(min=1e-8)

        return output
