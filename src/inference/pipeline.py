"""End-to-end inference pipeline for target speaker extraction.

Provides a simple API for loading a trained model and running inference
on arbitrary audio files or in-memory tensors.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from src.models.auditory_tse import AuditoryTSE

logger = logging.getLogger(__name__)


class InferencePipeline:
    """End-to-end inference pipeline for Auditory-TSE.

    Handles model loading, audio I/O, preprocessing, inference,
    and post-processing. Designed for both interactive use and
    batch processing.

    Args:
        model_path: Path to a trained model checkpoint (.ckpt/.pt).
        device: Device for inference (auto-detected if None).
        sample_rate: Target sample rate (must match training).
        chunk_duration_seconds: Duration of each processing chunk
            for long audio. Set to 0 to disable chunking.
        overlap_seconds: Overlap between consecutive chunks.

    Example:
        >>> pipeline = InferencePipeline.from_checkpoint("checkpoints/best.ckpt")
        >>> separated = pipeline.run("mixture.wav", "enrollment.wav", "output.wav")
    """

    def __init__(
        self,
        model: AuditoryTSE,
        device: Optional[torch.device] = None,
        sample_rate: int = 16000,
        chunk_duration_seconds: float = 10.0,
        overlap_seconds: float = 1.0,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.sample_rate = sample_rate
        self.chunk_duration_seconds = chunk_duration_seconds
        self.overlap_seconds = overlap_seconds

        self.model = model.to(device)
        self.model.eval()
        logger.info(f"InferencePipeline ready | device={device}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        sample_rate: int = 16000,
        **model_kwargs,
    ) -> "InferencePipeline":
        """Load a trained model from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.ckpt.
            device: Device for inference.
            sample_rate: Target sample rate.
            **model_kwargs: Arguments passed to AuditoryTSE constructor.

        Returns:
            Initialized InferencePipeline.
        """
        checkpoint_path = Path(checkpoint_path)

        # Create model with kwargs or defaults
        model = AuditoryTSE(
            encoder_kernel_size=model_kwargs.get("encoder_kernel_size", 16),
            encoder_stride=model_kwargs.get("encoder_stride", 8),
            encoder_channels=model_kwargs.get("encoder_channels", 512),
            speaker_embedding_dim=model_kwargs.get("speaker_embedding_dim", 256),
            separation_type=model_kwargs.get("separation_type", "conv_tasnet"),
            sample_rate=sample_rate,
            use_dual_path_encoder=model_kwargs.get("use_dual_path_encoder", False),
            **model_kwargs,
        )

        # Load weights
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Assume raw state dict
            model.load_state_dict(checkpoint)

        logger.info(f"Model loaded from {checkpoint_path}")
        return cls(model, device=device, sample_rate=sample_rate)

    @torch.no_grad()
    def run(
        self,
        mixture_path: Union[str, Path],
        enrollment_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> torch.Tensor:
        """Run inference on audio files.

        Args:
            mixture_path: Path to mixture audio file (multi-speaker).
            enrollment_path: Path to enrollment audio file (target speaker reference).
            output_path: Path to save separated audio. If None, only returns tensor.

        Returns:
            Separated waveform tensor ``(T,)``.
        """
        # Load audio
        mixture = self._load_audio(mixture_path)
        enrollment = self._load_audio(enrollment_path)

        # Run inference
        separated = self.run_tensor(mixture, enrollment)

        # Save output
        if output_path is not None:
            self._save_audio(separated, output_path)
            logger.info(f"Separated audio saved to {output_path}")

        return separated

    @torch.no_grad()
    def run_tensor(
        self,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
    ) -> torch.Tensor:
        """Run inference on in-memory tensors.

        Args:
            mixture: Mixture waveform ``(T,)`` or ``(1, T)``.
            enrollment: Enrollment waveform ``(T,)`` or ``(1, T)``.

        Returns:
            Separated waveform ``(T,)``.
        """
        # Ensure 2D: (1, T)
        if mixture.dim() == 1:
            mixture = mixture.unsqueeze(0)
        if enrollment.dim() == 1:
            enrollment = enrollment.unsqueeze(0)

        # Move to device
        mixture = mixture.to(self.device)
        enrollment = enrollment.to(self.device)

        # Run model
        if self.chunk_duration_seconds > 0:
            separated = self.model.inference(
                mixture,
                enrollment,
                chunk_size_seconds=self.chunk_duration_seconds,
                overlap_seconds=self.overlap_seconds,
            )
        else:
            separated = self.model.separate(mixture, enrollment)

        # Return as 1D
        return separated.squeeze(0).squeeze(0).cpu()

    @torch.no_grad()
    def run_with_cached_embedding(
        self,
        mixture: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Run inference with a pre-computed speaker embedding.

        Useful for batch processing: compute embeddings once per speaker,
        then apply to all mixtures involving that speaker.

        Args:
            mixture: Mixture waveform ``(T,)``.
            speaker_embedding: Pre-computed speaker embedding ``(D_spk,)``.

        Returns:
            Separated waveform ``(T,)``.
        """
        if mixture.dim() == 1:
            mixture = mixture.unsqueeze(0)

        mixture = mixture.to(self.device)
        speaker_embedding = speaker_embedding.unsqueeze(0).to(self.device)

        separated, _ = self.model.separate_with_embedding(mixture, speaker_embedding)
        return separated.squeeze(0).squeeze(0).cpu()

    def compute_speaker_embedding(self, enrollment: torch.Tensor) -> torch.Tensor:
        """Compute speaker embedding for caching.

        Args:
            enrollment: Enrollment waveform ``(T,)``.

        Returns:
            Speaker embedding ``(D_spk,)``.
        """
        if enrollment.dim() == 1:
            enrollment = enrollment.unsqueeze(0)
        enrollment = enrollment.to(self.device)
        emb = self.model.get_speaker_embedding(enrollment)
        return emb.squeeze(0).cpu()

    def _load_audio(self, path: Union[str, Path]) -> torch.Tensor:
        """Load audio from file and convert to tensor.

        Args:
            path: Path to audio file.

        Returns:
            Audio tensor ``(1, T)`` at target sample rate.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            import soundfile as sf

            audio, sr = sf.read(str(path))
            audio = torch.from_numpy(audio.T).float()  # (C, T) or (T,)
        except Exception:
            import torchaudio

            audio, sr = torchaudio.load(str(path))

        # Resample if needed
        if sr != self.sample_rate:
            logger.info(f"Resampling from {sr}Hz to {self.sample_rate}Hz")
            try:
                import torchaudio

                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                audio = resampler(audio)
            except ImportError:
                ratio = self.sample_rate / sr
                new_len = int(audio.shape[-1] * ratio)
                audio = nn.functional.interpolate(
                    audio.unsqueeze(0) if audio.dim() == 1 else audio,
                    size=new_len,
                    mode="linear",
                    align_corners=False,
                )
                if audio.dim() > 1 and audio.shape[0] == 1 and audio.dim() > 1:
                    audio = audio.squeeze(0)

        # Ensure mono
        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)  # Mix to mono

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (1, T)

        # Normalize
        peak = audio.abs().max()
        if peak > 1.0:
            audio = audio / peak

        return audio

    def _save_audio(self, waveform: torch.Tensor, path: Union[str, Path]) -> None:
        """Save waveform to audio file.

        Args:
            waveform: Audio tensor ``(T,)``.
            path: Output file path.
        """
        audio_np = waveform.numpy()
        try:
            import soundfile as sf

            sf.write(str(path), audio_np, self.sample_rate)
        except Exception:
            import torchaudio

            torchaudio.save(str(path), waveform.unsqueeze(0), self.sample_rate)
