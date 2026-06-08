"""Evaluator — runs evaluation loop over a test dataset."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.models.auditory_tse import AuditoryTSE

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates a trained TSE model on a test dataset.

    Computes standard metrics (SI-SDRi, PESQ, STOI) for each sample
    and reports aggregate statistics.

    Args:
        model: Trained AuditoryTSE model.
        test_loader: DataLoader for test data.
        device: Device for computation.
        output_dir: Optional directory to save results.
        save_audio: If True, save separated audio files.
    """

    def __init__(
        self,
        model: AuditoryTSE,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        output_dir: Optional[Path] = None,
        save_audio: bool = False,
    ) -> None:
        self.model = model
        self.test_loader = test_loader

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = self.model.to(self.device)
        self.model.eval()

        self.output_dir = Path(output_dir) if output_dir else None
        self.save_audio = save_audio

        if self.output_dir and self.save_audio:
            audio_dir = self.output_dir / "separated_audio"
            audio_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(
        self, compute_pesq_stoi: bool = True, verbose: bool = True
    ) -> Dict[str, float]:
        """Run evaluation loop.

        Args:
            compute_pesq_stoi: If True, compute PESQ and STOI (slower).
            verbose: If True, show progress bar.

        Returns:
            Dictionary with averaged metrics across the test set.
        """
        all_metrics: Dict[str, List[float]] = {
            "si_sdr": [],
            "si_sdri": [],
            "pesq": [],
            "stoi": [],
        }

        sample_idx = 0
        loader = (
            tqdm(self.test_loader, desc="Evaluating")
            if verbose
            else self.test_loader
        )

        for batch in loader:
            mixture = batch["mixture"].to(self.device)
            enrollment = batch["enrollment"].to(self.device)
            target = batch["target"].to(self.device)

            # Forward
            output = self.model(mixture, enrollment)
            estimated = output["waveform"].squeeze(1)  # (B, T)

            # Compute metrics per sample in batch
            for i in range(mixture.shape[0]):
                est_i = estimated[i]
                tgt_i = target[i]
                mix_i = mixture[i]

                # Trim to same length
                min_len = min(est_i.shape[-1], tgt_i.shape[-1])
                est_i = est_i[:min_len]
                tgt_i = tgt_i[:min_len]
                mix_i = mix_i[:min_len]

                metrics = compute_metrics(
                    est_i.cpu(),
                    tgt_i.cpu(),
                    mix_i.cpu(),
                    sample_rate=self.model.sample_rate,
                    compute_all=compute_pesq_stoi,
                )

                for key in all_metrics:
                    if key in metrics:
                        all_metrics[key].append(metrics[key])

                # Save audio
                if self.save_audio and self.output_dir:
                    self._save_audio(est_i, sample_idx)
                    sample_idx += 1

        # Aggregate
        summary = {}
        for key, values in all_metrics.items():
            if values:
                summary[key] = float(torch.tensor(values).mean().item())
                summary[f"{key}_std"] = float(torch.tensor(values).std().item())

        # Report
        logger.info("=" * 50)
        logger.info("Evaluation Results:")
        for key in ["si_sdr", "si_sdri", "pesq", "stoi"]:
            if key in summary:
                logger.info(f"  {key.upper():10s}: {summary[key]:.4f} ± {summary.get(f'{key}_std', 0):.4f}")
        logger.info("=" * 50)

        return summary

    def _save_audio(self, waveform: torch.Tensor, idx: int) -> None:
        """Save separated waveform to WAV file.

        Args:
            waveform: Audio tensor ``(T,)``.
            idx: Sample index for naming.
        """
        try:
            import soundfile as sf

            audio_dir = self.output_dir / "separated_audio"
            audio_np = waveform.cpu().numpy()
            sf.write(str(audio_dir / f"separated_{idx:05d}.wav"), audio_np, self.model.sample_rate)
        except Exception as e:
            logger.warning(f"Failed to save audio {idx}: {e}")
