"""Tests for loss functions and evaluation metrics."""

import pytest
import torch

from src.training.losses import si_snr, si_snr_loss, SiSNRLoss, CompositeLoss


class TestSISNR:
    """Test SI-SNR computation."""

    def test_perfect_reconstruction(self) -> None:
        """SI-SNR is high for identical signals."""
        signal = torch.randn(1, 16000)
        score = si_snr(signal, signal)
        # Should be very high for identical signals (> 30 dB)
        assert score.item() > 30

    def test_scaled_reconstruction(self) -> None:
        """SI-SNR is scale-invariant: scaling doesn't change score."""
        target = torch.randn(1, 16000)
        estimate = target * 3.0  # Scale by 3x
        score = si_snr(estimate, target)
        assert score.item() > 30

    def test_noise_is_worse(self) -> None:
        """SI-SNR is lower for noise than for signal."""
        target = torch.randn(1, 16000)
        noise = torch.randn(1, 16000)
        estimate = target + noise * 0.1  # Add small noise

        score_clean = si_snr(target, target)
        score_noisy = si_snr(estimate, target)

        assert score_noisy.item() < score_clean.item()

    def test_zero_mean_normalization(self) -> None:
        """SI-SNR works correctly after zero-mean normalization."""
        target = torch.randn(1, 16000) + 5.0  # Add DC offset
        estimate = target.clone()

        score = si_snr(estimate, target)
        assert score.item() > 30  # DC offset should be removed

    def test_batch_computation(self) -> None:
        """SI-SNR works on batches."""
        target = torch.randn(4, 8000)
        estimate = target + torch.randn(4, 8000) * 0.05

        scores = si_snr(estimate, target)
        assert scores.shape == (4,)
        assert (scores > -50).all()  # Reasonable range

    def test_orthogonal_signals(self) -> None:
        """Orthogonal signals give very low SI-SNR."""
        target = torch.randn(1, 1000)
        # Create a signal orthogonal to target
        estimate = torch.randn(1, 1000)
        # Remove projection onto target
        proj = (estimate * target).sum() / (target * target).sum() * target
        estimate = estimate - proj

        score = si_snr(estimate, target)
        assert score.item() < -10  # Orthogonal => low SI-SNR


class TestSISNRLoss:
    """Test SI-SNR loss function."""

    def test_perfect_loss_is_negative(self) -> None:
        """Loss is negative for good reconstruction (we minimize -SI-SNR)."""
        signal = torch.randn(2, 8000)
        loss = si_snr_loss(signal, signal)
        assert loss.item() < -20  # Very negative = very good

    def test_noise_loss_is_higher(self) -> None:
        """Loss is higher (less negative) for worse reconstruction."""
        signal = torch.randn(2, 8000)
        noisy = signal + torch.randn(2, 8000)

        loss_clean = si_snr_loss(signal, signal)
        loss_noisy = si_snr_loss(noisy, signal)

        assert loss_noisy.item() > loss_clean.item()

    def test_with_mask(self) -> None:
        """Masked loss ignores padded regions."""
        signal = torch.randn(2, 100)
        mask = torch.ones(2, 100, dtype=torch.bool)
        mask[0, 50:] = False  # Second half of first sample is padding
        mask[1, 80:] = False

        loss = si_snr_loss(signal, signal, mask=mask)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_empty_mask(self) -> None:
        """Loss handles completely masked samples gracefully."""
        signal = torch.randn(2, 100)
        mask = torch.zeros(2, 100, dtype=torch.bool)
        loss = si_snr_loss(signal, signal, mask=mask)
        assert loss.item() == 0.0


class TestSiSNRLossModule:
    """Test the nn.Module wrapper."""

    def test_module_interface(self) -> None:
        """Module works like a standard loss function."""
        loss_fn = SiSNRLoss()
        estimate = torch.randn(4, 500, requires_grad=True)
        target = torch.randn(4, 500)

        loss = loss_fn(estimate, target)
        assert loss.dim() == 0  # Scalar
        assert loss.requires_grad  # Differentiable


class TestCompositeLoss:
    """Test composite loss function."""

    def test_si_snr_only(self) -> None:
        """Composite loss with only SI-SNR works."""
        loss_fn = CompositeLoss(si_snr_weight=1.0)
        estimate = torch.randn(2, 1000)
        target = torch.randn(2, 1000)

        total, components = loss_fn(estimate, target)
        assert "si_snr" in components
        assert "l1" not in components
        assert total.dim() == 0

    def test_with_l1(self) -> None:
        """Composite loss with L1 works."""
        loss_fn = CompositeLoss(si_snr_weight=1.0, l1_weight=0.1)
        estimate = torch.randn(2, 1000)
        target = torch.randn(2, 1000)

        total, components = loss_fn(estimate, target)
        assert "si_snr" in components
        assert "l1" in components
        assert total.dim() == 0

    def test_with_stft(self) -> None:
        """Composite loss with STFT magnitude works."""
        loss_fn = CompositeLoss(si_snr_weight=1.0, stft_weight=0.1)
        estimate = torch.randn(2, 4000)
        target = torch.randn(2, 4000)

        total, components = loss_fn(estimate, target)
        assert "stft" in components
