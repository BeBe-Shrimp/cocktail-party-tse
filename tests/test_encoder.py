"""Tests for the auditory encoder and decoder modules."""

import pytest
import torch

from src.models.encoder import ConvEncoder, DualPathEncoder
from src.models.decoder import ConvDecoder


class TestConvEncoder:
    """Test the Conv1D encoder."""

    def test_output_shape(self) -> None:
        """Encoder produces expected output shape."""
        encoder = ConvEncoder(
            kernel_size=16, stride=8, in_channels=1, out_channels=512
        )
        waveform = torch.randn(2, 1, 16000)  # 1 second at 16kHz
        features = encoder(waveform)

        expected_length = encoder.get_output_length(16000)
        assert features.shape == (2, 512, expected_length)

    def test_1d_input(self) -> None:
        """Encoder handles (B, T) input by adding channel dim."""
        encoder = ConvEncoder(kernel_size=16, stride=8)
        waveform = torch.randn(4, 16000)
        features = encoder(waveform)
        assert features.shape[0] == 4
        assert features.shape[1] == 512

    def test_get_output_length(self) -> None:
        """Output length calculation is correct."""
        encoder = ConvEncoder(kernel_size=16, stride=8)
        # Formula: (T - kernel_size) // stride + 1
        assert encoder.get_output_length(16000) == (16000 - 16) // 8 + 1

    def test_different_kernel_sizes(self) -> None:
        """Encoder works with different kernel sizes."""
        for kernel_size in [8, 16, 32, 64]:
            encoder = ConvEncoder(kernel_size=kernel_size, stride=kernel_size // 2)
            waveform = torch.randn(1, 1, 8000)
            features = encoder(waveform)
            assert features.shape[0] == 1
            assert features.shape[1] == 512

    @pytest.mark.gpu
    def test_gpu(self) -> None:
        """Encoder works on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        encoder = ConvEncoder().cuda()
        waveform = torch.randn(2, 1, 8000).cuda()
        features = encoder(waveform)
        assert features.device.type == "cuda"


class TestConvDecoder:
    """Test the transposed Conv1D decoder."""

    def test_encoder_decoder_symmetry(self) -> None:
        """Decoder approximately inverts encoder (reconstruction)."""
        encoder = ConvEncoder(kernel_size=16, stride=8, out_channels=512)
        decoder = ConvDecoder(kernel_size=16, stride=8, in_channels=512)

        waveform = torch.randn(1, 1, 16000)
        features = encoder(waveform)
        reconstructed = decoder(features)

        # Check shape: should be close to original length
        assert abs(reconstructed.shape[-1] - waveform.shape[-1]) < 10

    def test_output_shape(self) -> None:
        """Decoder produces expected output shape."""
        decoder = ConvDecoder(kernel_size=16, stride=8, in_channels=512)
        features = torch.randn(2, 512, 100)
        waveform = decoder(features)

        expected_len = decoder.get_output_length(100)
        assert waveform.shape == (2, 1, expected_len)


class TestDualPathEncoder:
    """Test the dual-path (learned + Gammatone) encoder."""

    def test_output_shape(self) -> None:
        """DualPathEncoder fuses both pathways correctly."""
        encoder = DualPathEncoder(
            kernel_size=16, stride=8,
            conv_channels=256, gammatone_channels=64,
            out_channels=512,
        )
        waveform = torch.randn(2, 1, 16000)
        features = encoder(waveform)
        assert features.shape[0] == 2
        assert features.shape[1] == 512
