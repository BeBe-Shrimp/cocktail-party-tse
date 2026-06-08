"""Integration tests for the full Auditory-TSE model and components."""

import pytest
import torch

from src.models.encoder import ConvEncoder
from src.models.speaker_encoder import ECAPATDNN
from src.models.separation.conv_tasnet import ConvTasNet
from src.models.separation.sepformer import SepFormer
from src.models.separation.cross_attn_mask import CrossAttentionMaskEstimator
from src.models.auditory_tse import AuditoryTSE


class TestSpeakerEncoder:
    """Test ECAPA-TDNN speaker encoder."""

    def test_output_shape(self) -> None:
        """Speaker encoder produces fixed-dim embedding."""
        encoder = ECAPATDNN(channels=512, embedding_size=256)
        enrollment = torch.randn(2, 1, 48000)  # 3 seconds at 16kHz
        embedding = encoder(enrollment)

        assert embedding.shape == (2, 256)

    def test_l2_normalized(self) -> None:
        """Output embeddings are L2-normalized."""
        encoder = ECAPATDNN(embedding_size=256)
        enrollment = torch.randn(1, 1, 32000)
        embedding = encoder(enrollment)

        norm = torch.norm(embedding, p=2, dim=1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_variable_length_input(self) -> None:
        """Speaker encoder handles variable-length enrollment."""
        encoder = ECAPATDNN(embedding_size=256)
        for length in [16000, 32000, 48000, 64000]:
            enrollment = torch.randn(2, 1, length)
            embedding = encoder(enrollment)
            assert embedding.shape == (2, 256)

    def test_1d_input(self) -> None:
        """Handles (B, T) without channel dim."""
        encoder = ECAPATDNN(embedding_size=128)
        enrollment = torch.randn(3, 32000)
        embedding = encoder(enrollment)
        assert embedding.shape == (3, 128)


class TestConvTasNet:
    """Test Conv-TasNet separation network."""

    def test_output_shape(self) -> None:
        """ConvTasNet produces a valid mask."""
        model = ConvTasNet(feature_dim=256, speaker_dim=128)
        features = torch.randn(2, 256, 100)
        speaker_emb = torch.randn(2, 128)

        mask = model(features, speaker_emb)
        assert mask.shape == (2, 256, 100)
        assert mask.min() >= 0 and mask.max() <= 1  # Sigmoid output

    def test_causal_mode(self) -> None:
        """ConvTasNet supports causal convolutions."""
        model = ConvTasNet(feature_dim=256, speaker_dim=128, causal=True)
        features = torch.randn(1, 256, 50)
        speaker_emb = torch.randn(1, 128)

        mask = model(features, speaker_emb)
        assert mask.shape == (1, 256, 50)


class TestSepFormer:
    """Test SepFormer separation network."""

    def test_output_shape(self) -> None:
        """SepFormer produces a valid mask."""
        model = SepFormer(
            feature_dim=256, embed_dim=128, speaker_dim=128,
            num_layers=1, chunk_size=50,
        )
        features = torch.randn(2, 256, 100)
        speaker_emb = torch.randn(2, 128)

        mask = model(features, speaker_emb)
        assert mask.shape == (2, 256, 100)
        assert mask.min() >= 0 and mask.max() <= 1


class TestCrossAttentionMask:
    """Test cross-attention mask estimator."""

    def test_output_shape(self) -> None:
        """Cross-attention module produces valid mask."""
        model = CrossAttentionMaskEstimator(
            feature_dim=256, speaker_dim=128, num_heads=4, num_layers=2,
        )
        features = torch.randn(2, 256, 100)
        speaker_emb = torch.randn(2, 128)

        mask = model(features, speaker_emb)
        assert mask.shape == (2, 256, 100)
        assert mask.min() >= 0 and mask.max() <= 1


class TestAuditoryTSE:
    """Integration tests for the full Auditory-TSE model."""

    @pytest.fixture
    def model(self) -> AuditoryTSE:
        """Create a small model for testing."""
        return AuditoryTSE(
            encoder_kernel_size=16,
            encoder_stride=8,
            encoder_channels=256,  # Smaller for fast tests
            speaker_embedding_dim=128,
            separation_type="conv_tasnet",
            bottleneck_channels=128,
            num_tcn_blocks=4,
            tcn_repeats=2,
        )

    def test_forward_pass(self, model: AuditoryTSE) -> None:
        """Model runs forward pass without errors."""
        mixture = torch.randn(2, 16000)
        enrollment = torch.randn(2, 32000)

        output = model(mixture, enrollment)

        assert "waveform" in output
        assert "speaker_embedding" in output
        assert output["waveform"].shape[0] == 2
        assert output["speaker_embedding"].shape == (2, 128)

    def test_separate_method(self, model: AuditoryTSE) -> None:
        """Convenience method works."""
        mixture = torch.randn(1, 8000)
        enrollment = torch.randn(1, 16000)

        separated = model.separate(mixture, enrollment)
        assert separated.dim() == 3  # (B, 1, T)
        assert separated.shape[0] == 1

    def test_speaker_embedding_caching(self, model: AuditoryTSE) -> None:
        """Pre-computed speaker embeddings work."""
        mixture = torch.randn(1, 8000)
        enrollment = torch.randn(1, 16000)

        # Compute embedding once
        emb = model.get_speaker_embedding(enrollment)
        assert emb.shape == (1, 128)

        # Use cached embedding
        separated, mask = model.separate_with_embedding(mixture, emb)
        assert separated.dim() == 3

    def test_return_mask(self, model: AuditoryTSE) -> None:
        """Model can return the estimated mask."""
        mixture = torch.randn(1, 8000)
        enrollment = torch.randn(1, 8000)

        output = model(mixture, enrollment, return_mask=True)
        assert "mask" in output
        assert output["mask"].shape[0] == 1

    def test_separation_types(self) -> None:
        """All separation types can be instantiated."""
        for sep_type in ["conv_tasnet", "sepformer", "cross_attn"]:
            model = AuditoryTSE(
                encoder_channels=128,
                speaker_embedding_dim=64,
                separation_type=sep_type,
            )
            output = model(
                torch.randn(1, 4000),
                torch.randn(1, 4000),
            )
            assert "waveform" in output

    def test_gradient_flow(self, model: AuditoryTSE) -> None:
        """Gradients flow through the model."""
        mixture = torch.randn(1, 8000)
        enrollment = torch.randn(1, 8000)

        output = model(mixture, enrollment)
        loss = output["waveform"].mean()
        loss.backward()

        # Check that encoder has gradients
        assert model.encoder.conv.weight.grad is not None
        # Check speaker encoder has gradients
        assert model.speaker_encoder.conv1.weight.grad is not None

    def test_inference_chunking(self, model: AuditoryTSE) -> None:
        """Chunked inference works for long audio."""
        long_mixture = torch.randn(1, 32000)  # 2 seconds
        enrollment = torch.randn(1, 16000)

        output = model.inference(
            long_mixture, enrollment,
            chunk_size_seconds=0.5,
            overlap_seconds=0.1,
        )
        assert output.dim() == 3  # (B, 1, T)
        assert output.shape[-1] == 32000

    def test_dual_path_encoder_model(self) -> None:
        """Model with dual-path encoder works."""
        model = AuditoryTSE(
            encoder_channels=256,
            speaker_embedding_dim=128,
            use_dual_path_encoder=True,
        )
        output = model(
            torch.randn(1, 8000),
            torch.randn(1, 8000),
        )
        assert "waveform" in output
