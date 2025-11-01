"""
Integration tests for embedding models with real model loading.

These tests require downloading actual models from HuggingFace and are marked as slow.
Run with: pytest -m integration tests/test_mert_integration.py
Skip with: pytest -m "not integration" tests/
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from embedding_models import MertModel, MuQEmbeddingModel, MusicLatentSpaceModel
from models import TrackSegment
from tests.test_utils import generate_sine_wave, assert_embedding_valid


# ==============================================================================
# MertModel Integration Tests
# ==============================================================================


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_models
def test_mert_load_real_model():
    """Test loading actual MERT-v1-330M model from HuggingFace."""
    model = MertModel(device="cpu", timeout_seconds=600)

    try:
        # This will download the model if not cached
        with model.model_session() as loaded_model:
            assert loaded_model is not None
            assert model.processor is not None
            assert model.processor.sampling_rate == 24000
    finally:
        model.shutdown()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_models
def test_mert_real_inference_with_synthetic_audio(synthetic_audio_generator):
    """Test real MERT inference with synthetic audio."""
    model = MertModel(
        device="cpu",
        chunk_duration_seconds=10.0,  # Shorter for faster test
        hop_duration_seconds=2.0,
        timeout_seconds=600,
    )

    try:
        # Generate 10 seconds of test audio
        audio = synthetic_audio_generator(duration_seconds=10.0, sample_rate=24000)

        # Create TrackSegment
        segment = TrackSegment(index=1, title="Test", start=0.0, end=10.0)

        # This requires the real model to be loaded
        with model.model_session() as loaded_model:
            # Mock the audio loading to use our synthetic audio
            original_load = model._load_audio_segment

            def mock_load_audio(music_file, offset, duration):
                return audio

            model._load_audio_segment = mock_load_audio

            try:
                result = model._embed_single_segment(
                    model=loaded_model,
                    music_file="dummy.wav",
                    track_segment=segment,
                )

                # Verify embedding properties
                assert result is not None
                assert result.index == 1
                assert result.title == "Test"
                assert len(result.embedding) == 76_800

                # Verify embedding is valid
                assert_embedding_valid(
                    result.embedding,
                    expected_dim=76_800,
                    check_normalization=True,
                )
            finally:
                model._load_audio_segment = original_load

    finally:
        model.shutdown()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_models
def test_mert_real_embedding_dimensions():
    """Validate that real MERT produces correct 76,800-D embeddings."""
    model = MertModel(
        device="cpu",
        chunk_duration_seconds=10.0,
        timeout_seconds=600,
    )

    try:
        with model.model_session() as loaded_model:
            # Generate minimal audio
            audio = generate_sine_wave(duration_seconds=5.0, sample_rate=24000)

            # Preprocess with real processor
            inputs = model.processor(
                audio,
                sampling_rate=24000,
                return_tensors="pt",
            )

            # Move to CPU
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

            # Real forward pass
            with torch.inference_mode():
                outputs = loaded_model(**inputs, output_hidden_states=True)

            # Verify 25 layers
            assert len(outputs.hidden_states) == 25

            # Verify each layer shape
            for i, hidden_state in enumerate(outputs.hidden_states):
                assert hidden_state.shape[0] == 1  # Batch size
                assert hidden_state.shape[2] == 1024  # Hidden dim

            # Extract and process
            all_layers = torch.stack(outputs.hidden_states).squeeze()
            time_reduced = all_layers.mean(dim=1)
            concatenated = time_reduced.reshape(-1)

            # Verify concatenated dimension
            assert concatenated.shape[0] == 25_600

    finally:
        model.shutdown()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_models
@pytest.mark.skip(reason="Requires actual audio file - enable manually")
def test_mert_full_pipeline_with_audio_file(tmp_path):
    """
    End-to-end test with actual audio file.

    To run this test:
    1. Place a test audio file in the test directory
    2. Remove the @pytest.mark.skip decorator
    3. Update the audio_path below
    """
    audio_path = tmp_path / "test_audio.wav"

    # Create test audio file with soundfile or similar
    # audio = generate_sine_wave(30.0, 24000)
    # import soundfile as sf
    # sf.write(str(audio_path), audio, 24000)

    if not audio_path.exists():
        pytest.skip("Test audio file not available")

    model = MertModel(device="cpu", timeout_seconds=600)

    try:
        result = model.embed_music(
            music_file=str(audio_path),
            music_name="Integration Test Track",
        )

        # Verify output structure
        assert "music_file" in result
        assert "model_id" in result
        assert "segments" in result
        assert len(result["segments"]) > 0

        # Verify first segment
        first_segment = result["segments"][0]
        assert "embedding" in first_segment
        assert len(first_segment["embedding"]) == 76_800

    finally:
        model.shutdown()


# ==============================================================================
# MuQEmbeddingModel Integration Tests
# ==============================================================================


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_models
@pytest.mark.skip(reason="Requires MuQ model - enable if available")
def test_muq_load_real_model():
    """Test loading actual MuQ MuLan model."""
    model = MuQEmbeddingModel(device="cpu", timeout_seconds=600)

    try:
        with model.model_session() as loaded_model:
            assert loaded_model is not None
    finally:
        model.shutdown()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_models
@pytest.mark.skip(reason="Requires MuQ model - enable if available")
def test_muq_text_embedding_real():
    """Test real MuQ text embedding."""
    model = MuQEmbeddingModel(device="cpu", timeout_seconds=600)

    try:
        embedding = model.embed_string("rock music with guitar")

        # MuQ produces 512-D audio embeddings
        assert len(embedding) == 512
        assert_embedding_valid(embedding, expected_dim=512, check_normalization=False)
    finally:
        model.shutdown()


# ==============================================================================
# MusicLatentSpaceModel Integration Tests
# ==============================================================================


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_models
@pytest.mark.skip(reason="Requires Music2Latent model - enable if available")
def test_latent_load_real_model():
    """Test loading actual Music2Latent EncoderDecoder."""
    model = MusicLatentSpaceModel(device="cpu", timeout_seconds=600)

    try:
        with model.model_session() as loaded_model:
            assert loaded_model is not None
    finally:
        model.shutdown()
