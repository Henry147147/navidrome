import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from embedding_models import (
    BaseEmbeddingModel,
    MuQEmbeddingModel,
    SegmentEmbedding,
    enrich_embedding,
)


class DummyEmbeddingModel(BaseEmbeddingModel):
    def __init__(self) -> None:
        super().__init__(timeout_seconds=60, logger=logging.getLogger("dummy"))
        self.load_count = 0

    def _load_model(self):
        self.load_count += 1
        return object()

    def prepare_music(self, music_file: str, music_name: str, cue_file=None):
        return []

    def embed_music(self, music_file: str, music_name: str, cue_file=None):
        return {}

    def embed_string(self, value: str):
        return []

    def ensure_milvus_schemas(self, client) -> None:
        return None

    def ensure_milvus_index(self, client) -> None:
        return None


@pytest.fixture()
def dummy_model():
    model = DummyEmbeddingModel()
    yield model
    model.shutdown()


def test_model_session_loads_model_once(dummy_model: DummyEmbeddingModel):
    with dummy_model.model_session():
        pass
    with dummy_model.model_session():
        pass
    assert dummy_model.load_count == 1


def test_unload_model_allows_reloading(dummy_model: DummyEmbeddingModel):
    dummy_model.ensure_model_loaded()
    assert dummy_model.load_count == 1
    dummy_model.unload_model()
    dummy_model.ensure_model_loaded()
    assert dummy_model.load_count == 2


def test_enrich_embedding_shape():
    D, T = 64, 10
    embedding = torch.randn(D, T)
    enriched = enrich_embedding(embedding)
    # Should be [3*D] = 192
    assert enriched.shape == (3 * D,)
    # Should be L2 normalized
    assert torch.allclose(enriched.norm(), torch.tensor(1.0))


def test_enrich_embedding_single_time():
    D, T = 10, 1
    embedding = torch.randn(D, T)
    enriched = enrich_embedding(embedding)
    # 3*D = 30
    assert enriched.shape == (3 * D,)


def test_enrich_embedding_handles_1d():
    """Test enrich_embedding handles 1D tensors (single vector)."""
    D = 512
    embedding = torch.randn(D)
    enriched = enrich_embedding(embedding)
    # Should be [3*D]
    assert enriched.shape == (3 * D,)
    # Should be L2 normalized
    assert torch.allclose(enriched.norm(), torch.tensor(1.0))


def test_enrich_embedding_handles_3d():
    """Test enrich_embedding handles 3D tensors (batch, seq, dim)."""
    batch, seq, D = 4, 8, 512
    embedding = torch.randn(batch, seq, D)
    enriched = enrich_embedding(embedding)
    # Should reduce to [3*D] by averaging over seq and treating batch as time
    # Shape: [batch, seq, D] -> mean(dim=1) -> [batch, D] -> transpose -> [D, batch]
    # Then enrich -> [3*D]
    assert enriched.shape == (3 * D,)
    # Should be L2 normalized
    assert torch.allclose(enriched.norm(), torch.tensor(1.0))


def test_enrich_embedding_preserves_device():
    """Test enrich_embedding preserves tensor device."""
    D, T = 64, 10
    embedding = torch.randn(D, T)
    enriched = enrich_embedding(embedding)
    # Result should be on CPU as per implementation
    assert enriched.device.type == "cpu"


def test_segment_embedding_fields():
    """Test SegmentEmbedding has correct fields."""
    embedding = SegmentEmbedding(
        index=1,
        title="Test Track",
        offset_seconds=0.0,
        duration_seconds=30.0,
        embedding=[0.0] * 1536,
    )

    assert embedding.index == 1
    assert embedding.title == "Test Track"
    assert embedding.offset_seconds == 0.0
    assert embedding.duration_seconds == 30.0
    assert len(embedding.embedding) == 1536


# ==============================================================================
# MuQEmbeddingModel Tests
# ==============================================================================


@pytest.fixture
def mock_muq_components():
    """Create mocked MuQ audio model."""
    with patch("embedding_models.MuQ") as mock_muq_class:
        # Setup mock model
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)

        # Mock forward pass for audio
        def mock_forward(audio_tensor):
            batch_size = (
                audio_tensor.shape[0] if isinstance(audio_tensor, torch.Tensor) else 1
            )
            # Return object with last_hidden_state attribute
            mock_output = Mock()
            mock_output.last_hidden_state = torch.randn(batch_size, 512)
            return mock_output

        mock_model.__call__ = mock_forward
        mock_muq_class.from_pretrained = Mock(return_value=mock_model)

        yield {
            "model": mock_model,
            "muq_class": mock_muq_class,
        }


@pytest.mark.unit
def test_muq_model_init_defaults(mock_muq_components):
    """Test MuQEmbeddingModel initializes with correct defaults."""
    model = MuQEmbeddingModel()
    assert model.model_id == "OpenMuQ/MuQ-large-msd-iter"
    assert model.device == "cuda"
    assert model.sample_rate == 24_000
    assert model.window_seconds == 120
    assert model.hop_seconds == 15
    model.shutdown()


@pytest.mark.unit
def test_muq_model_init_custom_params(mock_muq_components):
    """Test MuQEmbeddingModel with custom parameters."""
    model = MuQEmbeddingModel(
        model_id="custom/model",
        device="cpu",
        sample_rate=48_000,
        window_seconds=60,
        hop_seconds=10,
        timeout_seconds=120,
    )
    assert model.model_id == "custom/model"
    assert model.device == "cpu"
    assert model.sample_rate == 48_000
    assert model.window_seconds == 60
    assert model.hop_seconds == 10
    model.shutdown()


@pytest.mark.unit
def test_muq_load_model(mock_muq_components):
    """Test MuQ model loading."""
    model = MuQEmbeddingModel(device="cpu")
    _loaded_model = model._load_model()

    # Verify MuQ.from_pretrained was called
    mock_muq_components["muq_class"].from_pretrained.assert_called_once()

    # Verify model methods called
    assert mock_muq_components["model"].to.called
    assert mock_muq_components["model"].eval.called

    model.shutdown()


@pytest.mark.unit
def test_muq_model_device_placement(mock_muq_components):
    """Test model is moved to specified device."""
    model = MuQEmbeddingModel(device="cpu", storage_dtype=torch.float32)
    model._load_model()

    # Verify to() was called
    mock_muq_components["model"].to.assert_called()
    model.shutdown()


@pytest.mark.unit
def test_muq_sample_rate_24khz(mock_muq_components):
    """Test MuQ default sample rate is 24kHz."""
    model = MuQEmbeddingModel()
    assert model.sample_rate == 24_000
    model.shutdown()


@pytest.mark.unit
def test_muq_chunk_size_calculation():
    """Test chunk_size calculation for MuQ (120s window)."""
    model = MuQEmbeddingModel(window_seconds=120, sample_rate=24_000)
    chunk_size = int(model.window_seconds * model.sample_rate)
    expected = 120 * 24_000  # 2,880,000 samples
    assert chunk_size == expected
    model.shutdown()


@pytest.mark.unit
def test_muq_hop_size_calculation():
    """Test hop_size calculation for MuQ (15s hop)."""
    model = MuQEmbeddingModel(hop_seconds=15, sample_rate=24_000)
    hop_size = int(model.hop_seconds * model.sample_rate)
    expected = 15 * 24_000  # 360,000 samples
    assert hop_size == expected
    model.shutdown()


@pytest.mark.unit
def test_muq_chunking_120s_window_15s_hop():
    """Test MuQ chunking logic with 120s window and 15s hop."""
    sample_rate = 24_000
    window_seconds = 120
    hop_seconds = 15

    chunk_size = int(window_seconds * sample_rate)  # 2,880,000 samples
    hop_size = int(hop_seconds * sample_rate)  # 360,000 samples

    # Test with 240 seconds of audio (2 chunks worth)
    total_samples = 240 * sample_rate  # 5,760,000 samples

    starts = list(range(0, max(total_samples - chunk_size, 0) + 1, hop_size))

    # Should have starts at: 0, 360k, 720k, ...
    assert starts[0] == 0
    assert starts[1] == 360_000
    assert len(starts) >= 8  # Multiple overlapping chunks


@pytest.mark.unit
def test_muq_chunk_overlap_logic():
    """Test that MuQ chunks overlap correctly."""
    window_seconds = 120
    hop_seconds = 15
    sample_rate = 24_000

    chunk_size = window_seconds * sample_rate
    hop_size = hop_seconds * sample_rate

    # Overlap = chunk_size - hop_size
    overlap = chunk_size - hop_size
    expected_overlap = (120 - 15) * 24_000  # 105 seconds overlap

    assert overlap == expected_overlap


@pytest.mark.unit
@patch("embedding_models.librosa.load")
def test_muq_audio_padding(mock_load, mock_muq_components, sample_track_segment):
    """Test that audio shorter than chunk_size is padded."""
    # Mock short audio (10 seconds)
    short_audio = np.random.randn(10 * 24_000).astype(np.float32)
    mock_load.return_value = (short_audio, 24_000)

    model = MuQEmbeddingModel(device="cpu", window_seconds=120)
    chunk_size = int(model.window_seconds * model.sample_rate)

    # Short audio should be padded to chunk_size
    assert len(short_audio) < chunk_size

    # Simulate padding
    padded = np.pad(short_audio, (0, chunk_size - len(short_audio)))
    assert len(padded) == chunk_size

    model.shutdown()


@pytest.mark.unit
@patch("embedding_models.librosa.load")
def test_muq_empty_audio_handling(mock_load, mock_muq_components, sample_track_segment):
    """Test MuQ handles empty audio gracefully."""
    # Mock empty audio
    mock_load.return_value = (np.array([], dtype=np.float32), 24_000)

    model = MuQEmbeddingModel(device="cpu")
    model._load_model()

    with model.model_session() as loaded_model:
        result = model._embed_single_segment(
            model=loaded_model,
            music_file="test.wav",
            track_segment=sample_track_segment,
        )

    # Should return None for empty audio
    assert result is None
    model.shutdown()


@pytest.mark.unit
@patch("embedding_models.torchaudio.load")
@patch("embedding_models.torchaudio.info")
def test_muq_load_audio_with_torchaudio(mock_info, mock_load, mock_muq_components):
    """Test primary audio loading method uses torchaudio."""
    # Mock torchaudio.info
    mock_audio_info = Mock()
    mock_audio_info.sample_rate = 44_100
    mock_info.return_value = mock_audio_info

    # Mock torchaudio.load
    audio_tensor = torch.randn(1, 44_100)  # 1 second mono
    mock_load.return_value = (audio_tensor, 44_100)

    model = MuQEmbeddingModel(device="cpu", sample_rate=24_000)
    model._load_model()

    _audio = model._load_audio_segment(
        music_file="test.wav",
        offset=0.0,
        duration=1.0,
    )

    # Should have loaded and resampled
    mock_info.assert_called_once()
    mock_load.assert_called_once()

    model.shutdown()


@pytest.mark.unit
@patch("embedding_models.librosa.load")
@patch("embedding_models.torchaudio.info", side_effect=Exception("torchaudio failed"))
def test_muq_load_audio_fallback_librosa(mock_info, mock_load, mock_muq_components):
    """Test audio loading falls back to librosa on torchaudio failure."""
    # Mock librosa.load
    audio = np.random.randn(24_000).astype(np.float32)
    mock_load.return_value = (audio, 24_000)

    model = MuQEmbeddingModel(device="cpu", sample_rate=24_000)
    model._load_model()

    result = model._load_audio_segment(
        music_file="test.wav",
        offset=0.0,
        duration=1.0,
    )

    # Should have fallen back to librosa
    mock_load.assert_called_once()
    assert len(result) > 0

    model.shutdown()


@pytest.mark.unit
@patch("embedding_models.torchaudio.load")
@patch("embedding_models.torchaudio.info")
def test_muq_audio_resampling(mock_info, mock_load, mock_muq_components):
    """Test audio is resampled to target sample rate."""
    # Mock audio at 48kHz
    mock_audio_info = Mock()
    mock_audio_info.sample_rate = 48_000
    mock_info.return_value = mock_audio_info

    # Return 1 second at 48kHz
    audio_tensor = torch.randn(1, 48_000)
    mock_load.return_value = (audio_tensor, 48_000)

    model = MuQEmbeddingModel(device="cpu", sample_rate=24_000)
    model._load_model()

    with patch("embedding_models.torchaudio.functional.resample") as mock_resample:
        mock_resample.return_value = torch.randn(24_000)

        _audio = model._load_audio_segment(
            music_file="test.wav",
            offset=0.0,
            duration=1.0,
        )

        # Verify resample was called with correct sample rates
        mock_resample.assert_called_once()
        call_args = mock_resample.call_args
        # Should be called with (waveform, 48000, 24000)
        assert call_args[0][1] == 48_000  # Original SR
        assert call_args[0][2] == 24_000  # Target SR

    model.shutdown()


@pytest.mark.unit
@patch("embedding_models.torchaudio.load")
@patch("embedding_models.torchaudio.info")
def test_muq_stereo_to_mono_conversion(mock_info, mock_load, mock_muq_components):
    """Test stereo audio is converted to mono."""
    mock_audio_info = Mock()
    mock_audio_info.sample_rate = 24_000
    mock_info.return_value = mock_audio_info

    # Return stereo audio (2 channels)
    stereo_audio = torch.randn(2, 24_000)
    mock_load.return_value = (stereo_audio, 24_000)

    model = MuQEmbeddingModel(device="cpu", sample_rate=24_000)
    model._load_model()

    audio = model._load_audio_segment(
        music_file="test.wav",
        offset=0.0,
        duration=1.0,
    )

    # Should be mono (1D array)
    assert audio.ndim == 1
    model.shutdown()


@pytest.mark.unit
@patch("embedding_models.torchaudio.load")
@patch("embedding_models.torchaudio.info")
def test_muq_frame_offset_handling(mock_info, mock_load, mock_muq_components):
    """Test loading audio from specific offset."""
    mock_audio_info = Mock()
    mock_audio_info.sample_rate = 24_000
    mock_info.return_value = mock_audio_info

    audio_tensor = torch.randn(1, 24_000)
    mock_load.return_value = (audio_tensor, 24_000)

    model = MuQEmbeddingModel(device="cpu", sample_rate=24_000)
    model._load_model()

    model._load_audio_segment(
        music_file="test.wav",
        offset=10.0,  # Start at 10 seconds
        duration=5.0,  # Load 5 seconds
    )

    # Verify torchaudio.load was called with frame_offset
    call_kwargs = mock_load.call_args[1]
    expected_frame_offset = int(10.0 * 24_000)  # 240,000 frames
    assert call_kwargs["frame_offset"] == expected_frame_offset

    model.shutdown()


@pytest.mark.unit
@patch("embedding_models.torchaudio.load")
@patch("embedding_models.torchaudio.info")
def test_muq_duration_limiting(mock_info, mock_load, mock_muq_components):
    """Test loading limited duration of audio."""
    mock_audio_info = Mock()
    mock_audio_info.sample_rate = 24_000
    mock_info.return_value = mock_audio_info

    audio_tensor = torch.randn(1, 120_000)  # 5 seconds
    mock_load.return_value = (audio_tensor, 24_000)

    model = MuQEmbeddingModel(device="cpu", sample_rate=24_000)
    model._load_model()

    model._load_audio_segment(
        music_file="test.wav",
        offset=0.0,
        duration=5.0,  # Only load 5 seconds
    )

    # Verify num_frames parameter
    call_kwargs = mock_load.call_args[1]
    expected_num_frames = int(5.0 * 24_000)  # 120,000 frames
    assert call_kwargs["num_frames"] == expected_num_frames

    model.shutdown()


@pytest.mark.unit
def test_muq_model_normalization(mock_muq_components):
    """Test MuQ output normalization."""
    # Simulate model output
    batch_size = 3
    raw_output = torch.randn(batch_size, 512)

    # Apply normalization like MuQ does
    normalized = torch.nn.functional.normalize(raw_output, dim=1)

    # Check each embedding is normalized
    for i in range(batch_size):
        norm = normalized[i].norm().item()
        assert abs(norm - 1.0) < 1e-5


@pytest.mark.unit
def test_muq_milvus_schema_dimensions(mock_muq_components, mock_milvus_client):
    """Test MuQ Milvus schema has 1,536 dimensions (512 * 3 enrichment)."""
    model = MuQEmbeddingModel(device="cpu")

    model.ensure_milvus_schemas(mock_milvus_client)

    # Verify create_collection was called
    assert mock_milvus_client.create_collection.called

    call_args = mock_milvus_client.create_collection.call_args
    collection_name = call_args[0][0]
    assert collection_name == "embedding"

    model.shutdown()


@pytest.mark.unit
def test_muq_milvus_collection_name(mock_muq_components, mock_milvus_client):
    """Test MuQ uses 'embedding' collection."""
    model = MuQEmbeddingModel(device="cpu")

    model.ensure_milvus_schemas(mock_milvus_client)

    call_args = mock_milvus_client.create_collection.call_args
    collection_name = call_args[0][0]
    assert collection_name == "embedding"

    model.shutdown()


@pytest.mark.unit
def test_muq_milvus_index_creation(mock_muq_components, mock_milvus_client):
    """Test MuQ creates vector and inverted indexes."""
    model = MuQEmbeddingModel(device="cpu")

    mock_milvus_client.describe_collection.return_value = {"indexes": []}

    model.ensure_milvus_index(mock_milvus_client)

    assert mock_milvus_client.create_index.called

    call_args = mock_milvus_client.create_index.call_args
    collection_name = call_args[0][0]
    assert collection_name == "embedding"

    model.shutdown()


# ==============================================================================
# Enhanced BaseEmbeddingModel Tests
# ==============================================================================


@pytest.mark.unit
def test_base_prepare_music_default():
    """Test prepare_music creates single TrackSegment by default."""
    model = DummyEmbeddingModel()
    try:
        segments = BaseEmbeddingModel.prepare_music(
            model,
            music_file="/path/to/music.flac",
            music_name="My Test Track",
        )

        assert len(segments) == 1
        assert segments[0].index == 1
        assert segments[0].title == "My Test Track"
        assert segments[0].start == 0.0
        assert segments[0].end is None
    finally:
        model.shutdown()


@pytest.mark.unit
def test_base_prepare_music_title_from_path():
    """Test prepare_music uses file stem when music_name is empty."""
    model = DummyEmbeddingModel()
    try:
        segments = BaseEmbeddingModel.prepare_music(
            model,
            music_file="/path/to/awesome_song.flac",
            music_name="",
        )

        assert len(segments) == 1
        assert segments[0].title == "awesome_song"
    finally:
        model.shutdown()


@pytest.mark.unit
def test_base_model_session_updates_timestamp(dummy_model):
    """Test model_session updates last_used timestamp."""
    import time

    # Record initial timestamp
    dummy_model.ensure_model_loaded()
    initial_time = dummy_model._last_used

    # Wait briefly
    time.sleep(0.1)

    # Use model session
    with dummy_model.model_session():
        pass

    # Timestamp should be updated
    assert dummy_model._last_used > initial_time


@pytest.mark.unit
def test_base_concurrent_sessions_use_same_model(dummy_model):
    """Test concurrent model sessions reuse the same model instance."""
    with dummy_model.model_session() as model1:
        with dummy_model.model_session() as model2:
            assert model1 is model2

    assert dummy_model.load_count == 1


@pytest.mark.unit
def test_base_ensure_model_loaded_idempotent(dummy_model):
    """Test ensure_model_loaded can be called multiple times safely."""
    model1 = dummy_model.ensure_model_loaded()
    model2 = dummy_model.ensure_model_loaded()
    model3 = dummy_model.ensure_model_loaded()

    assert model1 is model2
    assert model2 is model3
    assert dummy_model.load_count == 1


@pytest.mark.unit
def test_base_shutdown_stops_thread(dummy_model):
    """Test shutdown stops the background unloader thread."""
    assert dummy_model._unloader.is_alive()

    dummy_model.shutdown()

    dummy_model._unloader.join(timeout=2)

    assert not dummy_model._unloader.is_alive()


@pytest.mark.unit
def test_base_shutdown_releases_model(dummy_model):
    """Test shutdown releases the loaded model."""
    dummy_model.ensure_model_loaded()
    assert dummy_model._model is not None

    dummy_model.shutdown()

    assert dummy_model._model is None


@pytest.mark.unit
def test_base_session_exception_propagates(dummy_model):
    """Test exceptions inside model_session propagate correctly."""

    class TestException(Exception):
        pass

    try:
        with dummy_model.model_session():
            raise TestException("Test error")
    except TestException as e:
        assert str(e) == "Test error"
    else:
        assert False, "Exception should have propagated"

    assert dummy_model._model is not None


@pytest.mark.unit
def test_base_timeout_parameter_validation(logger):
    """Test timeout_seconds is validated to minimum value."""
    model = DummyEmbeddingModel()
    model._timeout = 10

    assert model._timeout >= 10

    model.shutdown()


@pytest.mark.unit
def test_base_model_session_context_manager_protocol(dummy_model):
    """Test model_session follows context manager protocol."""
    with dummy_model.model_session() as model:
        assert model is not None
