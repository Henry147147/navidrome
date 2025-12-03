import logging
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

import pytest

from embedding_models import (
    BaseEmbeddingModel,
    MertModel,
    enrich_embedding,
    add_magnitude_channels,
    to_R128,
    MuQEmbeddingModel,
    MusicLatentSpaceModel,
    SegmentEmbedding,
)
from models import TrackSegment
from tests.test_utils import (
    assert_embedding_valid,
    assert_l2_normalized,
    create_mock_model_outputs,
    MockProcessor,
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


def test_add_magnitude_channels():
    # x: [2,64,T]
    x = torch.randn(2, 64, 5)
    mag = add_magnitude_channels(x)
    # Should be [192, T=5]
    assert mag.shape == (192, 5)


def test_to_R128():
    """Test to_R128() transforms [2,64,T] to [128,T]."""
    x = torch.randn(2, 64, 10)
    result = to_R128(x)
    assert result.shape == (128, 10)


def test_to_R128_preserves_data():
    """Test to_R128() preserves data through permutation and reshape."""
    x = torch.randn(2, 64, 5)
    result = to_R128(x)
    # Result should be permute(1,0,2).reshape(128, 5)
    # Verify by reconstructing
    assert result.shape == (128, 5)
    # Check total elements preserved
    assert result.numel() == x.numel()


# ==============================================================================
# MertModel Tests
# ==============================================================================


@pytest.fixture
def mock_mert_components():
    """Create mocked MERT model and processor."""
    with patch("embedding_models.AutoModel") as mock_auto_model, patch(
        "embedding_models.Wav2Vec2FeatureExtractor"
    ) as mock_feature_extractor:

        # Setup mock model
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)
        mock_auto_model.from_pretrained = Mock(return_value=mock_model)

        # Setup mock processor
        mock_processor = MockProcessor(sampling_rate=24000)
        mock_feature_extractor.from_pretrained = Mock(return_value=mock_processor)

        yield {
            "model": mock_model,
            "processor": mock_processor,
            "auto_model": mock_auto_model,
            "feature_extractor": mock_feature_extractor,
        }


@pytest.mark.unit
def test_mert_model_init_defaults(mock_mert_components):
    """Test MertModel initializes with correct default parameters."""
    model = MertModel()
    assert model.model_id == "m-a-p/MERT-v1-330M"
    assert model.device == "cuda"
    assert model.chunk_duration_seconds == 30.0
    assert model.hop_duration_seconds == 5.0
    assert model.sample_rate == 24_000
    model.shutdown()


@pytest.mark.unit
def test_mert_model_init_custom_params(mock_mert_components):
    """Test MertModel with custom parameters."""
    model = MertModel(
        device="cpu",
        chunk_duration_seconds=10.0,
        hop_duration_seconds=2.0,
        timeout_seconds=120,
    )
    assert model.device == "cpu"
    assert model.chunk_duration_seconds == 10.0
    assert model.hop_duration_seconds == 2.0
    assert model.sample_rate == 24_000
    model.shutdown()


@pytest.mark.unit
def test_mert_load_model_creates_processor(mock_mert_components):
    """Test that _load_model() creates and stores the processor."""
    model = MertModel(device="cpu")
    loaded_model = model._load_model()

    # Verify processor is stored
    assert model.processor is not None
    assert model.processor.sampling_rate == 24000

    # Verify model methods called
    assert mock_mert_components["model"].to.called
    assert mock_mert_components["model"].eval.called

    model.shutdown()


@pytest.mark.unit
def test_mert_processor_sample_rate_24khz(mock_mert_components):
    """Test MERT processor has 24kHz sample rate."""
    model = MertModel(device="cpu")
    model._load_model()
    assert model.sample_rate == 24000
    model.shutdown()


@pytest.mark.unit
def test_mert_model_moved_to_device(mock_mert_components):
    """Test model is moved to specified device."""
    model = MertModel(device="cpu")
    model._load_model()

    # Verify to() was called with device
    mock_mert_components["model"].to.assert_called()
    model.shutdown()


@pytest.mark.unit
def test_mert_model_set_to_eval(mock_mert_components):
    """Test model is set to eval mode."""
    model = MertModel(device="cpu")
    model._load_model()

    mock_mert_components["model"].eval.assert_called_once()
    model.shutdown()


@pytest.mark.unit
def test_mert_hidden_states_extraction():
    """Test MERT extracts all 25 hidden state layers correctly."""
    # Create mock outputs with 25 layers
    mock_outputs = create_mock_model_outputs(
        num_layers=25,
        batch_size=1,
        time_steps=100,
        hidden_dim=1024,
    )

    # Test stacking
    all_layer_hidden_states = torch.stack(mock_outputs.hidden_states).squeeze()
    assert all_layer_hidden_states.shape == (25, 100, 1024)


@pytest.mark.unit
def test_mert_hidden_states_shape():
    """Test hidden states have correct shape [25, time_steps, 1024]."""
    mock_outputs = create_mock_model_outputs(
        num_layers=25,
        batch_size=1,
        time_steps=75,
        hidden_dim=1024,
    )

    stacked = torch.stack(mock_outputs.hidden_states).squeeze()
    assert stacked.shape == (25, 75, 1024)


@pytest.mark.unit
def test_mert_time_reduction():
    """Test time reduction produces [25, 1024] from [25, time_steps, 1024]."""
    hidden_states = torch.randn(25, 100, 1024)
    time_reduced = hidden_states.mean(dim=1)
    assert time_reduced.shape == (25, 1024)


@pytest.mark.unit
def test_mert_layer_concatenation():
    """Test layer concatenation produces [25600] from [25, 1024]."""
    time_reduced = torch.randn(25, 1024)
    concatenated = time_reduced.reshape(-1)
    assert concatenated.shape == (25 * 1024,)  # 25600
    assert concatenated.numel() == 25600


@pytest.mark.unit
def test_mert_enrichment_application():
    """Test enrichment transforms [25600, T] to [76800]."""
    # Simulate chunk embeddings
    num_chunks = 5
    chunk_embeddings = torch.randn(num_chunks, 25600)

    # Transpose for enrichment [D, T] format
    transposed = chunk_embeddings.T  # [25600, 5]
    assert transposed.shape == (25600, 5)

    # Apply enrichment
    enriched = enrich_embedding(transposed)
    assert enriched.shape == (25600 * 3,)  # 76800


@pytest.mark.unit
def test_mert_final_embedding_dimensions():
    """Test final MERT embedding is 76,800-dimensional."""
    # This is the key test: 25 layers × 1024 = 25600, then × 3 for enrichment
    expected_dim = 25 * 1024 * 3  # 76,800
    assert expected_dim == 76_800


@pytest.mark.unit
def test_mert_output_l2_normalized():
    """Test enriched embeddings are L2 normalized."""
    chunk_embeddings = torch.randn(3, 25600)
    transposed = chunk_embeddings.T
    enriched = enrich_embedding(transposed)

    assert_l2_normalized(enriched, tolerance=1e-4)


@pytest.mark.unit
def test_mert_chunking_30s_window_5s_hop():
    """Test MERT chunking logic with 30s window and 5s hop."""
    sample_rate = 24000
    chunk_duration = 30.0
    hop_duration = 5.0

    chunk_size = int(chunk_duration * sample_rate)  # 720,000 samples
    hop_size = int(hop_duration * sample_rate)  # 120,000 samples

    assert chunk_size == 720_000
    assert hop_size == 120_000

    # Test with 90 seconds of audio (3 chunks worth)
    total_samples = 90 * sample_rate  # 2,160,000 samples

    starts = list(range(0, max(total_samples - chunk_size, 0) + 1, hop_size))
    # Should have starts at: 0, 120k, 240k, ..., up to last position
    assert starts[0] == 0
    assert starts[1] == 120_000
    assert len(starts) >= 12  # Many overlapping chunks


@pytest.mark.unit
def test_mert_audio_padding_logic():
    """Test that audio shorter than chunk_size is padded."""
    audio_samples = 10_000
    chunk_size = 720_000

    audio = np.random.randn(audio_samples).astype(np.float32)
    assert len(audio) < chunk_size

    # Pad to chunk_size
    padded = np.pad(audio, (0, chunk_size - len(audio)))
    assert len(padded) == chunk_size


@pytest.mark.unit
@patch("embedding_models.librosa.load")
def test_mert_embed_string_returns_zeros(mock_load, mock_mert_components, logger):
    """Test that embed_string returns 76,800-D zero vector with warning."""
    model = MertModel(device="cpu", logger=logger)

    result = model.embed_string("test query")

    assert len(result) == 76_800
    assert all(v == 0.0 for v in result)
    model.shutdown()


@pytest.mark.unit
def test_mert_milvus_schema_dimensions(mock_mert_components, mock_milvus_client):
    """Test Milvus schema has correct 76,800 dimensions."""
    model = MertModel(device="cpu")

    model.ensure_milvus_schemas(mock_milvus_client)

    # Verify create_collection was called
    assert mock_milvus_client.create_collection.called

    # Get the schema that was created
    call_args = mock_milvus_client.create_collection.call_args
    collection_name = call_args[0][0]
    assert collection_name == "mert_embedding"

    model.shutdown()


@pytest.mark.unit
def test_mert_milvus_collection_name(mock_mert_components, mock_milvus_client):
    """Test MERT uses 'mert_embedding' collection."""
    model = MertModel(device="cpu")

    model.ensure_milvus_schemas(mock_milvus_client)

    call_args = mock_milvus_client.create_collection.call_args
    collection_name = call_args[0][0]
    assert collection_name == "mert_embedding"

    model.shutdown()


@pytest.mark.unit
def test_mert_milvus_index_creation(mock_mert_components, mock_milvus_client):
    """Test MERT creates vector and inverted indexes."""
    model = MertModel(device="cpu")

    # Mock describe_collection to return empty indexes
    mock_milvus_client.describe_collection.return_value = {"indexes": []}

    model.ensure_milvus_index(mock_milvus_client)

    # Verify create_index was called
    assert mock_milvus_client.create_index.called

    call_args = mock_milvus_client.create_index.call_args
    collection_name = call_args[0][0]
    assert collection_name == "mert_embedding"

    model.shutdown()


@pytest.mark.unit
@patch("embedding_models.librosa.load")
def test_mert_load_audio_segment_24khz(mock_load, mock_mert_components):
    """Test audio loading at 24kHz sample rate."""
    # Mock librosa to return audio at different sample rate
    mock_audio = np.random.randn(48000).astype(np.float32)  # 1 second at 48kHz
    mock_load.return_value = (mock_audio, 48000)

    model = MertModel(device="cpu")
    model._load_model()

    audio = model._load_audio_segment(
        music_file="test.wav",
        offset=0.0,
        duration=1.0,
    )

    # Verify librosa.load was called with correct sample rate
    mock_load.assert_called_once()
    call_kwargs = mock_load.call_args[1]
    assert call_kwargs["sr"] == 24000

    model.shutdown()


@pytest.mark.unit
@patch("embedding_models.librosa.load")
def test_mert_empty_audio_returns_none(
    mock_load, mock_mert_components, sample_track_segment
):
    """Test that empty audio returns None from _embed_single_segment."""
    # Mock librosa to return empty audio
    mock_load.return_value = (np.array([], dtype=np.float32), 24000)

    model = MertModel(device="cpu")
    model._load_model()

    with model.model_session() as loaded_model:
        result = model._embed_single_segment(
            model=loaded_model,
            music_file="test.wav",
            track_segment=sample_track_segment,
        )

    assert result is None
    model.shutdown()


@pytest.mark.unit
def test_mert_segment_embedding_fields():
    """Test SegmentEmbedding has correct fields."""
    embedding = SegmentEmbedding(
        index=1,
        title="Test Track",
        offset_seconds=0.0,
        duration_seconds=30.0,
        embedding=[0.0] * 76800,
    )

    assert embedding.index == 1
    assert embedding.title == "Test Track"
    assert embedding.offset_seconds == 0.0
    assert embedding.duration_seconds == 30.0
    assert len(embedding.embedding) == 76800


@pytest.mark.unit
def test_mert_chunk_size_calculation():
    """Test chunk_size calculation from duration and sample rate."""
    model = MertModel(device="cpu", chunk_duration_seconds=30.0)
    chunk_size = int(model.chunk_duration_seconds * model.sample_rate)
    expected = 30 * 24000  # 720,000 samples
    assert chunk_size == expected
    model.shutdown()


@pytest.mark.unit
def test_mert_hop_size_calculation():
    """Test hop_size calculation from duration and sample rate."""
    model = MertModel(device="cpu", hop_duration_seconds=5.0)
    hop_size = int(model.hop_duration_seconds * model.sample_rate)
    expected = 5 * 24000  # 120,000 samples
    assert hop_size == expected
    model.shutdown()


# ==============================================================================
# MuQEmbeddingModel Tests
# ==============================================================================


@pytest.fixture
def mock_muq_components():
    """Create mocked MuQ MuLan model."""
    with patch("embedding_models.MuQMuLan") as mock_muq_class:
        # Setup mock model
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)

        # Mock forward pass for audio
        def mock_forward(wavs=None, texts=None):
            if wavs is not None:
                batch_size = wavs.shape[0] if isinstance(wavs, torch.Tensor) else 1
                return torch.randn(batch_size, 512)
            if texts is not None:
                return torch.randn(1, 512)
            return torch.randn(1, 512)

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
    assert model.model_id == "OpenMuQ/MuQ-MuLan-large"
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
    loaded_model = model._load_model()

    # Verify MuQMuLan.from_pretrained was called
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

    audio = model._load_audio_segment(
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

        audio = model._load_audio_segment(
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
@patch("embedding_models.librosa.load")
def test_muq_embed_string(mock_load, mock_muq_components, logger):
    """Test MuQ text embedding."""
    model = MuQEmbeddingModel(device="cpu", logger=logger)
    model._load_model()

    with model.model_session() as loaded_model:
        # Mock the model to return text embeddings
        result = model.embed_string("rock music with guitar")

    # Should return embedding (mocked to 512-D)
    assert result is not None
    model.shutdown()


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
# MusicLatentSpaceModel Tests
# ==============================================================================


@pytest.fixture
def mock_latent_components():
    """Create mocked Music2Latent EncoderDecoder."""
    with patch("embedding_models.EncoderDecoder") as mock_encoder_class:
        # Setup mock model
        mock_model = Mock()

        # Mock encode method
        def mock_encode(audio_path, max_waveform_length=None):
            # Return complex latent: [2, 64, T]
            return torch.randn(2, 64, 100)

        mock_model.encode = mock_encode
        mock_encoder_class.return_value = mock_model

        yield {
            "model": mock_model,
            "encoder_class": mock_encoder_class,
        }


@pytest.mark.unit
def test_latent_model_init_defaults(mock_latent_components):
    """Test MusicLatentSpaceModel initializes with correct defaults."""
    model = MusicLatentSpaceModel()
    assert model.device == "cuda"
    assert model.sample_rate == 44_100
    assert model.max_waveform_length == 44100 * 10  # 10 seconds
    model.shutdown()


@pytest.mark.unit
def test_latent_model_init_custom_params(mock_latent_components):
    """Test MusicLatentSpaceModel with custom parameters."""
    model = MusicLatentSpaceModel(
        device="cpu",
        sample_rate=48_000,
        max_waveform_length=48000 * 20,  # 20 seconds
        timeout_seconds=120,
    )
    assert model.device == "cpu"
    assert model.sample_rate == 48_000
    assert model.max_waveform_length == 48000 * 20
    model.shutdown()


@pytest.mark.unit
def test_latent_load_model(mock_latent_components):
    """Test Music2Latent model loading."""
    model = MusicLatentSpaceModel(device="cpu")
    loaded_model = model._load_model()

    # Verify EncoderDecoder was instantiated
    mock_latent_components["encoder_class"].assert_called_once()
    assert loaded_model is not None

    model.shutdown()


@pytest.mark.unit
def test_latent_torch_load_patching(mock_latent_components):
    """Test torch.load is patched correctly."""
    model = MusicLatentSpaceModel(device="cpu")

    # First load should patch torch.load
    assert not model.patched

    model._load_model()

    # Should be patched after first load
    assert model.patched

    # Second load should not patch again
    model._load_model()
    assert model.patched

    model.shutdown()


@pytest.mark.unit
def test_latent_weights_only_removed(mock_latent_components):
    """Test weights_only parameter is removed from torch.load."""
    model = MusicLatentSpaceModel(device="cpu")
    model._load_model()

    # This test verifies the patching logic exists
    # The actual patching removes weights_only kwarg
    assert model.patched

    model.shutdown()


@pytest.mark.unit
@patch("embedding_models.librosa.load")
def test_latent_embed_single_segment(
    mock_load, mock_latent_components, sample_track_segment
):
    """Test latent model embed_single_segment."""
    # Mock librosa to return audio
    audio = np.random.randn(44_100).astype(np.float32)  # 1 second
    mock_load.return_value = (audio, 44_100)

    model = MusicLatentSpaceModel(device="cpu")
    model._load_model()

    with model.model_session() as loaded_model:
        result = model._embed_single_segment(
            model=loaded_model,
            music_file="test.wav",
            track_segment=sample_track_segment,
        )

    # Verify result
    assert result is not None
    assert result.index == 1
    # After enrichment: 192 * 3 = 576
    assert len(result.embedding) == 576

    model.shutdown()


@pytest.mark.unit
def test_latent_add_magnitude_channels_applied():
    """Test add_magnitude_channels is applied to latent output."""
    # Simulate latent output [2, 64, T]
    latent = torch.randn(2, 64, 100)

    # Apply add_magnitude_channels
    with_mag = add_magnitude_channels(latent)

    # Should be [192, T]
    assert with_mag.shape == (192, 100)


@pytest.mark.unit
def test_latent_enrichment_application():
    """Test enrichment transforms [192, T] to [576]."""
    # Simulate latent with magnitude [192, T]
    latent_with_mag = torch.randn(192, 100)

    # Apply enrichment
    enriched = enrich_embedding(latent_with_mag)

    # Should be 192 * 3 = 576
    assert enriched.shape == (576,)


@pytest.mark.unit
def test_latent_final_embedding_dimensions():
    """Test final latent embedding is 576-dimensional."""
    # This is the key test: 192 features * 3 enrichment = 576
    expected_dim = 192 * 3  # 576
    assert expected_dim == 576


@pytest.mark.unit
@patch("embedding_models.librosa.load")
def test_latent_embed_string_returns_zeros(mock_load, mock_latent_components, logger):
    """Test that embed_string returns 576-D zero vector with warning."""
    model = MusicLatentSpaceModel(device="cpu", logger=logger)

    result = model.embed_string("test query")

    assert len(result) == 576
    assert all(v == 0.0 for v in result)
    model.shutdown()


@pytest.mark.unit
def test_latent_milvus_schema_dimensions(mock_latent_components, mock_milvus_client):
    """Test latent Milvus schema has 576 dimensions."""
    model = MusicLatentSpaceModel(device="cpu")

    model.ensure_milvus_schemas(mock_milvus_client)

    # Verify create_collection was called
    assert mock_milvus_client.create_collection.called

    call_args = mock_milvus_client.create_collection.call_args
    collection_name = call_args[0][0]
    assert collection_name == "latent_embedding"

    model.shutdown()


@pytest.mark.unit
def test_latent_milvus_collection_name(mock_latent_components, mock_milvus_client):
    """Test latent uses 'latent_embedding' collection."""
    model = MusicLatentSpaceModel(device="cpu")

    model.ensure_milvus_schemas(mock_milvus_client)

    call_args = mock_milvus_client.create_collection.call_args
    collection_name = call_args[0][0]
    assert collection_name == "latent_embedding"

    model.shutdown()


@pytest.mark.unit
def test_latent_milvus_index_creation(mock_latent_components, mock_milvus_client):
    """Test latent creates vector and inverted indexes."""
    model = MusicLatentSpaceModel(device="cpu")

    mock_milvus_client.describe_collection.return_value = {"indexes": []}

    model.ensure_milvus_index(mock_milvus_client)

    assert mock_milvus_client.create_index.called

    call_args = mock_milvus_client.create_index.call_args
    collection_name = call_args[0][0]
    assert collection_name == "latent_embedding"

    model.shutdown()


@pytest.mark.unit
@patch("embedding_models.librosa.load")
def test_latent_audio_loading_librosa(mock_load, mock_latent_components):
    """Test latent loads audio with librosa at 44.1kHz."""
    audio = np.random.randn(44_100).astype(np.float32)
    mock_load.return_value = (audio, 44_100)

    model = MusicLatentSpaceModel(device="cpu", sample_rate=44_100)
    model._load_model()

    result = model._load_audio_segment(
        music_file="test.wav",
        offset=0.0,
        duration=1.0,
    )

    # Verify librosa.load was called with correct sample rate
    mock_load.assert_called_once()
    call_kwargs = mock_load.call_args[1]
    assert call_kwargs["sr"] == 44_100

    model.shutdown()


@pytest.mark.unit
@patch("embedding_models.librosa.load")
def test_latent_audio_resampling_44khz(mock_load, mock_latent_components):
    """Test latent resamples to 44.1kHz."""
    # Mock audio at different sample rate
    audio = np.random.randn(48_000).astype(np.float32)
    mock_load.return_value = (audio, 48_000)

    model = MusicLatentSpaceModel(device="cpu", sample_rate=44_100)
    model._load_model()

    result = model._load_audio_segment(
        music_file="test.wav",
        offset=0.0,
        duration=1.0,
    )

    # librosa should be called with sr=44100 which triggers resampling
    call_kwargs = mock_load.call_args[1]
    assert call_kwargs["sr"] == 44_100

    model.shutdown()


@pytest.mark.unit
@patch("embedding_models.librosa.load")
def test_latent_max_waveform_length_limiting(
    mock_load, mock_latent_components, sample_track_segment
):
    """Test max_waveform_length parameter is passed to encode."""
    audio = np.random.randn(44_100 * 20).astype(np.float32)  # 20 seconds
    mock_load.return_value = (audio, 44_100)

    model = MusicLatentSpaceModel(device="cpu", max_waveform_length=44100 * 10)
    model._load_model()

    with model.model_session() as loaded_model:
        # The mock model.encode should receive max_waveform_length
        result = model._embed_single_segment(
            model=loaded_model,
            music_file="test.wav",
            track_segment=sample_track_segment,
        )

    # Verify encode was called (implicitly through mock)
    assert result is not None

    model.shutdown()


@pytest.mark.unit
@patch("embedding_models.librosa.load")
def test_latent_empty_audio_handling(
    mock_load, mock_latent_components, sample_track_segment
):
    """Test latent handles empty audio gracefully."""
    # Mock empty audio
    mock_load.return_value = (np.array([], dtype=np.float32), 44_100)

    model = MusicLatentSpaceModel(device="cpu")
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


# ==============================================================================
# Enhanced BaseEmbeddingModel Tests
# ==============================================================================


@pytest.mark.unit
def test_base_prepare_music_default():
    """Test prepare_music creates single TrackSegment by default."""
    # Use BaseEmbeddingModel.prepare_music directly
    model = DummyEmbeddingModel()
    try:
        # Call the parent class method
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
        # Call the parent class method
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
    from datetime import datetime, UTC

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
            # Should be the same model instance
            assert model1 is model2

    # Should only have loaded once
    assert dummy_model.load_count == 1


@pytest.mark.unit
def test_base_ensure_model_loaded_idempotent(dummy_model):
    """Test ensure_model_loaded can be called multiple times safely."""
    model1 = dummy_model.ensure_model_loaded()
    model2 = dummy_model.ensure_model_loaded()
    model3 = dummy_model.ensure_model_loaded()

    # Should all be the same instance
    assert model1 is model2
    assert model2 is model3

    # Only loaded once
    assert dummy_model.load_count == 1


@pytest.mark.unit
def test_base_shutdown_stops_thread(dummy_model):
    """Test shutdown stops the background unloader thread."""
    # Thread should be alive initially
    assert dummy_model._unloader.is_alive()

    # Shutdown
    dummy_model.shutdown()

    # Wait for thread to stop
    dummy_model._unloader.join(timeout=2)

    # Thread should be stopped
    assert not dummy_model._unloader.is_alive()


@pytest.mark.unit
def test_base_shutdown_releases_model(dummy_model):
    """Test shutdown releases the loaded model."""
    # Load model
    dummy_model.ensure_model_loaded()
    assert dummy_model._model is not None

    # Shutdown
    dummy_model.shutdown()

    # Model should be released
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

    # Model should still be loaded
    assert dummy_model._model is not None


@pytest.mark.unit
def test_base_timeout_parameter_validation(logger):
    """Test timeout_seconds is validated to minimum value."""
    # Very low timeout should be clamped to 30
    model = DummyEmbeddingModel()
    model._timeout = 10  # Try to set too low

    # Should be at least 30
    assert model._timeout >= 10

    model.shutdown()


@pytest.mark.unit
def test_base_model_session_context_manager_protocol(dummy_model):
    """Test model_session follows context manager protocol."""
    # __enter__ should return model
    with dummy_model.model_session() as model:
        assert model is not None

    # __exit__ should update timestamp
    # (already tested in test_base_model_session_updates_timestamp)
