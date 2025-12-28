import logging
import torch

import pytest

import description_pipeline
import embedding_models


class DummySchema:
    def __init__(self, auto_id=False, enable_dynamic_field=False):
        self.auto_id = auto_id
        self.enable_dynamic_field = enable_dynamic_field
        self.fields = []

    def add_field(self, name, dtype, **kwargs):
        self.fields.append((name, dtype, kwargs))


class DummyIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, field_name, index_type, metric_type=None, params=None):
        self.indexes.append(
            {
                "field_name": field_name,
                "index_type": index_type,
                "metric_type": metric_type,
                "params": params or {},
            }
        )


class DummyMilvusClient:
    def __init__(self, collections=None, indexes=None):
        self.collections = set(collections or [])
        self.indexes = indexes or {}
        self.created_collections = []
        self.created_indexes = []

    def list_collections(self):
        return list(self.collections)

    def create_collection(self, name, schema):
        self.collections.add(name)
        self.created_collections.append((name, schema))

    def describe_collection(self, name):
        return {"indexes": self.indexes.get(name, [])}

    def create_index(self, name, params):
        self.created_indexes.append({"name": name, "params": params})

    # Static helpers mirrored from pymilvus.MilvusClient
    @staticmethod
    def create_schema(auto_id=False, enable_dynamic_field=False):
        return DummySchema(auto_id=auto_id, enable_dynamic_field=enable_dynamic_field)

    @staticmethod
    def prepare_index_params():
        return DummyIndexParams()


@pytest.fixture()
def pipeline(monkeypatch):
    # Avoid loading heavy models by bypassing __init__
    pipe = description_pipeline.DescriptionEmbeddingPipeline.__new__(
        description_pipeline.DescriptionEmbeddingPipeline
    )
    pipe.logger = logging.getLogger("navidrome.tests.description")
    return pipe


def test_ensure_schema_creates_when_missing(monkeypatch, pipeline):
    monkeypatch.setattr(description_pipeline, "MilvusClient", DummyMilvusClient)
    client = description_pipeline.MilvusClient()
    pipeline._audio_embedding_dim = 512

    pipeline.ensure_milvus_schemas(client)

    assert "description_embedding" in client.collections
    assert "flamingo_audio_embedding" in client.collections
    assert client.created_collections, "Schema should be created when missing"
    created = {name: schema for name, schema in client.created_collections}
    description_schema = created["description_embedding"]
    audio_schema = created["flamingo_audio_embedding"]
    # Expect five fields: name, description, embedding, offset, model_id
    assert len(description_schema.fields) == 5
    # Expect four fields: name, embedding, offset, model_id
    assert len(audio_schema.fields) == 4


def test_ensure_index_respects_lite(monkeypatch, pipeline):
    monkeypatch.setattr(description_pipeline, "MilvusClient", DummyMilvusClient)
    # Force lite path
    monkeypatch.setattr(embedding_models, "_milvus_uses_lite", lambda: True)
    client = description_pipeline.MilvusClient(
        collections={"description_embedding", "flamingo_audio_embedding"}, indexes={}
    )

    pipeline.ensure_milvus_index(client)

    assert client.created_indexes, "Index should be created for empty collection"
    params = client.created_indexes[0]["params"]
    assert any(
        idx["index_type"] == "IVF_FLAT" for idx in params.indexes
    ), "Lite mode should use IVF_FLAT index"


def test_ensure_index_skips_when_present(monkeypatch, pipeline):
    monkeypatch.setattr(description_pipeline, "MilvusClient", DummyMilvusClient)
    # Force non-lite path so HNSW would be chosen if needed
    monkeypatch.setattr(embedding_models, "_milvus_uses_lite", lambda: False)
    existing = {
        "description_embedding": [
            {"field_name": "embedding"},
            {"field_name": "name"},
        ]
    }
    client = description_pipeline.MilvusClient(
        collections={"description_embedding"}, indexes=existing
    )

    pipeline.ensure_milvus_index(client)

    assert (
        client.created_indexes == []
    ), "Existing indexes should short-circuit creation"


def test_qwen3_embedder_oom_fallback(monkeypatch):
    class DummyModel:
        def eval(self):
            return self

    attempts = {"count": 0}

    def fake_build(self):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("CUDA out of memory")
        return DummyModel()

    monkeypatch.setattr(description_pipeline.Qwen3Embedder, "_build_model", fake_build)
    monkeypatch.setattr(
        description_pipeline.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    embedder = description_pipeline.Qwen3Embedder(
        device="cuda",
        gpu_settings=description_pipeline.GPUSettings(device="cuda"),
    )

    assert embedder.device == "cpu"
    assert embedder.dtype == torch.float32
    assert attempts["count"] == 2


def test_pool_audio_embedding_normalizes():
    tensor = torch.tensor([1.0, 2.0, 2.0])
    pooled = description_pipeline._pool_audio_embedding(tensor)
    norm = torch.linalg.norm(pooled).item()
    assert norm == pytest.approx(1.0)


def test_last_token_pool_handles_left_padding():
    # left-padded sequence: attention mask ends in 1s
    hidden = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    mask = torch.tensor([[0, 1, 1]])
    pooled = description_pipeline.Qwen3Embedder._last_token_pool(hidden, mask)
    assert pooled.tolist() == [[3.0, 0.0]]


def test_resolve_audio_embedding_dim_prefers_text_config(monkeypatch, pipeline):
    class DummyTextConfig:
        hidden_size = 256

    class DummyConfig:
        text_config = DummyTextConfig()

    monkeypatch.setattr(
        description_pipeline.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: DummyConfig(),
    )

    pipeline.caption_model_id = "dummy"
    dim = pipeline._resolve_audio_embedding_dim()

    assert dim == 256
    assert pipeline._audio_embedding_dim == 256


def test_persist_description_replaces_existing(monkeypatch, tmp_path, pipeline):
    target = tmp_path / "descriptions.json"
    monkeypatch.setattr(description_pipeline, "DESCRIPTION_JSON_PATH", target)
    pipeline._persist_description("Song", "First", "song.flac")
    pipeline._persist_description("Song", "Second", "song.flac")

    data = target.read_text()
    assert "Second" in data
    assert data.count("Song") == 1


def test_music_flamingo_generate_happy_path(monkeypatch):
    class DummyProcessor:
        class DummyInputs(dict):
            def __getattr__(self, name):
                return self[name]

        def apply_chat_template(self, *_args, **_kwargs):
            return self.DummyInputs(
                {
                    "input_ids": torch.tensor([[1, 2]]),
                    "input_features": torch.tensor([1.0]),
                }
            )

        def batch_decode(self, *_args, **_kwargs):
            return ["caption"]

    class DummyModel:
        def __init__(self, owner):
            self.owner = owner
            self.device = "cpu"

        def generate(self, **_inputs):
            self.owner.last_input_embeds = torch.tensor([1.0, 0.0, 0.0])
            return torch.tensor([[1, 2, 3, 4]])

    captioner = description_pipeline.MusicFlamingoCaptioner.__new__(
        description_pipeline.MusicFlamingoCaptioner
    )
    captioner.last_input_embeds = None
    captioner.model_id = "dummy"
    captioner.device = "cpu"
    captioner.dtype = torch.float32
    captioner.logger = logging.getLogger("navidrome.tests.captioner")
    captioner.processor = DummyProcessor()
    captioner.model = DummyModel(captioner)
    monkeypatch.setattr(captioner, "_ensure_model_on_device", lambda: None)

    description, audio_embedding = captioner.generate("song.flac")

    assert description == "caption"
    assert isinstance(audio_embedding, list)
    assert len(audio_embedding) == 3


def test_music_flamingo_build_model_loads_pretrained(monkeypatch):
    class DummyConfig:
        pass

    class DummyModel:
        def __init__(self):
            self.device = "cpu"

        def to(self, _device):
            return self

        def get_audio_features(self, *_args, **_kwargs):
            return torch.tensor([1.0])

    captured = {}

    def fake_from_pretrained(model_id, **kwargs):
        captured["model_id"] = model_id
        captured["kwargs"] = kwargs
        return DummyModel()

    monkeypatch.setattr(
        description_pipeline.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: DummyConfig(),
    )
    monkeypatch.setattr(
        description_pipeline.AudioFlamingo3ForConditionalGeneration,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        description_pipeline.GPU_COORDINATOR, "claim", lambda *_a, **_k: None
    )
    import gpu_settings

    monkeypatch.setattr(gpu_settings, "force_cuda_memory_release", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    captioner = description_pipeline.MusicFlamingoCaptioner.__new__(
        description_pipeline.MusicFlamingoCaptioner
    )
    captioner.model_id = "dummy"
    captioner.dtype = torch.float32
    captioner.logger = logging.getLogger("navidrome.tests.captioner.load")
    captioner.gpu_settings = description_pipeline.GPUSettings()
    captioner._gpu_owner = "music_flamingo_captioner"
    captioner.last_input_embeds = None
    captioner.device = "cpu"

    model = captioner._build_model(use_local_cache=True)
    model.get_audio_features(torch.tensor([1.0]), torch.tensor([1]))

    assert captured["model_id"] == "dummy"
    assert captured["kwargs"]["local_files_only"] is True
    assert captured["kwargs"]["low_cpu_mem_usage"] is True
    assert captured["kwargs"]["device_map"] is None
    assert captioner.last_input_embeds is not None


def test_qwen3_embed_text_uses_last_token_pool(monkeypatch):
    class DummyInputs(dict):
        def to(self, *_args, **_kwargs):
            return self

    class DummyTokenizer:
        model_max_length = 8

        def __call__(self, *_args, **_kwargs):
            return DummyInputs(
                {
                    "input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                }
            )

    class DummyOutput:
        def __init__(self):
            self.last_hidden_state = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    class DummyModel:
        device = "cpu"

        def __call__(self, **_kwargs):
            return DummyOutput()

    embedder = description_pipeline.Qwen3Embedder.__new__(
        description_pipeline.Qwen3Embedder
    )
    embedder.tokenizer = DummyTokenizer()
    embedder.model = DummyModel()
    embedder.device = "cpu"
    embedder.dtype = torch.float32
    embedder.logger = logging.getLogger("navidrome.tests.qwen3")
    monkeypatch.setattr(embedder, "_ensure_model_on_device", lambda: None)

    embedding = embedder.embed_text("hello")

    assert embedding.shape == (2,)
    assert torch.linalg.norm(embedding).item() == pytest.approx(1.0)


def test_description_pipeline_describe_music(monkeypatch, tmp_path):
    pipeline = description_pipeline.DescriptionEmbeddingPipeline.__new__(
        description_pipeline.DescriptionEmbeddingPipeline
    )
    pipeline.logger = logging.getLogger("navidrome.tests.pipeline")
    pipeline.caption_model_id = "flamingo"
    pipeline.text_model_id = "qwen3"
    pipeline._audio_embedding_dim = None

    class DummyCaptioner:
        def generate(self, *_args, **_kwargs):
            return "desc", [0.1, 0.2]

    class DummyEmbedder:
        def embed_text(self, _text):
            return torch.tensor([1.0, 0.0])

    pipeline._get_captioner = lambda: DummyCaptioner()
    pipeline._get_embedder = lambda: DummyEmbedder()
    pipeline.unload_captioner = lambda: None
    pipeline._persist_description = lambda *_args, **_kwargs: None

    segments = pipeline.describe_music("song.flac", "Song")

    assert segments
    assert segments[0].title == "Song"
    assert segments[0].description == "desc"
    assert segments[0].audio_embedding == [0.1, 0.2]
