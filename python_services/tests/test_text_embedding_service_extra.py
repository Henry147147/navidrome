import pytest

import text_embedding_service
from text_embedding_service import TextEmbeddingService


def test_get_embedder_falls_back_to_stub(monkeypatch):
    class BoomPipeline:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        text_embedding_service, "DescriptionEmbeddingPipeline", BoomPipeline
    )
    service = TextEmbeddingService(use_stubs=False, device="cpu")

    _embedder = service.get_embedder("qwen3")

    assert service.is_stub("qwen3") is True


def test_embed_text_oom_error_raises_runtime():
    service = TextEmbeddingService(use_stubs=True, device="cpu")

    class OOMEmbedder:
        def embed_text(self, text):
            raise RuntimeError("CUDA out of memory")

    service.embedders["qwen3"] = OOMEmbedder()

    with pytest.raises(RuntimeError) as excinfo:
        service.embed_text("hello", "qwen3")

    assert "CUDA out of memory while embedding with qwen3" in str(excinfo.value)
