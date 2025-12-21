"""
Integration tests for MuQ embedding model with real model loading.

These tests require downloading actual models from HuggingFace and are marked as slow.
Run with: pytest -m integration tests/test_mert_integration.py
Skip with: pytest -m "not integration" tests/
"""

import pytest

from embedding_models import MuQEmbeddingModel
from tests.test_utils import assert_embedding_valid


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
