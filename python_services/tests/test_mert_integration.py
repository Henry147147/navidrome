"""
Integration tests for MuQ embedding model with real model loading.

These tests require downloading actual models from HuggingFace and are marked as slow.
Run with: pytest -m integration tests/test_mert_integration.py
Skip with: pytest -m "not integration" tests/
"""

import pytest

from embedding_models import MuQEmbeddingModel


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_models
def test_muq_load_real_model():
    """Test loading actual MuQ model."""
    model = MuQEmbeddingModel(device="cpu", timeout_seconds=600)

    try:
        with model.model_session() as loaded_model:
            assert loaded_model is not None
    finally:
        model.shutdown()
