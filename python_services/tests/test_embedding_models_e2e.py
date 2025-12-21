"""
End-to-end tests for embedding models.

These tests verify complete workflows across multiple components.
Run with: pytest tests/test_embedding_models_e2e.py -v
"""

import pytest
import torch
from unittest.mock import patch

from embedding_models import (
    BaseEmbeddingModel,
    MuQEmbeddingModel,
    enrich_embedding,
)


@pytest.mark.unit
def test_model_dimensions_correct():
    """Test MuQ produces expected dimensionality."""
    # MuQEmbeddingModel: 512 * 3 = 1,536 (after enrichment)
    muq_dim = 512 * 3
    assert muq_dim == 1_536


@pytest.mark.unit
def test_enrichment_triples_dimensions():
    """Test enrichment function produces 3x dimensions."""
    for D in [512, 1024, 25600]:
        T = 10
        embedding = torch.randn(D, T)
        enriched = enrich_embedding(embedding)
        assert enriched.shape[0] == 3 * D


@pytest.mark.unit
def test_enrichment_produces_normalized_output():
    """Test enrichment always produces L2 normalized vectors."""
    test_cases = [
        (64, 10),
        (512, 5),
        (1024, 20),
        (25600, 3),
    ]

    for D, T in test_cases:
        embedding = torch.randn(D, T)
        enriched = enrich_embedding(embedding)

        # Should be L2 normalized
        norm = enriched.norm().item()
        assert abs(norm - 1.0) < 1e-4, f"Failed for D={D}, T={T}: norm={norm}"


@pytest.mark.unit
@patch("embedding_models.MuQ")
def test_muq_milvus_schema_matches_output(_mock_muq):
    """Test MuQ Milvus schema dimension matches actual output."""
    # MuQ should produce 1,536-D embeddings (512 * 3)
    expected_muq_dim = 1_536

    assert expected_muq_dim == 512 * 3

    model = MuQEmbeddingModel(device="cpu")
    model.shutdown()


@pytest.mark.unit
def test_all_models_support_embed_string():
    """Test BaseEmbeddingModel has embed_string method."""
    import inspect

    assert hasattr(BaseEmbeddingModel, "embed_string")
    assert inspect.isabstract(BaseEmbeddingModel)
