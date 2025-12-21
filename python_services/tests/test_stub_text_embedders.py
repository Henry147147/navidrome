"""
Unit tests for stub text embedders
"""

import numpy as np
import pytest

from stub_text_embedders import (
    StubTextEmbedder,
    StubMuQTextEmbedder,
    StubQwen3TextEmbedder,
    get_stub_embedder,
)


class TestStubTextEmbedder:
    """Tests for base StubTextEmbedder class"""

    def test_init(self):
        """Test embedder initialization"""
        embedder = StubTextEmbedder(dimension=512, model_name="test")
        assert embedder.dimension == 512
        assert embedder.model_name == "test"

    def test_embed_text_returns_correct_shape(self):
        """Test that embedding has correct dimensionality"""
        embedder = StubTextEmbedder(dimension=512, model_name="test")
        embedding = embedder.embed_text("test query")
        assert embedding.shape == (512,)

    def test_embed_text_is_normalized(self):
        """Test that embeddings are L2-normalized"""
        embedder = StubTextEmbedder(dimension=512, model_name="test")
        embedding = embedder.embed_text("test query")
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_embed_text_is_deterministic(self):
        """Test that same text produces same embedding"""
        embedder = StubTextEmbedder(dimension=512, model_name="test")
        text = "upbeat rock music"

        embedding1 = embedder.embed_text(text)
        embedding2 = embedder.embed_text(text)

        assert np.allclose(embedding1, embedding2)

    def test_embed_text_different_for_different_texts(self):
        """Test that different texts produce different embeddings"""
        embedder = StubTextEmbedder(dimension=512, model_name="test")

        embedding1 = embedder.embed_text("rock music")
        embedding2 = embedder.embed_text("jazz music")

        # Should not be identical
        assert not np.allclose(embedding1, embedding2)

        # Should have different similarity
        similarity = np.dot(embedding1, embedding2)
        assert similarity < 1.0

    def test_embed_texts_batch(self):
        """Test batch embedding"""
        embedder = StubTextEmbedder(dimension=512, model_name="test")
        texts = ["rock", "jazz", "classical"]

        embeddings = embedder.embed_texts(texts)

        assert embeddings.shape == (3, 512)

        # Each embedding should be normalized
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_compute_similarity(self):
        """Test similarity computation"""
        embedder = StubTextEmbedder(dimension=512, model_name="test")

        # Create some dummy audio embeddings
        audio_embeddings = np.random.randn(10, 512).astype(np.float32)
        audio_embeddings = audio_embeddings / np.linalg.norm(
            audio_embeddings, axis=1, keepdims=True
        )

        similarities = embedder.compute_similarity("test query", audio_embeddings)

        assert similarities.shape == (10,)
        # Similarities should be in [-1, 1] range (cosine similarity)
        assert np.all(similarities >= -1.0)
        assert np.all(similarities <= 1.0)


class TestStubMuQTextEmbedder:
    """Tests for MuQ stub embedder"""

    def test_dimension(self):
        """Test that MuQ embedder has correct dimension"""
        embedder = StubMuQTextEmbedder()
        assert embedder.dimension == 1536
        assert embedder.model_name == "muq_stub"

    def test_embedding_shape(self):
        """Test embedding output shape"""
        embedder = StubMuQTextEmbedder()
        embedding = embedder.embed_text("test")
        assert embedding.shape == (1536,)


class TestStubQwen3TextEmbedder:
    """Tests for Qwen3 stub embedder"""

    def test_dimension(self):
        """Test that Qwen3 embedder has correct dimension"""
        embedder = StubQwen3TextEmbedder()
        assert embedder.dimension == 4096
        assert embedder.model_name == "qwen3_stub"

    def test_embedding_shape(self):
        """Test embedding output shape"""
        embedder = StubQwen3TextEmbedder()
        embedding = embedder.embed_text("test")
        assert embedding.shape == (4096,)


class TestGetStubEmbedder:
    """Tests for get_stub_embedder factory function"""

    def test_get_muq_embedder(self):
        """Test getting MuQ embedder"""
        embedder = get_stub_embedder("muq")
        assert isinstance(embedder, StubMuQTextEmbedder)
        assert embedder.dimension == 1536

    def test_get_qwen3_embedder(self):
        """Test getting Qwen3 embedder"""
        embedder = get_stub_embedder("qwen3")
        assert isinstance(embedder, StubQwen3TextEmbedder)
        assert embedder.dimension == 4096

    def test_invalid_model_raises_error(self):
        """Test that invalid model name raises ValueError"""
        with pytest.raises(ValueError, match="Unknown model"):
            get_stub_embedder("invalid_model")


class TestEmbeddingQuality:
    """Tests for embedding quality properties"""

    @pytest.mark.parametrize("model", ["muq", "qwen3"])
    def test_embedding_diversity(self, model):
        """Test that embeddings for different texts are diverse"""
        embedder = get_stub_embedder(model)

        texts = [
            "rock music",
            "jazz music",
            "classical music",
            "electronic music",
            "folk music",
        ]

        embeddings = embedder.embed_texts(texts)

        # Compute pairwise similarities
        similarities = embeddings @ embeddings.T

        # Off-diagonal elements should not all be very high
        # (indicating diversity in embeddings)
        off_diagonal = similarities[np.triu_indices(len(texts), k=1)]
        mean_similarity = np.mean(off_diagonal)

        # Mean pairwise similarity should be relatively low
        # (hash-based embeddings should be fairly random)
        assert mean_similarity < 0.5

    @pytest.mark.parametrize("model", ["muq", "qwen3"])
    def test_consistent_across_instances(self, model):
        """Test that different embedder instances produce same embeddings"""
        embedder1 = get_stub_embedder(model)
        embedder2 = get_stub_embedder(model)

        text = "upbeat rock music"

        embedding1 = embedder1.embed_text(text)
        embedding2 = embedder2.embed_text(text)

        assert np.allclose(embedding1, embedding2)

    def test_empty_string_handling(self):
        """Test handling of edge cases like empty strings"""
        embedder = StubTextEmbedder(dimension=512, model_name="test")

        # Empty string should still produce valid embedding
        embedding = embedder.embed_text("")
        assert embedding.shape == (512,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-6)

    def test_long_text_handling(self):
        """Test handling of very long text"""
        embedder = StubTextEmbedder(dimension=512, model_name="test")

        # Very long text
        long_text = " ".join(["word"] * 10000)
        embedding = embedder.embed_text(long_text)

        assert embedding.shape == (512,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-6)

    def test_special_characters_handling(self):
        """Test handling of special characters"""
        embedder = StubTextEmbedder(dimension=512, model_name="test")

        texts = [
            "rock & roll",
            "jazz/blues",
            "música latina",
            "日本の音楽",
            "emoji 🎵🎶",
        ]

        for text in texts:
            embedding = embedder.embed_text(text)
            assert embedding.shape == (512,)
            assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-6)
