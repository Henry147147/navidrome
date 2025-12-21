"""
Stub implementations of text embedding models for development/testing.

These stubs generate deterministic embeddings based on text content,
allowing development and testing to proceed while real text-to-audio
projection models are being trained.

Once trained models are available, these stubs should be replaced with
real TextToAudioEmbedder instances loaded from checkpoint files.
"""

import hashlib
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class StubTextEmbedder:
    """
    Base stub that generates deterministic random embeddings.

    Uses MD5 hash of input text as seed to ensure:
    1. Same text always produces same embedding (reproducibility)
    2. Similar texts produce different embeddings (no semantic awareness)
    3. Embeddings are L2-normalized (unit vectors)
    """

    def __init__(self, dimension: int, model_name: str):
        self.dimension = dimension
        self.model_name = model_name
        logger.info(f"Initialized stub text embedder: {model_name} (dim={dimension})")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate deterministic embedding based on text hash.

        Args:
            text: Input text string

        Returns:
            L2-normalized embedding vector of shape (dimension,)
        """
        # Use hash of text as seed for reproducibility
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)

        # Generate random vector from normal distribution
        embedding = rng.randn(self.dimension).astype(np.float32)

        # L2 normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            # Handle edge case of zero vector
            embedding[0] = 1.0

        return embedding

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts in batch.

        Args:
            texts: List of text strings

        Returns:
            Array of shape (num_texts, dimension) with L2-normalized embeddings
        """
        return np.stack([self.embed_text(text) for text in texts])

    def compute_similarity(self, text: str, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between text and audio embeddings.

        Args:
            text: Query text
            embeddings: Audio embeddings of shape (N, dimension)

        Returns:
            Similarity scores of shape (N,)
        """
        text_emb = self.embed_text(text)

        # Normalize audio embeddings if not already
        audio_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (audio_norms + 1e-8)

        # Cosine similarity (dot product of normalized vectors)
        similarities = embeddings_norm @ text_emb

        return similarities


class StubMuQTextEmbedder(StubTextEmbedder):
    """
    Stub for MuQ text embeddings (1536D).

    Matches the dimensionality of MuQEmbeddingModel's enriched audio embeddings.
    """

    def __init__(self):
        super().__init__(dimension=1536, model_name="muq_stub")


class StubQwen3TextEmbedder(StubTextEmbedder):
    """
    Stub for Qwen3 text embeddings (4096D).
    """

    def __init__(self):
        super().__init__(dimension=4096, model_name="qwen3_stub")


def get_stub_embedder(model: str) -> StubTextEmbedder:
    """
    Factory function to get stub embedder by model name.

    Args:
        model: Model name ("muq" or "qwen3")

    Returns:
        Appropriate stub embedder instance

    Raises:
        ValueError: If model name is not recognized
    """
    embedders = {
        "muq": StubMuQTextEmbedder,
        "qwen3": StubQwen3TextEmbedder,
    }

    if model not in embedders:
        raise ValueError(
            f"Unknown model: {model}. Available models: {list(embedders.keys())}"
        )

    return embedders[model]()


# Example usage and testing
if __name__ == "__main__":
    # Test each stub embedder
    test_queries = [
        "upbeat rock music with guitar solos",
        "relaxing jazz piano",
        "electronic dance music",
        "acoustic folk guitar",
    ]

    for model_name in ["muq", "qwen3"]:
        print(f"\n{'=' * 60}")
        print(f"Testing {model_name.upper()} stub embedder")
        print(f"{'=' * 60}")

        embedder = get_stub_embedder(model_name)

        # Test single embedding
        query = test_queries[0]
        embedding = embedder.embed_text(query)
        print(f"\nQuery: '{query}'")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.6f}")
        print(f"First 5 dimensions: {embedding[:5]}")

        # Test reproducibility
        embedding2 = embedder.embed_text(query)
        print("\nReproducibility test:")
        print(
            f"Same query twice produces same embedding: "
            f"{np.allclose(embedding, embedding2)}"
        )

        # Test batch embedding
        batch_embeddings = embedder.embed_texts(test_queries)
        print(f"\nBatch embedding shape: {batch_embeddings.shape}")

        # Test similarity
        similarities = embedder.compute_similarity(query, batch_embeddings)
        print(f"\nSimilarities to all queries: {similarities}")
        print(f"Most similar query: '{test_queries[np.argmax(similarities)]}'")

        # Verify normalization
        norms = np.linalg.norm(batch_embeddings, axis=1)
        print(f"\nAll embeddings normalized: {np.allclose(norms, 1.0)}")
        print(f"Norms: {norms}")
