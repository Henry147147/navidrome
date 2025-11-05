"""
Unit tests for MultiModelSimilaritySearcher
"""

import pytest
from database_query import MultiModelSimilaritySearcher


class MockMilvusClient:
    """Mock Milvus client for testing"""

    def __init__(self):
        self.search_results = {}

    def search(self, collection_name, data, **kwargs):
        """Mock search method"""
        if collection_name not in self.search_results:
            return [[]]

        results = self.search_results[collection_name]
        return [results]  # Milvus returns list of lists

    def set_mock_results(self, collection_name, results):
        """Set mock search results for a collection"""
        self.search_results[collection_name] = results


class TestMultiModelSimilaritySearcher:
    """Tests for MultiModelSimilaritySearcher"""

    def test_init(self):
        """Test searcher initialization"""
        client = MockMilvusClient()
        searcher = MultiModelSimilaritySearcher(client)

        assert searcher.client == client
        assert searcher.COLLECTION_MAP == {
            "muq": "embedding",
            "mert": "mert_embedding",
            "latent": "latent_embedding",
        }

    def test_union_merge_strategy(self):
        """Test union merge strategy"""
        client = MockMilvusClient()
        searcher = MultiModelSimilaritySearcher(client)

        # Setup mock results
        muq_results = [
            {"entity": {"name": "Song A"}, "distance": 0.9},
            {"entity": {"name": "Song B"}, "distance": 0.8},
            {"entity": {"name": "Song C"}, "distance": 0.7},
        ]

        mert_results = [
            {"entity": {"name": "Song B"}, "distance": 0.85},
            {"entity": {"name": "Song D"}, "distance": 0.75},
        ]

        client.set_mock_results("embedding", muq_results)
        client.set_mock_results("mert_embedding", mert_results)

        # Search with union strategy
        embeddings = {"muq": [0.1] * 1536, "mert": [0.2] * 76800}

        results = searcher.search_multi_model(
            embeddings=embeddings, top_k=10, merge_strategy="union"
        )

        # Union should include all unique tracks
        track_names = [r["track_name"] for r in results]
        assert "Song A" in track_names
        assert "Song B" in track_names
        assert "Song C" in track_names
        assert "Song D" in track_names

    def test_intersection_merge_strategy(self):
        """Test intersection merge strategy"""
        client = MockMilvusClient()
        searcher = MultiModelSimilaritySearcher(client)

        # Setup mock results with overlap
        muq_results = [
            {"entity": {"name": "Song A"}, "distance": 0.9},
            {"entity": {"name": "Song B"}, "distance": 0.8},
        ]

        mert_results = [
            {"entity": {"name": "Song B"}, "distance": 0.85},
            {"entity": {"name": "Song C"}, "distance": 0.75},
        ]

        client.set_mock_results("embedding", muq_results)
        client.set_mock_results("mert_embedding", mert_results)

        embeddings = {"muq": [0.1] * 1536, "mert": [0.2] * 76800}

        results = searcher.search_multi_model(
            embeddings=embeddings, top_k=10, merge_strategy="intersection"
        )

        # Only Song B appears in both
        assert len(results) == 1
        assert results[0]["track_name"] == "Song B"

    def test_priority_merge_strategy(self):
        """Test priority merge strategy"""
        client = MockMilvusClient()
        searcher = MultiModelSimilaritySearcher(client)

        # Setup mock results
        muq_results = [
            {"entity": {"name": "Song A"}, "distance": 0.9},
            {"entity": {"name": "Song B"}, "distance": 0.8},
        ]

        mert_results = [
            {"entity": {"name": "Song C"}, "distance": 0.95},
            {"entity": {"name": "Song D"}, "distance": 0.85},
        ]

        client.set_mock_results("embedding", muq_results)
        client.set_mock_results("mert_embedding", mert_results)

        embeddings = {"muq": [0.1] * 1536, "mert": [0.2] * 76800}

        # muq has priority 1 (higher), mert has priority 2 (lower)
        results = searcher.search_multi_model(
            embeddings=embeddings,
            top_k=10,
            merge_strategy="priority",
            model_priorities={"muq": 1, "mert": 2},
        )

        # Should prioritize muq results first
        track_names = [r["track_name"] for r in results]
        assert track_names[0] == "Song A"
        assert track_names[1] == "Song B"

    def test_min_model_agreement(self):
        """Test minimum model agreement filtering"""
        client = MockMilvusClient()
        searcher = MultiModelSimilaritySearcher(client)

        # Setup results where only Song B appears in multiple models
        muq_results = [
            {"entity": {"name": "Song A"}, "distance": 0.9},
            {"entity": {"name": "Song B"}, "distance": 0.8},
        ]

        mert_results = [
            {"entity": {"name": "Song B"}, "distance": 0.85},
            {"entity": {"name": "Song C"}, "distance": 0.75},
        ]

        latent_results = [
            {"entity": {"name": "Song D"}, "distance": 0.8},
        ]

        client.set_mock_results("embedding", muq_results)
        client.set_mock_results("mert_embedding", mert_results)
        client.set_mock_results("latent_embedding", latent_results)

        embeddings = {"muq": [0.1] * 1536, "mert": [0.2] * 76800, "latent": [0.3] * 576}

        # Require at least 2 models to agree
        results = searcher.search_multi_model(
            embeddings=embeddings,
            top_k=10,
            merge_strategy="union",
            min_model_agreement=2,
        )

        # Only Song B appears in 2+ models
        assert len(results) == 1
        assert results[0]["track_name"] == "Song B"

    def test_single_model_fallback(self):
        """Test that single model works normally"""
        client = MockMilvusClient()
        searcher = MultiModelSimilaritySearcher(client)

        muq_results = [
            {"entity": {"name": "Song A"}, "distance": 0.9},
            {"entity": {"name": "Song B"}, "distance": 0.8},
        ]

        client.set_mock_results("embedding", muq_results)

        embeddings = {"muq": [0.1] * 1536}

        results = searcher.search_multi_model(embeddings=embeddings, top_k=10)

        assert len(results) == 2
        assert results[0]["track_name"] == "Song A"

    def test_model_metadata_preserved(self):
        """Test that model metadata is preserved in results"""
        client = MockMilvusClient()
        searcher = MultiModelSimilaritySearcher(client)

        muq_results = [
            {"entity": {"name": "Song A"}, "distance": 0.9},
        ]

        mert_results = [
            {"entity": {"name": "Song A"}, "distance": 0.85},
        ]

        client.set_mock_results("embedding", muq_results)
        client.set_mock_results("mert_embedding", mert_results)

        embeddings = {"muq": [0.1] * 1536, "mert": [0.2] * 76800}

        results = searcher.search_multi_model(
            embeddings=embeddings, top_k=10, merge_strategy="union"
        )

        # Check that models metadata is included
        assert len(results) == 1
        assert "models" in results[0]
        assert set(results[0]["models"]) == {"muq", "mert"}

    def test_empty_results(self):
        """Test handling of empty search results"""
        client = MockMilvusClient()
        searcher = MultiModelSimilaritySearcher(client)

        # No mock results set - will return empty
        embeddings = {"muq": [0.1] * 1536}

        results = searcher.search_multi_model(embeddings=embeddings, top_k=10)

        assert len(results) == 0

    def test_invalid_merge_strategy(self):
        """Test error handling for invalid merge strategy"""
        client = MockMilvusClient()
        searcher = MultiModelSimilaritySearcher(client)

        embeddings = {"muq": [0.1] * 1536}

        with pytest.raises(ValueError, match="Unknown merge strategy"):
            searcher.search_multi_model(
                embeddings=embeddings, top_k=10, merge_strategy="invalid"
            )

    def test_score_normalization(self):
        """Test that scores are properly averaged across models"""
        client = MockMilvusClient()
        searcher = MultiModelSimilaritySearcher(client)

        # Same song with different scores in different models
        muq_results = [
            {"entity": {"name": "Song A"}, "distance": 1.0},
        ]

        mert_results = [
            {"entity": {"name": "Song A"}, "distance": 0.8},
        ]

        client.set_mock_results("embedding", muq_results)
        client.set_mock_results("mert_embedding", mert_results)

        embeddings = {"muq": [0.1] * 1536, "mert": [0.2] * 76800}

        results = searcher.search_multi_model(
            embeddings=embeddings, top_k=10, merge_strategy="union"
        )

        # Average score should be (1.0 + 0.8) / 2 = 0.9
        assert len(results) == 1
        assert abs(results[0]["score"] - 0.9) < 0.01
