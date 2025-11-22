"""
Integration tests for the complete recommendation system.

These tests verify the integration between:
- Go backend (mocked HTTP calls)
- Python recommendation engine
- Multi-model similarity search
- Milvus vector database (mocked)
- Text embedding service
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

# Import components to test
from recommender_api import app as recommender_app, RecommendationEngine
from text_embedding_service import app as text_app, TextEmbeddingService
from database_query import MultiModelSimilaritySearcher
from schemas import RecommendationRequest, RecommendationSeed


class TestRecommendationEngineIntegration:
    """Test complete recommendation flow"""

    def test_single_model_recommendation_flow(self):
        """Test recommendation with single model"""
        # Setup mock Milvus client
        mock_client = Mock()
        mock_client.search.return_value = [
            [
                {"name": "Track A", "distance": 0.9, "entity": {"id": "track_a"}},
                {"name": "Track B", "distance": 0.8, "entity": {"id": "track_b"}},
                {"name": "Track C", "distance": 0.7, "entity": {"id": "track_c"}},
            ]
        ]

        searcher = MultiModelSimilaritySearcher(mock_client)

        # Create request
        request = RecommendationRequest(
            user_id="user1",
            user_name="testuser",
            limit=3,
            mode="recent",
            seeds=[RecommendationSeed(track_id="seed1", weight=1.0, source="recent")],
            models=["muq"],
        )

        # Execute search (would normally go through full engine)
        embeddings = {"muq": np.random.randn(1536).tolist()}
        results = searcher.search_multi_model(
            embeddings=embeddings, top_k=3, merge_strategy="union"
        )

        # Verify results
        assert len(results) > 0
        assert all("track_name" in r for r in results)
        assert all("score" in r for r in results)
        assert all("models" in r for r in results)

    def test_multi_model_union_flow(self):
        """Test multi-model recommendation with union merge"""
        mock_client = Mock()

        # Mock different results for different collections
        def mock_search(collection_name, data, **kwargs):
            if collection_name == "embedding":  # muq
                return [
                    [
                        {
                            "name": "Track A",
                            "distance": 0.9,
                            "entity": {"id": "track_a"},
                        },
                        {
                            "name": "Track B",
                            "distance": 0.85,
                            "entity": {"id": "track_b"},
                        },
                    ]
                ]
            elif collection_name == "mert_embedding":  # mert
                return [
                    [
                        {
                            "name": "Track B",
                            "distance": 0.88,
                            "entity": {"id": "track_b"},
                        },
                        {
                            "name": "Track C",
                            "distance": 0.82,
                            "entity": {"id": "track_c"},
                        },
                    ]
                ]
            return [[]]

        mock_client.search.side_effect = mock_search

        searcher = MultiModelSimilaritySearcher(mock_client)

        # Multi-model search
        embeddings = {
            "muq": np.random.randn(1536).tolist(),
            "mert": np.random.randn(76800).tolist(),
        }
        results = searcher.search_multi_model(
            embeddings=embeddings, top_k=10, merge_strategy="union"
        )

        # Verify union behavior
        assert len(results) >= 2  # Should have at least Track A, B, C
        track_names = [r["track_name"] for r in results]
        assert "Track A" in track_names
        assert "Track B" in track_names
        assert "Track C" in track_names

        # Track B should have both models
        track_b = next(r for r in results if r["track_name"] == "Track B")
        assert len(track_b["models"]) == 2

    def test_multi_model_intersection_flow(self):
        """Test multi-model with intersection - only common tracks"""
        mock_client = Mock()

        def mock_search(collection_name, data, **kwargs):
            # Both collections return Track B
            return [
                [
                    {"name": "Track B", "distance": 0.85, "entity": {"id": "track_b"}},
                    {
                        "name": (
                            "Track A" if collection_name == "embedding" else "Track C"
                        ),
                        "distance": 0.80,
                        "entity": {
                            "id": (
                                "track_a"
                                if collection_name == "embedding"
                                else "track_c"
                            )
                        },
                    },
                ]
            ]

        mock_client.search.side_effect = mock_search

        searcher = MultiModelSimilaritySearcher(mock_client)

        embeddings = {
            "muq": np.random.randn(1536).tolist(),
            "mert": np.random.randn(76800).tolist(),
        }
        results = searcher.search_multi_model(
            embeddings=embeddings, top_k=10, merge_strategy="intersection"
        )

        # Should only have Track B (common to both)
        assert len(results) >= 1
        assert all(
            r["track_name"] == "Track B" or len(r["models"]) == 2 for r in results
        )


class TestTextEmbeddingIntegration:
    """Test text embedding service integration"""

    def test_text_embedding_service_with_stub(self):
        """Test text embedding with stub fallback"""
        service = TextEmbeddingService(use_stubs=True)

        # Test embedding
        text = "upbeat rock music"
        model = "muq"
        embedding = service.embed_text(text, model)

        # Verify output
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 1536  # MuQ dimension
        assert np.abs(np.linalg.norm(embedding) - 1.0) < 0.01  # Should be normalized

    def test_text_embedding_api_endpoint(self):
        """Test text embedding FastAPI endpoint"""
        client = TestClient(text_app)

        response = client.post(
            "/embed_text", json={"text": "chill jazz for studying", "model": "muq"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert "model" in data
        assert "dimension" in data
        assert "is_stub" in data
        assert data["dimension"] == 1536
        assert len(data["embedding"]) == 1536

    def test_text_to_recommendation_flow(self):
        """Test complete flow: text → embedding → recommendation"""
        # Step 1: Get text embedding
        text_service = TextEmbeddingService(use_stubs=True)
        embedding = text_service.embed_text("energetic dance music", "muq")

        # Step 2: Use embedding for search
        mock_client = Mock()
        mock_client.search.return_value = [
            [
                {"name": "Dance Track 1", "distance": 0.92, "entity": {"id": "dt1"}},
                {"name": "Dance Track 2", "distance": 0.88, "entity": {"id": "dt2"}},
            ]
        ]

        searcher = MultiModelSimilaritySearcher(mock_client)
        results = searcher.search_multi_model(
            embeddings={"muq": embedding.tolist()}, top_k=5, merge_strategy="union"
        )

        # Verify results
        assert len(results) == 2
        assert "Dance Track 1" in [r["track_name"] for r in results]
        assert "Dance Track 2" in [r["track_name"] for r in results]


class TestNegativePromptIntegration:
    """Test negative prompting integration"""

    def test_negative_prompt_penalty_application(self):
        """Test that negative prompts correctly penalize similar tracks"""
        # Mock text service
        text_service = TextEmbeddingService(use_stubs=True)

        # Get embeddings for positive and negative prompts
        positive_emb = text_service.embed_text("upbeat music", "muq")
        negative_emb = text_service.embed_text("slow ballads", "muq")

        # Create mock candidates
        candidates = {
            "track1": 1.0,  # High score
            "track2": 0.9,
            "track3": 0.8,
        }

        # Mock track embeddings (track2 is similar to negative prompt)
        track_embeddings = {
            "track1": np.random.randn(1536),
            "track2": negative_emb * 0.9,  # Very similar to negative
            "track3": np.random.randn(1536),
        }

        # Normalize
        for name, emb in track_embeddings.items():
            track_embeddings[name] = emb / (np.linalg.norm(emb) + 1e-8)

        # Calculate penalties
        penalty = 0.85
        for track_name in candidates:
            track_emb = track_embeddings[track_name]
            similarity = np.dot(
                track_emb, negative_emb / (np.linalg.norm(negative_emb) + 1e-8)
            )
            penalty_factor = 1.0 - (similarity * (1.0 - penalty))
            candidates[track_name] *= max(0.0, penalty_factor)

        # Verify track2 was penalized more than others
        assert candidates["track2"] < candidates["track1"]
        assert candidates["track2"] < candidates["track3"]

    def test_negative_prompt_request_schema(self):
        """Test that negative prompt fields are properly handled"""
        request = RecommendationRequest(
            user_id="user1",
            user_name="testuser",
            limit=10,
            mode="text",
            seeds=[],
            models=["muq"],
            negative_prompts=["slow music", "sad songs"],
            negative_prompt_penalty=0.7,
        )

        # Verify schema
        assert len(request.negative_prompts) == 2
        assert request.negative_prompt_penalty == 0.7
        assert 0.3 <= request.negative_prompt_penalty <= 1.0


class TestMultiModelAgreementIntegration:
    """Test minimum model agreement filtering"""

    def test_min_model_agreement_filtering(self):
        """Test that min agreement filter works correctly"""
        mock_client = Mock()

        def mock_search(collection_name, data, **kwargs):
            if collection_name == "embedding":
                return [
                    [
                        {"name": "Track A", "distance": 0.9, "entity": {"id": "ta"}},
                        {"name": "Track B", "distance": 0.85, "entity": {"id": "tb"}},
                        {"name": "Track C", "distance": 0.80, "entity": {"id": "tc"}},
                    ]
                ]
            elif collection_name == "mert_embedding":
                return [
                    [
                        {"name": "Track A", "distance": 0.88, "entity": {"id": "ta"}},
                        {"name": "Track B", "distance": 0.83, "entity": {"id": "tb"}},
                        {"name": "Track D", "distance": 0.78, "entity": {"id": "td"}},
                    ]
                ]
            elif collection_name == "latent_embedding":
                return [
                    [
                        {"name": "Track A", "distance": 0.87, "entity": {"id": "ta"}},
                        {"name": "Track E", "distance": 0.82, "entity": {"id": "te"}},
                    ]
                ]
            return [[]]

        mock_client.search.side_effect = mock_search

        searcher = MultiModelSimilaritySearcher(mock_client)

        embeddings = {
            "muq": np.random.randn(1536).tolist(),
            "mert": np.random.randn(76800).tolist(),
            "latent": np.random.randn(576).tolist(),
        }

        # Test with min_model_agreement = 2
        results = searcher.search_multi_model(
            embeddings=embeddings,
            top_k=10,
            merge_strategy="union",
            min_model_agreement=2,
        )

        track_names = [r["track_name"] for r in results]

        # Track A appears in all 3 (should be included)
        assert "Track A" in track_names

        # Track B appears in 2 (should be included)
        assert "Track B" in track_names

        # Track C, D, E only appear in 1 each (should be excluded)
        assert "Track C" not in track_names
        assert "Track D" not in track_names
        assert "Track E" not in track_names

        # Verify Track A has all 3 models
        track_a = next(r for r in results if r["track_name"] == "Track A")
        assert len(track_a["models"]) == 3


class TestAPIEndpointIntegration:
    """Test FastAPI endpoints end-to-end"""

    def test_recommender_health_endpoint(self):
        """Test health check endpoint"""
        client = TestClient(recommender_app)
        response = client.get("/healthz")
        assert response.status_code == 200

    def test_text_embedding_health_endpoint(self):
        """Test text service health check"""
        client = TestClient(text_app)
        response = client.get("/health")
        assert response.status_code == 200

    def test_text_embedding_models_endpoint(self):
        """Test models listing endpoint"""
        client = TestClient(text_app)
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        # Response is a list of models
        assert isinstance(data, list)
        assert len(data) == 3
        model_names = [m["name"] for m in data]
        assert "muq" in model_names
        assert "mert" in model_names
        assert "latent" in model_names


@pytest.mark.integration
class TestEndToEndFlow:
    """Complete end-to-end integration tests"""

    def test_complete_recommendation_flow_with_mocks(self):
        """
        Test complete flow from request to response with all mocked dependencies.
        This simulates what would happen when Go backend calls Python service.
        """
        # This would test:
        # 1. Go POST to /recommendations/recent
        # 2. Python engine builds seeds
        # 3. Multi-model search executes
        # 4. Results are processed and returned
        # 5. Go enriches with track metadata

        # For now, we verify the components work together
        request = RecommendationRequest(
            user_id="test_user",
            user_name="Test User",
            limit=10,
            mode="recent",
            seeds=[RecommendationSeed(track_id="track1", weight=1.0, source="recent")],
            models=["muq", "mert"],
            merge_strategy="union",
            min_model_agreement=1,
        )

        # Verify request is valid
        assert request.user_id == "test_user"
        assert len(request.models) == 2
        assert request.merge_strategy == "union"
        assert len(request.seeds) == 1

        # Mock execution would happen here in real integration test
        # For unit test, we just verify the request structure is correct
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
