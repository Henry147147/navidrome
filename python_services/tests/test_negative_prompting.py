"""
Unit tests for negative prompting functionality
"""

import pytest
import numpy as np
from schemas import RecommendationRequest, RecommendationSeed


class TestNegativePromptSchemas:
    """Test schemas support negative prompting"""

    def test_negative_prompts_field(self):
        """Test that RecommendationRequest accepts negative prompts"""
        req = RecommendationRequest(
            user_id="test_user",
            user_name="Test User",
            mode="recent",
            negative_prompts=["sad music", "slow tempo"],
            negative_prompt_penalty=0.8,
        )

        assert req.negative_prompts == ["sad music", "slow tempo"]
        assert req.negative_prompt_penalty == 0.8

    def test_negative_prompt_penalty_validation(self):
        """Test that penalty must be within valid range"""
        # Too low
        with pytest.raises(ValueError):
            RecommendationRequest(
                user_id="test_user",
                user_name="Test User",
                mode="recent",
                negative_prompt_penalty=0.2,  # Below 0.3
            )

        # Too high
        with pytest.raises(ValueError):
            RecommendationRequest(
                user_id="test_user",
                user_name="Test User",
                mode="recent",
                negative_prompt_penalty=1.5,  # Above 1.0
            )

    def test_negative_embeddings_field(self):
        """Test that RecommendationRequest accepts pre-computed negative embeddings"""
        req = RecommendationRequest(
            user_id="test_user",
            user_name="Test User",
            mode="recent",
            negative_embeddings={
                "muq": [[0.1] * 1536, [0.2] * 1536],
                "qwen3": [[0.3] * 2560],
            },
        )

        assert "muq" in req.negative_embeddings
        assert len(req.negative_embeddings["muq"]) == 2
        assert len(req.negative_embeddings["qwen3"]) == 1


class MockRecommender:
    """Mock recommender for testing negative prompt logic"""

    def compute_negative_similarity(self, track_embedding, negative_embeddings):
        """
        Compute maximum similarity between track and negative prompts.
        Returns a value between 0 and 1.
        """
        if not negative_embeddings:
            return 0.0

        similarities = []
        for neg_emb in negative_embeddings:
            # Cosine similarity (dot product of normalized vectors)
            similarity = np.dot(track_embedding, neg_emb)
            similarities.append(similarity)

        return max(similarities)

    def apply_negative_penalty(self, score, negative_similarity, penalty):
        """
        Apply penalty to score based on negative similarity.
        penalty is between 0.3 and 1.0 (lower = stronger penalty)
        """
        if negative_similarity <= 0:
            return score

        # Exponential penalty: score * (penalty ^ negative_similarity)
        penalized_score = score * (penalty**negative_similarity)
        return penalized_score


class TestNegativePromptLogic:
    """Test negative prompting penalty calculation"""

    def test_compute_negative_similarity_no_negatives(self):
        """Test that no negatives returns 0 similarity"""
        recommender = MockRecommender()

        track_emb = np.random.randn(100)
        track_emb = track_emb / np.linalg.norm(track_emb)

        similarity = recommender.compute_negative_similarity(track_emb, [])

        assert similarity == 0.0

    def test_compute_negative_similarity_max(self):
        """Test that maximum similarity is returned"""
        recommender = MockRecommender()

        # Create track embedding
        track_emb = np.array([1.0, 0.0, 0.0])

        # Create negative embeddings with varying similarity
        neg1 = np.array([0.8, 0.6, 0.0])  # Lower similarity
        neg1 = neg1 / np.linalg.norm(neg1)

        neg2 = np.array([1.0, 0.0, 0.0])  # Perfect match

        negative_embeddings = [neg1, neg2]

        similarity = recommender.compute_negative_similarity(
            track_emb, negative_embeddings
        )

        # Should return the maximum (1.0 from neg2)
        assert abs(similarity - 1.0) < 0.01

    def test_apply_negative_penalty_no_similarity(self):
        """Test that no negative similarity preserves score"""
        recommender = MockRecommender()

        original_score = 0.9
        penalized = recommender.apply_negative_penalty(original_score, 0.0, 0.85)

        assert penalized == original_score

    def test_apply_negative_penalty_partial_similarity(self):
        """Test partial negative similarity applies moderate penalty"""
        recommender = MockRecommender()

        original_score = 0.9
        negative_similarity = 0.5
        penalty = 0.85

        penalized = recommender.apply_negative_penalty(
            original_score, negative_similarity, penalty
        )

        # Score should be reduced but not eliminated
        assert penalized < original_score
        assert penalized > 0.0

        # With penalty=0.85 and similarity=0.5:
        # penalized = 0.9 * (0.85^0.5) ≈ 0.9 * 0.922 ≈ 0.83
        expected = original_score * (penalty**negative_similarity)
        assert abs(penalized - expected) < 0.01

    def test_apply_negative_penalty_perfect_match(self):
        """Test perfect negative match applies maximum penalty"""
        recommender = MockRecommender()

        original_score = 0.9
        negative_similarity = 1.0
        penalty = 0.85

        penalized = recommender.apply_negative_penalty(
            original_score, negative_similarity, penalty
        )

        # Score should be reduced by penalty factor
        # penalized = 0.9 * 0.85 = 0.765
        expected = original_score * penalty
        assert abs(penalized - expected) < 0.01

    def test_penalty_strength_variation(self):
        """Test that different penalty values have expected effects"""
        recommender = MockRecommender()

        original_score = 1.0
        negative_similarity = 1.0

        # Stronger penalty (lower value)
        strong_penalty = recommender.apply_negative_penalty(
            original_score, negative_similarity, 0.5
        )

        # Weaker penalty (higher value)
        weak_penalty = recommender.apply_negative_penalty(
            original_score, negative_similarity, 0.9
        )

        # Stronger penalty should reduce score more
        assert strong_penalty < weak_penalty
        assert strong_penalty == 0.5
        assert weak_penalty == 0.9

    def test_multiple_tracks_ranking(self):
        """Test that negative penalties preserve relative ranking"""
        recommender = MockRecommender()

        # Three tracks with different similarities to positive query
        tracks = [
            ("Track A", 0.9),  # High match to positive query
            ("Track B", 0.7),  # Medium match
            ("Track C", 0.5),  # Low match
        ]

        # Track B has high similarity to negative prompt
        negative_similarities = {
            "Track A": 0.1,
            "Track B": 0.9,  # Should be heavily penalized
            "Track C": 0.2,
        }

        penalty = 0.7

        # Apply penalties
        penalized = []
        for name, score in tracks:
            neg_sim = negative_similarities[name]
            penalized_score = recommender.apply_negative_penalty(
                score, neg_sim, penalty
            )
            penalized.append((name, penalized_score))

        # Sort by penalized score
        penalized.sort(key=lambda x: x[1], reverse=True)

        # Track B should now rank lower despite initially being second
        # Track A: 0.9 * (0.7^0.1) ≈ 0.865
        # Track B: 0.7 * (0.7^0.9) ≈ 0.514
        # Track C: 0.5 * (0.7^0.2) ≈ 0.471

        assert penalized[0][0] == "Track A"  # Still highest
        assert penalized[1][0] == "Track B"  # Penalized but still above C
        assert penalized[2][0] == "Track C"  # Lowest

    def test_edge_case_zero_score(self):
        """Test that zero score remains zero"""
        recommender = MockRecommender()

        penalized = recommender.apply_negative_penalty(0.0, 0.8, 0.7)
        assert penalized == 0.0

    def test_edge_case_perfect_negative_strong_penalty(self):
        """Test perfect negative match with strong penalty"""
        recommender = MockRecommender()

        original_score = 0.8
        negative_similarity = 1.0
        penalty = 0.3  # Strongest allowed penalty

        penalized = recommender.apply_negative_penalty(
            original_score, negative_similarity, penalty
        )

        # Should be reduced to 0.8 * 0.3 = 0.24
        assert abs(penalized - 0.24) < 0.01


class TestNegativePromptIntegration:
    """Integration tests for negative prompting"""

    def test_empty_negative_prompts_no_effect(self):
        """Test that empty negative prompts have no effect"""
        req = RecommendationRequest(
            user_id="test_user",
            user_name="Test User",
            mode="recent",
            negative_prompts=[],
            seeds=[RecommendationSeed(track_id="123", source="recent")],
        )

        assert len(req.negative_prompts) == 0
        # Should process normally without errors

    def test_negative_prompts_without_embeddings(self):
        """Test that negative prompts can be specified without embeddings"""
        req = RecommendationRequest(
            user_id="test_user",
            user_name="Test User",
            mode="recent",
            negative_prompts=["sad music"],
            negative_embeddings=None,
            seeds=[RecommendationSeed(track_id="123", source="recent")],
        )

        assert len(req.negative_prompts) == 1
        assert req.negative_embeddings is None
        # Backend should compute embeddings from text

    def test_precomputed_negative_embeddings(self):
        """Test using pre-computed negative embeddings"""
        req = RecommendationRequest(
            user_id="test_user",
            user_name="Test User",
            mode="recent",
            negative_embeddings={"muq": [[0.1] * 1536]},
            seeds=[RecommendationSeed(track_id="123", source="recent")],
        )

        assert req.negative_embeddings is not None
        assert "muq" in req.negative_embeddings
        # Should use pre-computed embeddings directly
