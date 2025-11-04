import logging
from typing import Dict, Iterable, Sequence

import pytest

from recommender_api import RecommendationEngine
from schemas import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationSeed,
)
from track_name_resolver import TrackNameResolver


class RecordingSimilaritySearcher:
    def __init__(self, *, hit_map: Dict[str, Sequence[Dict[str, float]]]):
        self.hit_map = hit_map
        self.search_calls = []
        self.embedding_calls = []
        self.default_top_k = 25

    def get_embeddings_by_name(self, names: Iterable[str]):
        self.embedding_calls.append(list(names))
        payload = {}
        for name in names:
            if name in self.hit_map:
                payload[name] = [0.1, 0.2]
        return payload

    def search_similar_embeddings(self, embedding, *, top_k, exclude_names):
        self.search_calls.append(
            {
                "embedding": list(embedding),
                "top_k": top_k,
                "exclude": set(exclude_names),
            }
        )
        return list(self.hit_map.get("Track One", []))


class SimpleResolver(TrackNameResolver):
    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    def ids_to_names(self, track_ids):
        return {
            track_id: self.mapping.get(track_id, "")
            for track_id in track_ids
            if track_id in self.mapping
        }

    def name_to_id(self, name: str):
        for track_id, mapped in self.mapping.items():
            if mapped == name:
                return track_id
        return None


@pytest.fixture()
def logger() -> logging.Logger:
    return logging.getLogger("navidrome.tests")


def _make_request(**kwargs):
    seeds = kwargs.pop(
        "seeds",
        [
            RecommendationSeed(track_id="1", weight=1.0, source="seed"),
        ],
    )
    return RecommendationRequest(
        user_id="user",
        user_name="User",
        mode="similar",
        limit=5,
        seeds=seeds,
        exclude_track_ids=kwargs.get("exclude_track_ids", []),
        diversity=kwargs.get("diversity", 0.0),
    )


def test_recommendation_returns_ranked_candidates(logger):
    hits = [{"name": "Track Two", "distance": 0.8, "track_id": "2"}]
    searcher = RecordingSimilaritySearcher(hit_map={"Track One": hits})
    resolver = SimpleResolver({"1": "Track One"})
    engine = RecommendationEngine(searcher=searcher, name_resolver=resolver)

    response = engine.recommend(_make_request())

    assert isinstance(response, RecommendationResponse)
    assert response.track_ids == ["2"]
    assert searcher.search_calls[0]["top_k"] >= 5 * 3
    assert searcher.embedding_calls[0] == ["Track One"]
    assert isinstance(response.tracks[0], RecommendationItem)


def test_recommendation_handles_missing_embeddings(logger):
    searcher = RecordingSimilaritySearcher(hit_map={})
    resolver = SimpleResolver({"1": "Track One"})
    engine = RecommendationEngine(searcher=searcher, name_resolver=resolver)

    response = engine.recommend(_make_request())

    assert not response.tracks
    assert "No matching embeddings" in response.warnings[0]


def test_recommendation_falls_back_to_seed_order(logger):
    searcher = RecordingSimilaritySearcher(hit_map={"Track One": []})
    resolver = SimpleResolver({"1": "Track One"})
    engine = RecommendationEngine(
        searcher=searcher, name_resolver=resolver, debug_logging=True
    )

    response = engine.recommend(_make_request())

    assert response.track_ids == []
    assert any(
        "Similarity search did not yield any candidates" in msg
        for msg in response.warnings
    )
