import logging
from typing import Any, Dict, List

import numpy as np
import pytest

from database_query import MilvusSimilaritySearcher, SimilarityQuery
from models import SongEmbedding


class StubMilvusClient:
    def __init__(self, *, search_result=None, query_result=None) -> None:
        self.loaded: List[str] = []
        self.search_calls: List[Dict[str, Any]] = []
        self.query_calls: List[Dict[str, Any]] = []
        self._search_result = search_result if search_result is not None else [[]]
        self._query_result = query_result if query_result is not None else []

    def load_collection(self, collection_name: str) -> None:
        self.loaded.append(collection_name)

    def search(self, **kwargs: Any) -> List[List[Dict[str, Any]]]:
        self.search_calls.append(kwargs)
        return self._search_result

    def query(self, **kwargs: Any) -> List[Dict[str, Any]]:
        self.query_calls.append(kwargs)
        return self._query_result


@pytest.fixture()
def logger() -> logging.Logger:
    return logging.getLogger("navidrome.tests")


def test_similarity_searcher_requests_expected_fields(logger: logging.Logger):
    client = StubMilvusClient()
    searcher = MilvusSimilaritySearcher(
        client=client,
        logger=logger,
    )

    searcher.search_similar_embeddings(np.array([0.1, 0.2]))
    assert client.search_calls
    assert client.search_calls[0]["output_fields"] == ["name"]

    searcher.get_embeddings_by_name(["abc"])
    assert client.query_calls
    assert client.query_calls[0]["output_fields"] == ["name", "embedding"]


def test_find_similar_applies_filters_and_returns_hits(logger: logging.Logger):
    hits = [
        [
            {"name": "Track A", "distance": 0.9},
            {"name": "Track B", "distance": 0.7},
        ]
    ]
    client = StubMilvusClient(search_result=hits)
    searcher = MilvusSimilaritySearcher(client=client, logger=logger)

    result = searcher.find_similar(
        SimilarityQuery(name="Seed", embedding=[0.1, 0.2]),
        top_k=2,
        exclude_names=["Seed", ""],
    )

    assert result == hits[0]
    call = client.search_calls[0]
    assert call["limit"] == 2
    assert call["filter"] == "name not in {names}"
    assert call["filter_params"] == {"names": ["Seed"]}


def test_identify_duplicates_respects_threshold(logger: logging.Logger):
    hits = [
        [
            {"name": "Duplicate", "distance": 0.95},
            {"name": "Below", "distance": 0.5},
        ]
    ]
    client = StubMilvusClient(search_result=hits)
    searcher = MilvusSimilaritySearcher(client=client, logger=logger)

    payload = SongEmbedding(name="Seed", embedding=[0.3, 0.4], offset=0.0)
    duplicates = searcher.identify_duplicates(payload, 0.9, top_k=2)

    assert duplicates == ["Duplicate"]
