import logging
from typing import Any, Dict, List

import numpy as np
import pytest

from database_query import MilvusSimilaritySearcher


class StubMilvusClient:
    def __init__(self) -> None:
        self.loaded: List[str] = []
        self.search_calls: List[Dict[str, Any]] = []
        self.query_calls: List[Dict[str, Any]] = []

    def load_collection(self, collection_name: str) -> None:
        self.loaded.append(collection_name)

    def search(self, **kwargs: Any) -> List[List[Dict[str, Any]]]:
        self.search_calls.append(kwargs)
        return [[]]

    def query(self, **kwargs: Any) -> List[Dict[str, Any]]:
        self.query_calls.append(kwargs)
        return []


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
