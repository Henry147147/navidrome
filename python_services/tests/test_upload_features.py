import logging

import pytest

from models import SongEmbedding, UploadSettings
from upload_features import UploadFeaturePipeline


class RecordingSearcher:
    def __init__(self, *, duplicates=None):
        self.calls = []
        self.duplicates = duplicates or []

    def identify_duplicates(self, embedding_payload, threshold, *, top_k=None):
        self.calls.append(
            {
                "payload": embedding_payload,
                "threshold": threshold,
                "top_k": top_k,
            }
        )
        return list(self.duplicates)


@pytest.fixture()
def logger() -> logging.Logger:
    return logging.getLogger("navidrome.tests")


def _make_song(name="Track One"):
    return SongEmbedding(name=name, embedding=[0.1, 0.2], offset=0.0)


def test_scan_for_dups_skips_when_disabled(logger):
    searcher = RecordingSearcher()
    pipeline = UploadFeaturePipeline(similarity_searcher=searcher, logger=logger)
    settings = UploadSettings(similarity_search_enabled=False)

    result = pipeline.scan_for_dups([_make_song()], settings)

    assert result == []
    assert not searcher.calls


def test_scan_for_dups_returns_duplicate_names(logger):
    searcher = RecordingSearcher(duplicates=["Existing Track"])
    pipeline = UploadFeaturePipeline(similarity_searcher=searcher, logger=logger)
    settings = UploadSettings(similarity_search_enabled=True, dedup_threshold=0.9)
    song = _make_song("New Track")

    result = pipeline.scan_for_dups([song], settings)

    assert result == ["New Track"]
    assert searcher.calls[0]["threshold"] == pytest.approx(0.9)
