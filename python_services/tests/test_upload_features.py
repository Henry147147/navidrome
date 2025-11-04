import logging
from pathlib import Path

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


class DummyRenamer:
    instances = []

    def __init__(self, *, endpoint, model, reasoning_level, logger):
        self.endpoint = endpoint
        self.model = model
        self.reasoning_level = reasoning_level
        self.logger = logger
        self.calls = []
        DummyRenamer.instances.append(self)

    def rename_segments(self, name, prompt, metadata):
        self.calls.append(
            {
                "name": name,
                "prompt": prompt,
                "metadata": metadata,
            }
        )
        return f"renamed-{name}"


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


def test_rename_when_feature_disabled_returns_original(tmp_path, logger):
    searcher = RecordingSearcher()
    pipeline = UploadFeaturePipeline(similarity_searcher=searcher, logger=logger)
    settings = UploadSettings(rename_enabled=False)
    original = tmp_path / "Song.flac"

    resolved = pipeline.rename(str(original), settings, music_file=str(original))

    assert resolved == str(original)


def test_rename_uses_openai_stub(monkeypatch, tmp_path, logger):
    DummyRenamer.instances.clear()

    def fake_metadata(path):
        return {"title": "Example", "artists": ["Artist"]}

    monkeypatch.setattr("upload_features.best_effort_parse_metadata", fake_metadata)
    monkeypatch.setattr("upload_features.OpenAiRenamer", DummyRenamer)

    searcher = RecordingSearcher()
    pipeline = UploadFeaturePipeline(similarity_searcher=searcher, logger=logger)
    settings = UploadSettings(
        rename_enabled=True,
        renaming_prompt="Keep concise",
        openai_endpoint="https://example.com/api",
        openai_model="gpt-test",
        use_metadata=True,
    )

    source = tmp_path / "Song.flac"
    source.write_text("", encoding="utf-8")

    result = pipeline.rename(str(source), settings, music_file=str(source))

    assert result.endswith(".flac")
    assert "renamed-" in Path(result).stem
    renamer = DummyRenamer.instances[0]
    call = renamer.calls[0]
    assert call["name"] == "Song"
    assert call["metadata"] == {"title": "Example", "artists": ["Artist"]}
