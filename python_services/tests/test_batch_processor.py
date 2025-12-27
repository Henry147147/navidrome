"""
Tests for the batch processor module.
"""

import base64
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from batch_processor import ModelFirstBatchProcessor, TrackContext
from batch_queue_manager import BatchResult


class MockMuQModel:
    """Mock MuQ embedding model for testing."""

    def __init__(self, should_fail: bool = False, fail_for: Optional[set] = None):
        self.should_fail = should_fail
        self.fail_for = fail_for or set()
        self.embed_calls: List[Dict] = []
        self.ensure_loaded_calls = 0

    def ensure_model_loaded(self):
        self.ensure_loaded_calls += 1

    def embed_music(self, music_file: str, music_name: str) -> dict:
        self.embed_calls.append({"music_file": music_file, "music_name": music_name})

        if self.should_fail or music_name in self.fail_for:
            raise RuntimeError(f"MuQ failed for {music_name}")

        return {
            "music_file": music_file,
            "model_id": "mock-muq",
            "sample_rate": 24000,
            "window_seconds": 120,
            "hop_seconds": 15,
            "segments": [
                {
                    "index": 1,
                    "title": music_name,
                    "offset_seconds": 0.0,
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
        }


class MockCaptioner:
    """Mock Music Flamingo captioner for testing."""

    def __init__(self, should_fail: bool = False, fail_for: Optional[set] = None):
        self.should_fail = should_fail
        self.fail_for = fail_for or set()
        self.generate_calls: List[str] = []

    def generate(self, audio_path: str) -> Tuple[str, List[float]]:
        self.generate_calls.append(audio_path)

        if self.should_fail or audio_path in self.fail_for:
            raise RuntimeError(f"Flamingo failed for {audio_path}")

        return (
            f"A test caption for {Path(audio_path).name}",
            [0.4, 0.5, 0.6],  # audio embedding
        )


class MockEmbedder:
    """Mock Qwen3 text embedder for testing."""

    def __init__(self, should_fail: bool = False, fail_for: Optional[set] = None):
        self.should_fail = should_fail
        self.fail_for = fail_for or set()
        self.embed_calls: List[str] = []

    def embed_text(self, text: str) -> torch.Tensor:
        self.embed_calls.append(text)

        if self.should_fail or text in self.fail_for:
            raise RuntimeError(f"Qwen3 failed for {text}")

        return torch.tensor([0.7, 0.8, 0.9])


class MockDescriptionPipeline:
    """Mock description embedding pipeline for testing."""

    def __init__(
        self,
        captioner: Optional[MockCaptioner] = None,
        embedder: Optional[MockEmbedder] = None,
    ):
        self._captioner = captioner or MockCaptioner()
        self._embedder = embedder or MockEmbedder()
        self.unload_captioner_calls = 0

    def _get_captioner(self) -> MockCaptioner:
        return self._captioner

    def _get_embedder(self) -> MockEmbedder:
        return self._embedder

    def unload_captioner(self):
        self.unload_captioner_calls += 1


class TestTrackContext:
    """Tests for the TrackContext dataclass."""

    def test_track_context_creation(self):
        ctx = TrackContext(
            request_id="req-1",
            music_file="/path/to/song.mp3",
            music_name="Artist - Song",
            payload={"action": "embed"},
        )
        assert ctx.request_id == "req-1"
        assert ctx.music_file == "/path/to/song.mp3"
        assert ctx.music_name == "Artist - Song"
        assert ctx.muq_result is None
        assert ctx.caption is None
        assert ctx.error is None

    def test_track_context_with_error(self):
        ctx = TrackContext(
            request_id="req-1",
            music_file="",
            music_name="",
            payload={},
            error="File not found",
        )
        assert ctx.error == "File not found"


class TestModelFirstBatchProcessor:
    """Tests for the ModelFirstBatchProcessor class."""

    @pytest.fixture
    def temp_audio_files(self, tmp_path):
        """Create temporary audio files for testing."""
        files = []
        for i in range(3):
            path = tmp_path / f"song{i}.mp3"
            path.write_text("fake audio content")
            files.append(str(path))
        return files

    def test_process_batch_with_muq_only(self, temp_audio_files):
        """Test batch processing with only MuQ model (no description pipeline)."""
        muq_model = MockMuQModel()

        processor = ModelFirstBatchProcessor(
            muq_model=muq_model,
            description_pipeline=None,
        )

        payloads = [
            {"request_id": f"req-{i}", "music_file": f, "name": f"song{i}.mp3"}
            for i, f in enumerate(temp_audio_files)
        ]

        results = processor.process_batch(payloads)

        assert len(results) == 3
        assert all(r.success for r in results.values())
        assert muq_model.ensure_loaded_calls == 1
        assert len(muq_model.embed_calls) == 3

    def test_process_batch_with_description_pipeline(self, temp_audio_files):
        """Test batch processing with both MuQ and description pipeline."""
        muq_model = MockMuQModel()
        captioner = MockCaptioner()
        embedder = MockEmbedder()
        pipeline = MockDescriptionPipeline(captioner, embedder)

        processor = ModelFirstBatchProcessor(
            muq_model=muq_model,
            description_pipeline=pipeline,
        )

        payloads = [
            {"request_id": f"req-{i}", "music_file": f, "name": f"song{i}.mp3"}
            for i, f in enumerate(temp_audio_files)
        ]

        results = processor.process_batch(payloads)

        assert len(results) == 3
        assert all(r.success for r in results.values())

        # Verify all stages were called
        assert len(muq_model.embed_calls) == 3
        assert len(captioner.generate_calls) == 3
        assert len(embedder.embed_calls) == 3

        # Verify captioner was unloaded before embedder
        assert pipeline.unload_captioner_calls == 1

        # Verify results have all components
        for result in results.values():
            assert "segments" in result.payload
            assert "descriptions" in result.payload

    def test_description_entries_include_model_ids(self, temp_audio_files):
        """Descriptions should include text/audio model ids when available."""
        muq_model = MockMuQModel()
        pipeline = MockDescriptionPipeline()
        pipeline.text_model_id = "qwen3"
        pipeline.caption_model_id = "flamingo"

        processor = ModelFirstBatchProcessor(
            muq_model=muq_model,
            description_pipeline=pipeline,
        )

        payloads = [
            {"request_id": "req-1", "music_file": temp_audio_files[0], "name": "a.mp3"}
        ]

        result = processor.process_batch(payloads)["req-1"]
        desc = result.payload["descriptions"][0]

        assert desc["model_id"] == "qwen3"
        assert desc["audio_model_id"] == "flamingo"

    def test_canonical_name_matches_resolver(self, temp_audio_files):
        """Batch processor should normalize artist/title like the resolver."""
        muq_model = MockMuQModel()
        processor = ModelFirstBatchProcessor(
            muq_model=muq_model,
            description_pipeline=None,
        )

        payloads = [
            {
                "request_id": "req-1",
                "music_file": temp_audio_files[0],
                "name": "song.mp3",
                "artist": "AC/DC",
                "title": "Track\\Name",
            }
        ]

        processor.process_batch(payloads)

        assert muq_model.embed_calls[0]["music_name"] == "AC_DC - Track_Name"

    def test_muq_failure_marks_track_as_error(self, temp_audio_files):
        """Test that MuQ failure marks the track with an error."""
        muq_model = MockMuQModel(fail_for={"song1.mp3"})

        processor = ModelFirstBatchProcessor(
            muq_model=muq_model,
            description_pipeline=None,
        )

        payloads = [
            {"request_id": f"req-{i}", "music_file": f, "name": f"song{i}.mp3"}
            for i, f in enumerate(temp_audio_files)
        ]

        results = processor.process_batch(payloads)

        # One should fail, others should succeed
        success_count = sum(1 for r in results.values() if r.success)
        assert success_count == 2

        failed = [r for r in results.values() if not r.success]
        assert len(failed) == 1
        assert "MuQ embedding failed" in failed[0].error

    def test_flamingo_failure_uses_fallback_caption(self, temp_audio_files):
        """Test that Flamingo failure uses a fallback caption."""
        muq_model = MockMuQModel()
        # Fail for second file
        captioner = MockCaptioner(fail_for={temp_audio_files[1]})
        embedder = MockEmbedder()
        pipeline = MockDescriptionPipeline(captioner, embedder)

        processor = ModelFirstBatchProcessor(
            muq_model=muq_model,
            description_pipeline=pipeline,
        )

        payloads = [
            {"request_id": f"req-{i}", "music_file": f, "name": f"song{i}.mp3"}
            for i, f in enumerate(temp_audio_files)
        ]

        results = processor.process_batch(payloads)

        # All should succeed (fallback caption used)
        assert all(r.success for r in results.values())

        # Check fallback caption was used for the failed track
        req1_result = results["req-1"]
        desc = req1_result.payload["descriptions"][0]
        assert "Audio track titled" in desc["description"]
        assert desc["audio_embedding"] == []

    def test_qwen3_failure_still_succeeds(self, temp_audio_files):
        """Test that Qwen3 failure still produces a successful result."""
        muq_model = MockMuQModel()
        captioner = MockCaptioner()
        # Fail for one caption
        embedder = MockEmbedder(fail_for={"A test caption for song1.mp3"})
        pipeline = MockDescriptionPipeline(captioner, embedder)

        processor = ModelFirstBatchProcessor(
            muq_model=muq_model,
            description_pipeline=pipeline,
        )

        payloads = [
            {"request_id": f"req-{i}", "music_file": f, "name": f"song{i}.mp3"}
            for i, f in enumerate(temp_audio_files)
        ]

        results = processor.process_batch(payloads)

        # All should still succeed (Qwen3 failure is non-fatal)
        assert all(r.success for r in results.values())

        # The failed one should have no description embedding
        req1_result = results["req-1"]
        desc = req1_result.payload["descriptions"][0]
        assert desc["embedding"] is None or desc["embedding"] == []

    def test_missing_music_file_returns_error(self):
        """Test that missing music file returns an error."""
        muq_model = MockMuQModel()

        processor = ModelFirstBatchProcessor(
            muq_model=muq_model,
            description_pipeline=None,
        )

        payloads = [
            {"request_id": "req-0", "music_file": "/nonexistent/file.mp3"},
        ]

        results = processor.process_batch(payloads)

        assert len(results) == 1
        result = results["req-0"]
        assert result.success is False
        assert "not found" in result.error.lower() or "File not found" in result.error

    def test_base64_file_materialization(self, tmp_path):
        """Test that base64-encoded files are materialized correctly."""
        muq_model = MockMuQModel()

        processor = ModelFirstBatchProcessor(
            muq_model=muq_model,
            description_pipeline=None,
        )

        audio_content = b"fake audio bytes"
        b64_content = base64.b64encode(audio_content).decode("ascii")

        payloads = [
            {
                "request_id": "req-0",
                "music_file": "/nonexistent/path/song.mp3",
                "name": "song.mp3",
                "music_data_b64": b64_content,
            },
        ]

        results = processor.process_batch(payloads)

        assert len(results) == 1
        result = results["req-0"]
        assert result.success is True

        # Verify the materialized file was used
        assert len(muq_model.embed_calls) == 1
        used_path = muq_model.embed_calls[0]["music_file"]
        assert used_path != "/nonexistent/path/song.mp3"

    def test_artist_title_canonical_name(self, temp_audio_files):
        """Test that artist and title are used to create canonical name."""
        muq_model = MockMuQModel()

        processor = ModelFirstBatchProcessor(
            muq_model=muq_model,
            description_pipeline=None,
        )

        payloads = [
            {
                "request_id": "req-0",
                "music_file": temp_audio_files[0],
                "artist": "Test Artist",
                "title": "Test Song",
                "name": "file.mp3",
            },
        ]

        results = processor.process_batch(payloads)

        assert results["req-0"].success
        assert muq_model.embed_calls[0]["music_name"] == "Test Artist - Test Song"

    def test_empty_batch(self):
        """Test processing an empty batch."""
        muq_model = MockMuQModel()

        processor = ModelFirstBatchProcessor(
            muq_model=muq_model,
            description_pipeline=None,
        )

        results = processor.process_batch([])

        assert results == {}
        assert muq_model.ensure_loaded_calls == 0

    def test_temp_files_cleaned_up(self, tmp_path):
        """Test that temporary files are cleaned up after processing."""
        muq_model = MockMuQModel()

        processor = ModelFirstBatchProcessor(
            muq_model=muq_model,
            description_pipeline=None,
        )

        audio_content = b"fake audio bytes"
        b64_content = base64.b64encode(audio_content).decode("ascii")

        payloads = [
            {
                "request_id": "req-0",
                "music_file": "/nonexistent/path/song.mp3",
                "name": "song.mp3",
                "music_data_b64": b64_content,
            },
        ]

        results = processor.process_batch(payloads)

        # Get the path that was used
        used_path = muq_model.embed_calls[0]["music_file"]

        # The temp file should be cleaned up
        assert not Path(used_path).exists()

    def test_result_contains_music_name(self, temp_audio_files):
        """Test that results contain the music_name field."""
        muq_model = MockMuQModel()

        processor = ModelFirstBatchProcessor(
            muq_model=muq_model,
            description_pipeline=None,
        )

        payloads = [
            {
                "request_id": "req-0",
                "music_file": temp_audio_files[0],
                "name": "song.mp3",
            },
        ]

        results = processor.process_batch(payloads)

        assert results["req-0"].payload["music_name"] == "song.mp3"

    def test_all_stages_run_in_order(self, temp_audio_files):
        """Test that all stages run in the correct order (MuQ -> Flamingo -> Qwen3)."""
        call_order = []

        class OrderTrackingMuQ(MockMuQModel):
            def ensure_model_loaded(self):
                call_order.append("muq_load")
                super().ensure_model_loaded()

            def embed_music(self, music_file, music_name):
                call_order.append("muq_embed")
                return super().embed_music(music_file, music_name)

        class OrderTrackingCaptioner(MockCaptioner):
            def generate(self, audio_path):
                call_order.append("flamingo_generate")
                return super().generate(audio_path)

        class OrderTrackingEmbedder(MockEmbedder):
            def embed_text(self, text):
                call_order.append("qwen_embed")
                return super().embed_text(text)

        class OrderTrackingPipeline(MockDescriptionPipeline):
            def unload_captioner(self):
                call_order.append("unload_captioner")
                super().unload_captioner()

        muq = OrderTrackingMuQ()
        captioner = OrderTrackingCaptioner()
        embedder = OrderTrackingEmbedder()
        pipeline = OrderTrackingPipeline(captioner, embedder)

        processor = ModelFirstBatchProcessor(
            muq_model=muq,
            description_pipeline=pipeline,
        )

        payloads = [
            {
                "request_id": "req-0",
                "music_file": temp_audio_files[0],
                "name": "song.mp3",
            },
        ]

        processor.process_batch(payloads)

        # Verify order: MuQ load, MuQ embeds, Flamingo generates, unload, Qwen embeds
        assert call_order == [
            "muq_load",
            "muq_embed",
            "flamingo_generate",
            "unload_captioner",
            "qwen_embed",
        ]
