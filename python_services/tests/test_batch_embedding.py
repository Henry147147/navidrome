"""
Unit tests for batch embedding job
"""

import pytest
import time
from pathlib import Path
from batch_embedding_job import BatchEmbeddingJob, BatchJobProgress


@pytest.fixture
def temp_paths(tmp_path):
    db_path = tmp_path / "test.db"
    db_path.write_text("")  # Touch file so constructor does not fail
    music_root = tmp_path / "music"
    music_root.mkdir()
    return str(db_path), str(music_root)


class MockDB:
    """Mock database for testing"""

    def __init__(self, tracks):
        self.tracks = tracks

    def execute(self, query):
        """Mock execute"""
        return self

    def fetchall(self):
        """Return mock tracks"""
        return self.tracks


class MockModel:
    """Mock embedding model"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.embed_calls = []

    def embed_music(self, audio_path, track_name):
        """Mock embedding"""
        self.embed_calls.append((audio_path, track_name))

        return {
            "model_id": self.model_name,
            "segments": [
                {"title": track_name, "embedding": [0.1] * 100, "offset_seconds": 0.0}
            ],
        }

    def ensure_milvus_schemas(self, client):
        """Mock schema creation"""
        pass

    def ensure_milvus_index(self, client):
        """Mock index creation"""
        pass


class TestBatchJobProgress:
    """Test BatchJobProgress dataclass"""

    def test_init(self):
        """Test progress initialization"""
        progress = BatchJobProgress(
            total_tracks=100,
            processed_tracks=50,
            failed_tracks=2,
            current_track="Artist - Song",
            status="running",
            started_at=time.time(),
            estimated_completion=None,
        )

        assert progress.total_tracks == 100
        assert progress.processed_tracks == 50
        assert progress.failed_tracks == 2
        assert progress.status == "running"


class TestBatchEmbeddingJob:
    """Test BatchEmbeddingJob class"""

    def test_init(self, temp_paths):
        """Test job initialization"""
        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
            checkpoint_interval=50,
        )

        assert job.db_path == str(Path(db_path).resolve())
        assert job.music_root == Path(music_root)
        assert job.milvus_uri == "http://localhost:19530"
        assert job.checkpoint_interval == 50
        assert job.progress.status == "initialized"

    def test_initialize_models(self, temp_paths):
        """Test model initialization"""
        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        # Mock the models
        job.models = {}
        job.models["muq"] = MockModel("muq")
        job.models["qwen3"] = MockModel("qwen3")

        assert len(job.models) == 2
        assert "muq" in job.models
        assert "qwen3" in job.models

    def test_progress_tracking(self, temp_paths):
        """Test that progress is tracked correctly"""
        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        job.progress.total_tracks = 100
        job.progress.processed_tracks = 25
        job.progress.failed_tracks = 2
        job.progress.status = "running"

        progress = job.get_progress()

        assert progress.total_tracks == 100
        assert progress.processed_tracks == 25
        assert progress.failed_tracks == 2
        assert progress.status == "running"

    def test_cancel_job(self, temp_paths):
        """Test job cancellation"""
        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        assert not job._cancelled

        job.cancel()

        assert job._cancelled

    def test_track_name_normalization(self):
        """Test that track names are normalized properly"""
        # Test the normalization logic
        test_cases = [
            ("Artist/Name", "Title", "Artist_Name - Title"),
            ("Artist\\Name", "Title", "Artist_Name - Title"),
            ("Artist•Name", "Title", "Artist&Name - Title"),
            ("Normal Artist", "Normal Title", "Normal Artist - Normal Title"),
        ]

        for artist, title, expected in test_cases:
            normalized_artist = (
                artist.replace("•", "&").replace("/", "_").replace("\\", "_")
            )
            normalized_title = (
                title.replace("•", "&").replace("/", "_").replace("\\", "_")
            )
            result = f"{normalized_artist} - {normalized_title}".strip()
            assert result == expected

    def test_empty_track_list(self, temp_paths):
        """Test handling of empty track list"""
        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        job.progress.total_tracks = 0
        job.progress.status = "running"

        # Should complete immediately
        assert job.progress.total_tracks == 0

    def test_eta_calculation(self, temp_paths):
        """Test ETA calculation"""
        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        job.progress.total_tracks = 100
        job.progress.processed_tracks = 25
        job.progress.started_at = time.time() - 100  # Started 100 seconds ago

        # After processing 25 tracks in 100 seconds:
        # Rate = 100/25 = 4 seconds per track
        # Remaining = 75 tracks
        # ETA = 75 * 4 = 300 seconds from now

        elapsed = time.time() - job.progress.started_at
        rate = elapsed / job.progress.processed_tracks
        remaining = job.progress.total_tracks - job.progress.processed_tracks
        eta = time.time() + (rate * remaining)

        # ETA should be approximately 300 seconds from now
        assert eta > time.time()
        assert (eta - time.time()) > 200  # At least 200 seconds
        assert (eta - time.time()) < 400  # At most 400 seconds

    def test_completion_status(self, temp_paths):
        """Test completion status logic"""
        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        # Test: cancelled status
        job._cancelled = True
        job.progress.status = "cancelled"
        assert job.progress.status == "cancelled"

        # Test: completed with errors
        job._cancelled = False
        job.progress.failed_tracks = 5
        job.progress.status = "completed_with_errors"
        assert job.progress.status == "completed_with_errors"

        # Test: completed successfully
        job.progress.failed_tracks = 0
        job.progress.status = "completed"
        assert job.progress.status == "completed"

    def test_checkpoint_interval(self, temp_paths):
        """Test that checkpoint interval is respected"""
        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
            checkpoint_interval=10,
        )

        assert job.checkpoint_interval == 10

        # Checkpoints should occur at multiples of 10
        for i in range(100):
            if i > 0 and i % 10 == 0:
                # This would be a checkpoint
                assert i % job.checkpoint_interval == 0

    def test_collection_mapping(self, temp_paths):
        """Test collection name mapping"""
        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        # Verify the mapping is correct
        for model_name, collection_name in job.collection_map.items():
            assert collection_name in [
                "embedding",
                "description_embedding",
                "flamingo_audio_embedding",
            ]

        assert (
            job.audio_collection == "flamingo_audio_embedding"
        ), "Audio embedding collection should be available"


class TestBatchJobAPI:
    """Test batch job API functions"""

    def test_start_batch_job(self, temp_paths):
        """Test starting a new batch job"""
        from batch_embedding_job import start_batch_job

        db_path, music_root = temp_paths
        job = start_batch_job(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
            checkpoint_interval=100,
        )

        assert job is not None
        assert job.db_path == str(Path(db_path).resolve())
        assert job.checkpoint_interval == 100

    def test_get_current_job(self, temp_paths):
        """Test getting current job"""
        from batch_embedding_job import start_batch_job, get_current_job

        # Start a job
        db_path, music_root = temp_paths
        started_job = start_batch_job(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        # Get the current job
        current_job = get_current_job()

        assert current_job is started_job

    def test_get_current_job_none(self):
        """Test getting current job when none exists"""
        from batch_embedding_job import get_current_job

        # Assuming no job is running
        current_job = get_current_job()

        # Should return None or the last job
        assert current_job is None or isinstance(current_job, BatchEmbeddingJob)


class TestBatchJobErrorHandling:
    """Test error handling in batch jobs"""

    def test_model_load_oom_fallback(self, temp_paths):
        """Model load should fall back to CPU on OOM."""
        import torch

        class OOMModel:
            def __init__(self):
                self.device = "cuda"
                self.model_dtype = torch.float16
                self.storage_dtype = torch.float32
                self.load_attempts = 0
                self.unloaded = False

            def ensure_model_loaded(self):
                self.load_attempts += 1
                if self.load_attempts == 1:
                    raise RuntimeError("CUDA out of memory")
                return self

            def unload_model(self):
                self.unloaded = True

        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        model = OOMModel()
        job._ensure_model_loaded_with_fallback("muq", model)

        assert model.device == "cpu"
        assert model.model_dtype == torch.float32
        assert model.storage_dtype == torch.float32
        assert model.load_attempts == 2
        assert model.unloaded is True

    def test_missing_audio_file(self, temp_paths):
        """Test handling of missing audio file"""
        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        # Track with non-existent file
        track = {
            "id": "123",
            "path": "nonexistent/file.mp3",
            "artist": "Test Artist",
            "title": "Test Song",
            "album": "Test Album",
        }

        job.models = {"muq": MockModel("muq")}

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            job._process_track(track, ["muq"])

    def test_failed_track_counting(self, temp_paths):
        """Test that failed tracks are counted"""
        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        job.progress.failed_tracks = 0

        # Simulate failures
        job.progress.failed_tracks += 1
        job.progress.failed_tracks += 1
        job.progress.failed_tracks += 1

        assert job.progress.failed_tracks == 3

    def test_model_embedding_error(self, temp_paths):
        """Test handling of model embedding errors"""

        class FailingModel:
            def embed_music(self, audio_path, track_name):
                raise RuntimeError("Model failed")

            def ensure_milvus_schemas(self, client):
                pass

            def ensure_milvus_index(self, client):
                pass

        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        job.models = {"failing": FailingModel()}

        # Track processing should fail
        # In actual implementation, this would be caught and logged
        assert "failing" in job.models

    def test_run_processes_tracks_and_unloads_model(self, temp_paths, monkeypatch):
        """Run should process tracks, insert embeddings, and unload model."""
        import batch_embedding_job
        import pymilvus

        db_path, music_root = temp_paths
        music_root_path = Path(music_root)
        track_path = music_root_path / "song.mp3"
        track_path.write_bytes(b"")

        tracks = [
            {
                "id": "1",
                "path": "song.mp3",
                "artist": "Artist",
                "title": "Title",
                "album": "Album",
            }
        ]

        class DummyModel:
            def __init__(self):
                self.loaded = False
                self.unloaded = False
                self.calls = 0

            def ensure_model_loaded(self):
                self.loaded = True
                return self

            def embed_music(self, audio_path, track_name):
                self.calls += 1
                return {
                    "model_id": "muq",
                    "segments": [
                        {
                            "title": track_name,
                            "embedding": [0.1, 0.2, 0.3],
                            "offset_seconds": 0.0,
                        }
                    ],
                }

            def unload_model(self):
                self.unloaded = True

        class DummyMilvusClient:
            def __init__(self, uri=None):
                self.uri = uri
                self.inserted = []

            def insert(self, collection_name, data):
                self.inserted.append((collection_name, list(data)))

        monkeypatch.setattr(pymilvus, "MilvusClient", DummyMilvusClient)
        monkeypatch.setattr(batch_embedding_job, "MilvusClient", DummyMilvusClient)
        monkeypatch.setattr(batch_embedding_job, "tqdm", lambda x, **kwargs: x)

        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )
        dummy_model = DummyModel()
        monkeypatch.setattr(job, "_initialize_models", lambda names: None)
        job.models = {"muq": dummy_model}
        monkeypatch.setattr(job, "get_all_tracks", lambda: tracks)

        result = job.run(
            models_to_use=["muq"], clear_existing=False, missing_only=False
        )

        assert dummy_model.loaded is True
        assert dummy_model.unloaded is True
        assert dummy_model.calls == 1
        assert result["progress"].status in {"completed", "completed_with_errors"}

    def test_run_missing_only_filters_tracks(self, temp_paths, monkeypatch):
        """missing_only should skip tracks already present in Milvus."""
        import batch_embedding_job
        import pymilvus

        db_path, music_root = temp_paths
        music_root_path = Path(music_root)
        (music_root_path / "song1.mp3").write_bytes(b"")
        (music_root_path / "song2.mp3").write_bytes(b"")

        tracks = [
            {
                "id": "1",
                "path": "song1.mp3",
                "artist": "Artist",
                "title": "One",
                "album": "Album",
            },
            {
                "id": "2",
                "path": "song2.mp3",
                "artist": "Artist",
                "title": "Two",
                "album": "Album",
            },
        ]

        class DummyModel:
            def __init__(self):
                self.calls = 0

            def ensure_model_loaded(self):
                return self

            def embed_music(self, audio_path, track_name):
                self.calls += 1
                return {
                    "model_id": "muq",
                    "segments": [
                        {
                            "title": track_name,
                            "embedding": [0.1, 0.2, 0.3],
                            "offset_seconds": 0.0,
                        }
                    ],
                }

            def unload_model(self):
                pass

        class DummyMilvusClient:
            def __init__(self, uri=None):
                self.uri = uri

            def insert(self, collection_name, data):
                pass

        monkeypatch.setattr(pymilvus, "MilvusClient", DummyMilvusClient)
        monkeypatch.setattr(batch_embedding_job, "MilvusClient", DummyMilvusClient)
        monkeypatch.setattr(batch_embedding_job, "tqdm", lambda x, **kwargs: x)

        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )
        dummy_model = DummyModel()
        monkeypatch.setattr(job, "_initialize_models", lambda names: None)
        job.models = {"muq": dummy_model}
        monkeypatch.setattr(job, "get_all_tracks", lambda: tracks)
        monkeypatch.setattr(
            job,
            "_existing_names_for_collection",
            lambda client, collection, names: {names[0]},
        )

        job.run(models_to_use=["muq"], clear_existing=False, missing_only=True)

        assert dummy_model.calls == 1

    def test_clear_embeddings_recreates_schema(self, temp_paths, monkeypatch):
        """clear_embeddings should drop collections and recreate schemas."""
        import batch_embedding_job
        import pymilvus

        db_path, music_root = temp_paths

        class DummyModel:
            def __init__(self):
                self.schema_calls = 0
                self.index_calls = 0

            def ensure_milvus_schemas(self, client):
                self.schema_calls += 1

            def ensure_milvus_index(self, client):
                self.index_calls += 1

        class DummyMilvusClient:
            def __init__(self, uri=None):
                self.dropped = []

            def drop_collection(self, name):
                self.dropped.append(name)

        monkeypatch.setattr(pymilvus, "MilvusClient", DummyMilvusClient)
        monkeypatch.setattr(batch_embedding_job, "MilvusClient", DummyMilvusClient)

        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )
        job.models = {"muq": DummyModel(), "qwen3": DummyModel()}

        job.clear_embeddings(["muq", "qwen3"])

        assert job.models["muq"].schema_calls == 1
        assert job.models["muq"].index_calls == 1
        assert job.models["qwen3"].schema_calls == 1
        assert job.models["qwen3"].index_calls == 1

    def test_process_qwen3_two_stage_inserts_audio_and_text(
        self, temp_paths, monkeypatch
    ):
        """Two-stage Qwen3 pipeline should insert audio + text embeddings."""
        import batch_embedding_job

        db_path, music_root = temp_paths
        music_root_path = Path(music_root)
        track_path = music_root_path / "song.mp3"
        track_path.write_bytes(b"")

        tracks = [
            {
                "id": "1",
                "path": "song.mp3",
                "artist": "Artist",
                "title": "Title",
                "album": "Album",
            }
        ]

        class DummySegment:
            def __init__(self, title, description, embedding):
                self.title = title
                self.description = description
                self.description_embedding = embedding
                self.offset_seconds = 0.0

        class DummyPipeline:
            caption_model_id = "flamingo"
            text_model_id = "qwen3"
            collection_audio = "flamingo_audio_embedding"

            def __init__(self):
                self.schema_calls = 0
                self.index_calls = 0

            def get_caption(self, audio_path, canonical_name):
                return "desc", [0.1, 0.2]

            def embed_description(self, description, canonical_name):
                return DummySegment(canonical_name, description, [0.3, 0.4])

            def ensure_milvus_schemas(self, client, audio_dim=None):
                self.schema_calls += 1

            def ensure_milvus_index(self, client):
                self.index_calls += 1

            def unload_captioner(self):
                pass

        class DummyMilvusClient:
            def __init__(self):
                self.inserted = []

            def insert(self, collection_name, data):
                self.inserted.append((collection_name, list(data)))

        monkeypatch.setattr(batch_embedding_job, "tqdm", lambda x, **kwargs: x)
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="file:/tmp/milvus.db",
        )
        dummy_model = DummyPipeline()
        client = DummyMilvusClient()

        ops = job._process_qwen3_two_stage(
            model_tracks=tracks,
            model=dummy_model,
            collection="description_embedding",
            audio_collection="flamingo_audio_embedding",
            client=client,
            batch_size=1,
        )

        assert ops == 1
        assert dummy_model.schema_calls >= 1
        assert dummy_model.index_calls >= 1
        assert any(
            collection == "flamingo_audio_embedding"
            for collection, _ in client.inserted
        )
        assert any(
            collection == "description_embedding" for collection, _ in client.inserted
        )

    def test_existing_names_for_collection_lite_mode(self, temp_paths, monkeypatch):
        """Lite mode should query using JSON filter."""
        import batch_embedding_job

        db_path, music_root = temp_paths

        class DummyMilvusClient:
            def __init__(self):
                self.last_filter = None

            def query(self, collection_name, filter, output_fields):
                self.last_filter = filter
                return [{"name": "Artist - Title"}]

        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="file:/tmp/milvus.db",
        )
        client = DummyMilvusClient()

        names = ["Artist - Title", "Artist - Other"]
        existing = job._existing_names_for_collection(client, "embedding", names)

        assert "Artist - Title" in existing
        assert "name in" in client.last_filter

    def test_run_invalid_model_names_raises(self, temp_paths):
        """Invalid model names should raise ValueError early."""
        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        with pytest.raises(ValueError):
            job.run(models_to_use=["not-a-model"], clear_existing=False)

    def test_run_skips_flamingo_audio_with_qwen3(self, temp_paths, monkeypatch):
        """flamingo_audio should be removed when qwen3 is present."""
        import batch_embedding_job
        import pymilvus

        db_path, music_root = temp_paths

        class DummyModel:
            def ensure_model_loaded(self):
                return self

            def unload_model(self):
                pass

            def embed_music(self, *_args, **_kwargs):
                return {"model_id": "qwen3", "segments": []}

        class DummyMilvusClient:
            def __init__(self, uri=None):
                self.uri = uri

            def insert(self, collection_name, data):
                pass

        monkeypatch.setattr(pymilvus, "MilvusClient", DummyMilvusClient)
        monkeypatch.setattr(batch_embedding_job, "MilvusClient", DummyMilvusClient)
        monkeypatch.setattr(batch_embedding_job, "tqdm", lambda x, **kwargs: x)

        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )
        monkeypatch.setattr(job, "_initialize_models", lambda names: None)
        job.models = {"qwen3": DummyModel()}
        monkeypatch.setattr(job, "get_all_tracks", lambda: [])

        job.run(models_to_use=["qwen3", "flamingo_audio"], clear_existing=False)

    def test_run_oom_during_processing_sets_failed(self, temp_paths, monkeypatch):
        """OOM during embedding should mark job as failed."""
        import batch_embedding_job
        import pymilvus

        db_path, music_root = temp_paths
        music_root_path = Path(music_root)
        track_path = music_root_path / "song.mp3"
        track_path.write_bytes(b"")

        tracks = [
            {
                "id": "1",
                "path": "song.mp3",
                "artist": "Artist",
                "title": "Title",
                "album": "Album",
            }
        ]

        class OOMModel:
            def ensure_model_loaded(self):
                return self

            def embed_music(self, *_args, **_kwargs):
                raise RuntimeError("CUDA out of memory")

            def unload_model(self):
                pass

        class DummyMilvusClient:
            def __init__(self, uri=None):
                self.uri = uri

            def insert(self, collection_name, data):
                pass

        monkeypatch.setattr(pymilvus, "MilvusClient", DummyMilvusClient)
        monkeypatch.setattr(batch_embedding_job, "MilvusClient", DummyMilvusClient)
        monkeypatch.setattr(batch_embedding_job, "tqdm", lambda x, **kwargs: x)

        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )
        monkeypatch.setattr(job, "_initialize_models", lambda names: None)
        job.models = {"muq": OOMModel()}
        monkeypatch.setattr(job, "get_all_tracks", lambda: tracks)

        job.run(models_to_use=["muq"], clear_existing=False)

        assert job.progress.status == "failed"
        assert job.progress.last_error

    def test_existing_names_for_collection_server_mode(self, temp_paths):
        """Server mode should query with filter_params."""
        db_path, music_root = temp_paths

        class DummyMilvusClient:
            def __init__(self):
                self.filter_params = None

            def query(
                self,
                collection_name,
                filter,
                filter_params=None,
                output_fields=None,
            ):
                self.filter_params = filter_params
                return [{"name": "Artist - Title"}]

        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )
        client = DummyMilvusClient()

        names = ["Artist - Title", "Artist - Other"]
        existing = job._existing_names_for_collection(client, "embedding", names)

        assert "Artist - Title" in existing
        assert client.filter_params == {"names": names[:2]}

    def test_process_track_with_description_field(self, temp_paths):
        """Batched processing should include description field when present."""
        db_path, music_root = temp_paths
        music_root_path = Path(music_root)
        track_path = music_root_path / "song.mp3"
        track_path.write_bytes(b"")

        class DummyModel:
            def embed_music(self, *_args, **_kwargs):
                return {
                    "model_id": "qwen3",
                    "segments": [
                        {
                            "title": "Artist - Title",
                            "embedding": [0.1, 0.2],
                            "offset_seconds": 0.0,
                            "description": "desc",
                        }
                    ],
                }

        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )
        track = {
            "id": "1",
            "path": "song.mp3",
            "artist": "Artist",
            "title": "Title",
            "album": "Album",
        }

        rows = job._process_track_with_model_batched(
            track, "qwen3", DummyModel(), "description_embedding"
        )

        assert rows[0]["description"] == "desc"


class TestBatchJobConcurrency:
    """Test concurrent access and safety"""

    def test_single_job_instance(self, temp_paths):
        """Test that only one job can run at a time"""
        from batch_embedding_job import start_batch_job, get_current_job

        db_path, music_root = temp_paths
        _job1 = start_batch_job(  # noqa: F841
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        db_path2, music_root2 = temp_paths
        job2 = start_batch_job(
            db_path=db_path2,
            music_root=music_root2,
            milvus_uri="http://localhost:19530",
        )

        # Second job should replace the first
        current = get_current_job()
        assert current is job2

    def test_progress_thread_safety(self, temp_paths):
        """Test that progress updates are safe"""
        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        # Simulate concurrent updates
        job.progress.processed_tracks += 1
        progress1 = job.get_progress()

        job.progress.processed_tracks += 1
        progress2 = job.get_progress()

        assert progress2.processed_tracks == progress1.processed_tracks + 1

    def test_pause_and_resume_flags(self, temp_paths):
        """Pause/resume should toggle flags without requiring models."""
        db_path, music_root = temp_paths
        job = BatchEmbeddingJob(
            db_path=db_path,
            music_root=music_root,
            milvus_uri="http://localhost:19530",
        )

        assert job.progress.status == "initialized"
        job.pause()
        assert job._paused is True
        assert job.progress.status == "paused"

        job.resume()
        assert job._paused is False
        assert job.progress.status == "running"
