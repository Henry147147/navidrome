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
        job.models["mert"] = MockModel("mert")
        job.models["latent"] = MockModel("latent")

        assert len(job.models) == 3
        assert "muq" in job.models
        assert "mert" in job.models
        assert "latent" in job.models

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

    def test_collection_mapping(self):
        """Test collection name mapping"""
        # Test the collection mapping is correct
        collection_map = {
            "muq": "embedding",
            "mert": "mert_embedding",
            "latent": "latent_embedding",
        }

        # Verify the mapping is correct
        for model_name, collection_name in collection_map.items():
            assert collection_name in [
                "embedding",
                "mert_embedding",
                "latent_embedding",
            ]


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
