"""
Batch Embedding Job

Manages batch re-embedding of entire music library across multiple embedding models.
Provides progress tracking, error handling, and graceful cancellation.
"""

import logging
import sqlite3
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from embedding_models import MertModel, MuQEmbeddingModel, MusicLatentSpaceModel

logger = logging.getLogger(__name__)


@dataclass
class BatchJobProgress:
    """Tracks progress of batch embedding job"""

    total_tracks: int
    processed_tracks: int
    failed_tracks: int
    current_track: Optional[str]
    status: str  # running, completed, failed, cancelled
    started_at: float
    estimated_completion: Optional[float]


class BatchEmbeddingJob:
    """
    Manages batch re-embedding of entire music library.

    Features:
    - Progress tracking with ETA calculation
    - Graceful cancellation
    - Error recovery with detailed logging
    - Support for multiple embedding models
    - Checkpoint intervals for safety
    """

    def __init__(
        self,
        db_path: str,
        music_root: str,
        milvus_uri: str,
        checkpoint_interval: int = 100,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize batch embedding job.

        Args:
            db_path: Path to Navidrome SQLite database
            music_root: Root directory containing music files
            milvus_uri: URI for Milvus connection
            checkpoint_interval: Save progress every N tracks
            logger: Optional logger instance
        """
        self.db_path = db_path
        self.music_root = Path(music_root)
        self.milvus_uri = milvus_uri
        self.checkpoint_interval = checkpoint_interval
        self.logger = logger or logging.getLogger(__name__)

        self.progress = BatchJobProgress(
            total_tracks=0,
            processed_tracks=0,
            failed_tracks=0,
            current_track=None,
            status="initialized",
            started_at=0,
            estimated_completion=None,
        )

        # Initialize models (lazy loading)
        self.models = {}
        self._cancelled = False

        # Collection mapping for each model
        self.collection_map = {
            "muq": "embedding",
            "mert": "mert_embedding",
            "latent": "latent_embedding",
        }

    def _initialize_models(self, model_names: List[str]) -> None:
        """Initialize embedding models."""
        for model_name in model_names:
            if model_name == "muq":
                self.logger.info("Initializing MuQ model...")
                self.models["muq"] = MuQEmbeddingModel(logger=self.logger)
            elif model_name == "mert":
                self.logger.info("Initializing MERT model...")
                self.models["mert"] = MertModel(logger=self.logger)
            elif model_name == "latent":
                self.logger.info("Initializing Latent Space model...")
                self.models["latent"] = MusicLatentSpaceModel(logger=self.logger)
            else:
                self.logger.warning(f"Unknown model: {model_name}")

    def get_all_tracks(self) -> List[Dict]:
        """Query all tracks from Navidrome database."""
        self.logger.info(f"Querying tracks from database: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT id, path, artist, title, album
                FROM media_file
                WHERE path IS NOT NULL
                ORDER BY id
            """
            )

            tracks = [dict(row) for row in cursor.fetchall()]
            self.logger.info(f"Found {len(tracks)} tracks in database")
            return tracks

        finally:
            conn.close()

    def clear_embeddings(self, models_to_use: List[str]) -> None:
        """
        Clear specified Milvus collections.

        Args:
            models_to_use: List of model names whose collections should be cleared
        """
        from pymilvus import MilvusClient

        client = MilvusClient(uri=self.milvus_uri)

        collection_map = {
            "muq": "embedding",
            "mert": "mert_embedding",
            "latent": "latent_embedding",
        }

        for model_name in models_to_use:
            collection = collection_map.get(model_name)
            if not collection:
                continue

            try:
                self.logger.info(f"Dropping collection: {collection}")
                client.drop_collection(collection)
            except Exception as e:
                self.logger.warning(f"Failed to drop collection {collection}: {e}")

        # Recreate schemas
        self._recreate_schemas(models_to_use, client)

    def _recreate_schemas(self, models_to_use: List[str], client) -> None:
        """Recreate Milvus schemas and indexes for specified models."""
        for model_name in models_to_use:
            if model_name not in self.models:
                continue

            try:
                model = self.models[model_name]
                self.logger.info(f"Creating schema for {model_name}...")
                model.ensure_milvus_schemas(client)
                model.ensure_milvus_index(client)
            except Exception as e:
                self.logger.error(f"Failed to create schema for {model_name}: {e}")

    def run(
        self,
        models_to_use: Optional[List[str]] = None,
        clear_existing: bool = True,
    ) -> Dict:
        """
        Run batch embedding job with SEQUENTIAL model processing.

        This processes all tracks with one model at a time to avoid GPU overflow.
        Pattern: Load Model 1 -> Process all tracks -> Unload Model 1 -> Load Model 2 -> etc.

        Args:
            models_to_use: List of model names (default: all)
            clear_existing: Whether to clear existing embeddings first

        Returns:
            Dict with progress and failed tracks
        """
        if models_to_use is None:
            models_to_use = ["muq", "mert", "latent"]

        self.logger.info(f"Starting batch embedding job with models: {models_to_use}")

        # Initialize models
        self._initialize_models(models_to_use)

        # Get all tracks
        tracks = self.get_all_tracks()

        # Calculate total operations (tracks × models)
        total_operations = len(tracks) * len(models_to_use)
        self.progress.total_tracks = total_operations
        self.progress.status = "running"
        self.progress.started_at = time.time()

        # Clear existing embeddings
        if clear_existing:
            self.clear_embeddings(models_to_use)

        # Process tracks sequentially by model
        failed_tracks = []
        operations_completed = 0

        from pymilvus import MilvusClient
        client = MilvusClient(uri=self.milvus_uri)

        # SEQUENTIAL MODEL PROCESSING - one model at a time to avoid GPU overflow
        for model_idx, model_name in enumerate(models_to_use):
            if self._cancelled:
                self.progress.status = "cancelled"
                self.logger.info("Job cancelled by user")
                break

            self.logger.info(f"Processing all tracks with model: {model_name} ({model_idx + 1}/{len(models_to_use)})")
            model = self.models[model_name]
            collection = self.collection_map[model_name]

            # Ensure model is loaded
            model.ensure_model_loaded()

            try:
                # Process all tracks with this model
                for track_idx, track in enumerate(tqdm(tracks, desc=f"Embedding with {model_name}")):
                    if self._cancelled:
                        self.progress.status = "cancelled"
                        self.logger.info("Job cancelled by user")
                        break

                    self.progress.current_track = f"[{model_name}] {track['artist']} - {track['title']}"

                    try:
                        self._process_track_with_model(track, model_name, model, collection, client)
                        operations_completed += 1
                        self.progress.processed_tracks = operations_completed
                    except Exception as e:
                        self.logger.error(f"Failed to process track {track['id']} with {model_name}: {e}")
                        failed_tracks.append((track["id"], model_name, str(e)))
                        self.progress.failed_tracks += 1

                    # Update estimated completion every 10 operations
                    if operations_completed > 0 and operations_completed % 10 == 0:
                        elapsed = time.time() - self.progress.started_at
                        rate = elapsed / operations_completed
                        remaining = total_operations - operations_completed
                        self.progress.estimated_completion = time.time() + (rate * remaining)

                # CRITICAL: Explicitly unload model after processing all tracks
                self.logger.info(f"Unloading {model_name} model to free GPU memory")
                model.unload_model()

            except Exception as e:
                self.logger.error(f"Failed during {model_name} processing: {e}")
                # Ensure model is unloaded even on error
                try:
                    model.unload_model()
                except Exception as unload_error:
                    self.logger.error(f"Failed to unload {model_name}: {unload_error}")
                raise

        # Finalize
        if self._cancelled:
            self.progress.status = "cancelled"
        elif self.progress.failed_tracks > 0:
            self.progress.status = "completed_with_errors"
        else:
            self.progress.status = "completed"

        elapsed = time.time() - self.progress.started_at
        self.logger.info(
            f"Job completed in {elapsed:.1f}s: "
            f"{operations_completed}/{total_operations} operations "
            f"({len(tracks)} tracks × {len(models_to_use)} models)"
        )
        self.logger.info(f"Failed operations: {self.progress.failed_tracks}")

        return {"progress": self.progress, "failed_tracks": failed_tracks}

    def _process_track(self, track: Dict, models: List[str]) -> None:
        """Process a single track with specified models (DEPRECATED - use _process_track_with_model instead)."""
        from pymilvus import MilvusClient

        # Resolve audio file path
        audio_path = self.music_root / track["path"]
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Normalize track name (artist - title format)
        artist = (
            str(track["artist"]).replace("•", "&").replace("/", "_").replace("\\", "_")
        )
        title = (
            str(track["title"]).replace("•", "&").replace("/", "_").replace("\\", "_")
        )
        canonical_name = f"{artist} - {title}".strip()

        client = MilvusClient(uri=self.milvus_uri)

        # Generate embeddings with each model
        for model_name in models:
            model = self.models.get(model_name)
            if not model:
                continue

            try:
                # Generate embedding
                result = model.embed_music(str(audio_path), canonical_name)

                # Store in Milvus
                collection = self.collection_map[model_name]

                for segment in result["segments"]:
                    client.insert(
                        collection_name=collection,
                        data=[
                            {
                                "name": segment["title"],
                                "embedding": segment["embedding"],
                                "offset": segment["offset_seconds"],
                                "model_id": result["model_id"],
                            }
                        ],
                    )

                if self.logger.level <= logging.DEBUG:
                    self.logger.debug(
                        f"Embedded {canonical_name} with {model_name} "
                        f"({len(result['segments'])} segments)"
                    )

            except Exception as e:
                self.logger.error(
                    f"Failed to embed {canonical_name} with {model_name}: {e}"
                )
                raise

    def _process_track_with_model(self, track: Dict, model_name: str, model, collection: str, client) -> None:
        """Process a single track with a specific model."""
        # Resolve audio file path
        audio_path = self.music_root / track["path"]
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Normalize track name (artist - title format)
        artist = (
            str(track["artist"]).replace("•", "&").replace("/", "_").replace("\\", "_")
        )
        title = (
            str(track["title"]).replace("•", "&").replace("/", "_").replace("\\", "_")
        )
        canonical_name = f"{artist} - {title}".strip()

        # Generate embedding
        result = model.embed_music(str(audio_path), canonical_name)

        # Store in Milvus
        for segment in result["segments"]:
            client.insert(
                collection_name=collection,
                data=[
                    {
                        "name": segment["title"],
                        "embedding": segment["embedding"],
                        "offset": segment["offset_seconds"],
                        "model_id": result["model_id"],
                    }
                ],
            )

        if self.logger.level <= logging.DEBUG:
            self.logger.debug(
                f"Embedded {canonical_name} with {model_name} "
                f"({len(result['segments'])} segments)"
            )

    def cancel(self) -> None:
        """Cancel the running job."""
        self._cancelled = True
        self.logger.info("Job cancellation requested")

    def get_progress(self) -> BatchJobProgress:
        """Get current job progress."""
        return replace(self.progress)


# Global job instance (for API access)
_current_job: Optional[BatchEmbeddingJob] = None


def start_batch_job(
    db_path: str,
    music_root: str,
    milvus_uri: str,
    checkpoint_interval: int = 100,
) -> BatchEmbeddingJob:
    """
    Start a new batch embedding job.

    Args:
        db_path: Path to Navidrome SQLite database
        music_root: Root directory containing music files
        milvus_uri: URI for Milvus connection
        checkpoint_interval: Save progress every N tracks

    Returns:
        BatchEmbeddingJob instance
    """
    global _current_job
    _current_job = BatchEmbeddingJob(
        db_path, music_root, milvus_uri, checkpoint_interval
    )
    return _current_job


def get_current_job() -> Optional[BatchEmbeddingJob]:
    """Get the current batch embedding job (if any)."""
    return _current_job
