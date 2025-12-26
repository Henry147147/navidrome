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

from embedding_models import MuQEmbeddingModel
from description_pipeline import (
    DEFAULT_AUDIO_COLLECTION,
    DescriptionEmbeddingPipeline,
)
from gpu_settings import GPUSettings, is_oom_error, load_gpu_settings
from pymilvus import MilvusClient
import json

logger = logging.getLogger(__name__)


@dataclass
class BatchJobProgress:
    """Tracks progress of batch embedding job"""

    total_tracks: int = 0
    total_operations: int = 0  # tracks × models
    processed_tracks: int = 0
    processed_operations: int = 0
    failed_tracks: int = 0
    current_track: Optional[str] = None
    current_model: Optional[str] = None
    status: str = "initialized"  # running, completed, failed, cancelled
    started_at: float = 0.0
    estimated_completion: Optional[float] = None
    last_error: Optional[str] = None


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
        gpu_settings: Optional[GPUSettings] = None,
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
        # Resolve to absolute paths to avoid accidentally creating a new, empty
        # SQLite file when the working directory changes (e.g. running from
        # python_services/ would previously create python_services/navidrome.db).
        self.db_path = str(Path(db_path).expanduser().resolve())
        self.music_root = Path(music_root).expanduser().resolve()
        self.milvus_uri = milvus_uri
        self.checkpoint_interval = checkpoint_interval
        self.logger = logger or logging.getLogger(__name__)
        self.gpu_settings = gpu_settings or load_gpu_settings()

        if not Path(self.db_path).exists():
            raise FileNotFoundError(
                f"Navidrome database not found at {self.db_path}. "
                "Set NAVIDROME_DB_PATH or pass an explicit path."
            )

        self.progress = BatchJobProgress(
            total_tracks=0,
            total_operations=0,
            processed_tracks=0,
            processed_operations=0,
            failed_tracks=0,
            current_track=None,
            current_model=None,
            status="initialized",
            started_at=0,
            estimated_completion=None,
        )

        # Initialize models (lazy loading)
        self.models = {}
        self._cancelled = False
        self._paused = False

        self.audio_collection = DEFAULT_AUDIO_COLLECTION
        # Collection mapping for each model
        self.collection_map = {
            "muq": "embedding",
            "qwen3": "description_embedding",
            "flamingo_audio": self.audio_collection,
        }

    def pause(self) -> None:
        """Pause the job and unload any loaded models to free VRAM."""
        if self._paused:
            return
        self._paused = True
        self.progress.status = "paused"
        for model in self.models.values():
            try:
                if hasattr(model, "unload_model"):
                    model.unload_model()
            except Exception:
                pass
        self.logger.info("Job paused; models unloaded")

    def resume(self) -> None:
        """Resume a paused job (models will reload lazily)."""
        if not self._paused:
            return
        self._paused = False
        self.progress.status = "running"
        self.logger.info("Job resumed")

    def _initialize_models(self, model_names: List[str]) -> None:
        """Initialize embedding models."""
        for model_name in model_names:
            if model_name == "muq":
                self.logger.info("Initializing MuQ model...")
                self.models["muq"] = MuQEmbeddingModel(logger=self.logger)
            elif model_name == "qwen3":
                self.logger.info("Initializing description embedding pipeline...")
                self.models["qwen3"] = DescriptionEmbeddingPipeline(
                    logger=self.logger, gpu_settings=self.gpu_settings
                )
            elif model_name == "flamingo_audio":
                self.logger.info("Initializing Flamingo audio embedding pipeline...")
                self.models["flamingo_audio"] = DescriptionEmbeddingPipeline(
                    logger=self.logger, gpu_settings=self.gpu_settings
                )
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
            "qwen3": "description_embedding",
            "flamingo_audio": self.audio_collection,
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

            if model_name == "qwen3":
                try:
                    self.logger.info("Dropping collection: %s", self.audio_collection)
                    client.drop_collection(self.audio_collection)
                except Exception as e:
                    self.logger.warning(
                        "Failed to drop collection %s: %s",
                        self.audio_collection,
                        e,
                    )

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
        missing_only: bool = False,
    ) -> Dict:
        """
        Run batch embedding job with SEQUENTIAL model processing.

        This processes all tracks with one model at a time to avoid GPU overflow.
        Pattern: Load Model 1 -> Process all tracks -> Unload Model 1 -> Load Model 2 -> etc.

        Args:
            models_to_use: List of model names (default: all)
            clear_existing: Whether to clear existing embeddings first
            missing_only: If True, skip tracks that already exist in Milvus

        Returns:
            Dict with progress and failed tracks
        """
        if models_to_use is None:
            models_to_use = ["muq", "qwen3"]

        # Validate model names
        valid_models = {"muq", "qwen3", "flamingo_audio"}
        invalid_models = [m for m in models_to_use if m not in valid_models]
        if invalid_models:
            error_msg = (
                f"Invalid model names: {invalid_models}. Valid options: {valid_models}"
            )
            self.logger.error(error_msg)
            self.progress.status = "failed"
            raise ValueError(error_msg)

        if "qwen3" in models_to_use and "flamingo_audio" in models_to_use:
            self.logger.info(
                "flamingo_audio requested alongside qwen3; skipping standalone "
                "flamingo_audio since qwen3 already generates audio embeddings"
            )
            models_to_use = [m for m in models_to_use if m != "flamingo_audio"]

        self.logger.info(
            f"Starting batch embedding job with models: {models_to_use} "
            f"(missing_only={missing_only})"
        )
        self.progress.status = "running"

        # Initialize models
        self._initialize_models(models_to_use)

        # Get all tracks with proper error handling
        try:
            tracks = self.get_all_tracks()
        except Exception as e:
            self.logger.error(f"Failed to query tracks from database: {e}")
            self.progress.status = "failed"
            raise

        # Calculate total operations dynamically (important for missing_only)
        total_operations = 0
        self.progress.total_tracks = len(tracks)
        self.progress.total_operations = 0
        self.progress.processed_tracks = 0
        self.progress.processed_operations = 0
        self.progress.status = "running"
        self.progress.started_at = time.time()

        # Clear existing embeddings
        if clear_existing:
            self.clear_embeddings(models_to_use)

        # Process tracks sequentially by model
        failed_tracks = []
        operations_completed = 0
        batch_data = []  # For batched Milvus insertions
        BATCH_SIZE = 100  # Insert every 100 embeddings
        stop_due_to_error = False

        from pymilvus import MilvusClient

        client = None
        try:
            client = MilvusClient(uri=self.milvus_uri)

            # SEQUENTIAL MODEL PROCESSING - one model at a time to avoid GPU overflow
            for model_idx, model_name in enumerate(models_to_use):
                if self._cancelled:
                    self.progress.status = "cancelled"
                    self.logger.info("Job cancelled by user")
                    break
                if stop_due_to_error:
                    break

                self.logger.info(
                    f"Processing all tracks with model: {model_name} ({model_idx + 1}/{len(models_to_use)})"
                )
                self.progress.current_model = model_name
                model = self.models[model_name]
                collection = self.collection_map[model_name]
                tracks_processed_this_model = 0

                # Filter to missing-only if requested
                model_tracks = tracks
                if missing_only:
                    name_list = [self._canonical_name(t) for t in tracks]
                    existing = self._existing_names_for_collection(
                        client, collection, name_list
                    )
                    if model_name == "qwen3":
                        existing_audio = self._existing_names_for_collection(
                            client, self.audio_collection, name_list
                        )
                        complete = existing.intersection(existing_audio)
                        model_tracks = [
                            t
                            for t, name in zip(tracks, name_list)
                            if name not in complete
                        ]
                        self.logger.info(
                            "Missing-only enabled: %s tracks already present in %s and %s; %s remaining",
                            len(tracks) - len(model_tracks),
                            collection,
                            self.audio_collection,
                            len(model_tracks),
                        )
                    else:
                        model_tracks = [
                            t
                            for t, name in zip(tracks, name_list)
                            if name not in existing
                        ]
                        self.logger.info(
                            "Missing-only enabled: %s tracks already present in %s; %s remaining",
                            len(tracks) - len(model_tracks),
                            collection,
                            len(model_tracks),
                        )
                total_operations += len(model_tracks)
                self.progress.total_operations = total_operations

                # Ensure model is loaded
                model.ensure_model_loaded()

                try:
                    if model_name == "qwen3" and getattr(model, "caption_only", False):
                        ops_done = self._process_qwen3_two_stage(
                            model_tracks=model_tracks,
                            model=model,
                            collection=collection,
                            audio_collection=model.collection_audio,
                            client=client,
                            batch_size=BATCH_SIZE,
                        )
                        operations_completed += ops_done
                        tracks_processed_this_model += len(model_tracks)
                        self.progress.processed_operations = operations_completed
                    elif model_name == "flamingo_audio":
                        ops_done = self._process_flamingo_audio(
                            model_tracks=model_tracks,
                            model=model,
                            collection=collection,
                            client=client,
                            batch_size=BATCH_SIZE,
                        )
                        operations_completed += ops_done
                        tracks_processed_this_model += len(model_tracks)
                        self.progress.processed_operations = operations_completed
                    else:
                        # Process all tracks with this model
                        for track_idx, track in enumerate(
                            tqdm(model_tracks, desc=f"Embedding with {model_name}")
                        ):
                            if self._cancelled:
                                self.progress.status = "cancelled"
                                self.logger.info("Job cancelled by user")
                                break
                            if stop_due_to_error:
                                break

                            # Handle pauses
                            if self._paused:
                                try:
                                    model.unload_model()
                                except Exception:
                                    pass
                                while self._paused and not self._cancelled:
                                    time.sleep(0.5)
                                if self._cancelled:
                                    break
                                model.ensure_model_loaded()

                            self.progress.current_track = (
                                f"{track['artist']} - {track['title']}"
                            )

                            try:
                                # Process track and add to batch
                                embedding_data = self._process_track_with_model_batched(
                                    track, model_name, model, collection
                                )
                                batch_data.extend(embedding_data)

                                # Insert batch if it's large enough
                                if len(batch_data) >= BATCH_SIZE:
                                    client.insert(
                                        collection_name=collection, data=batch_data
                                    )
                                    batch_data = []

                                operations_completed += 1
                                tracks_processed_this_model += 1
                                self.progress.processed_operations = (
                                    operations_completed
                                )
                            except Exception as e:
                                if is_oom_error(e):
                                    message = (
                                        f"CUDA out of memory while processing "
                                        f"{track.get('title') or track.get('id')} "
                                        f"with {model_name}. Lower the GPU memory cap "
                                        "or enable CPU offload in batch settings."
                                    )
                                    self.logger.error(message)
                                    self.progress.status = "failed"
                                    self.progress.last_error = message
                                    stop_due_to_error = True
                                    try:
                                        model.unload_model()
                                    except Exception:
                                        pass
                                    try:
                                        import torch

                                        torch.cuda.empty_cache()
                                    except Exception:
                                        pass
                                    break
                                self.logger.error(
                                    f"Failed to process track {track['id']} with {model_name}: {e}"
                                )
                                failed_tracks.append((track["id"], model_name, str(e)))
                                self.progress.failed_tracks += 1
                                self.progress.last_error = str(e)

                            # Update estimated completion every 10 operations
                            if (
                                operations_completed > 0
                                and operations_completed % 10 == 0
                            ):
                                elapsed = time.time() - self.progress.started_at
                                rate = elapsed / operations_completed
                                remaining = total_operations - operations_completed
                                self.progress.estimated_completion = time.time() + (
                                    rate * remaining
                                )

                    # Insert remaining batch for this model
                    if batch_data:
                        client.insert(collection_name=collection, data=batch_data)
                        batch_data = []

                    # CRITICAL: Explicitly unload model after processing all tracks
                    self.logger.info(f"Unloading {model_name} model to free GPU memory")
                    model.unload_model()

                except Exception as e:
                    if is_oom_error(e):
                        message = (
                            f"CUDA out of memory while running model {model_name}. "
                            "Reduce GPU memory target or enable CPU offload."
                        )
                        self.logger.error(message)
                        self.progress.status = "failed"
                        self.progress.last_error = message
                        stop_due_to_error = True
                    else:
                        self.logger.error(f"Failed during {model_name} processing: {e}")
                    # Insert any remaining batch data before failing
                    if batch_data:
                        try:
                            client.insert(collection_name=collection, data=batch_data)
                            batch_data = []
                        except Exception as insert_error:
                            self.logger.error(
                                f"Failed to insert final batch: {insert_error}"
                            )

                    # Ensure model is unloaded even on error
                    try:
                        model.unload_model()
                    except Exception as unload_error:
                        self.logger.error(
                            f"Failed to unload {model_name}: {unload_error}"
                        )

                    # Don't re-raise - continue with next model
                    self.progress.status = "failed"
                    self.logger.error(
                        f"Continuing to next model after failure in {model_name}"
                    )

        finally:
            # Always close Milvus client
            if client is not None:
                try:
                    del client
                except Exception as e:
                    self.logger.error(f"Error closing Milvus client: {e}")

        # Finalize
        if self._cancelled:
            self.progress.status = "cancelled"
        elif self.progress.status != "failed" and self.progress.failed_tracks > 0:
            self.progress.status = "completed_with_errors"
            # Set processed_tracks to total since we attempted all tracks
            self.progress.processed_tracks = len(tracks)
        elif self.progress.status != "failed":
            self.progress.status = "completed"
            # Set processed_tracks to total since all tracks were processed
            self.progress.processed_tracks = len(tracks)

        elapsed = time.time() - self.progress.started_at
        self.logger.info(
            f"Job completed in {elapsed:.1f}s: "
            f"{operations_completed}/{total_operations} operations "
            f"({self.progress.processed_tracks}/{len(tracks)} tracks × {len(models_to_use)} models)"
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

    def _process_track_with_model(
        self, track: Dict, model_name: str, model, collection: str, client
    ) -> None:
        """Process a single track with a specific model (DEPRECATED - use _process_track_with_model_batched for better performance)."""
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
            row = {
                "name": segment["title"],
                "embedding": segment["embedding"],
                "offset": segment.get("offset_seconds", 0.0),
                "model_id": result["model_id"],
            }
            if "description" in segment:
                row["description"] = segment.get("description", "")
            client.insert(collection_name=collection, data=[row])

        if self.logger.level <= logging.DEBUG:
            self.logger.debug(
                f"Embedded {canonical_name} with {model_name} "
                f"({len(result['segments'])} segments)"
            )

    def _process_track_with_model_batched(
        self, track: Dict, model_name: str, model, collection: str
    ) -> List[Dict]:
        """Process a single track with a specific model and return data for batched insertion."""
        # Resolve audio file path
        audio_path = self.music_root / track["path"]
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Normalize track name (artist - title format)
        canonical_name = self._canonical_name(track)

        # Generate embedding
        result = model.embed_music(str(audio_path), canonical_name)

        # Prepare data for batched insertion
        batch_data = []
        for segment in result["segments"]:
            row = {
                "name": segment["title"],
                "embedding": segment["embedding"],
                "offset": segment.get("offset_seconds", 0.0),
                "model_id": result["model_id"],
            }
            if "description" in segment:
                row["description"] = segment.get("description", "")
            batch_data.append(row)

        if self.logger.level <= logging.DEBUG:
            self.logger.debug(
                f"Embedded {canonical_name} with {model_name} "
                f"({len(result['segments'])} segments)"
            )

        return batch_data

    def _process_qwen3_two_stage(
        self,
        model_tracks: List[Dict],
        model: DescriptionEmbeddingPipeline,
        collection: str,
        audio_collection: str,
        client,
        batch_size: int,
    ) -> int:
        """
        Two-stage pipeline for qwen3:
        1) Run Music Flamingo captions for all tracks (fp16 on GPU), writing descriptions to JSON.
        2) Unload Flamingo, load Qwen3, embed all descriptions, and insert into Milvus.
        Returns number of completed embedding operations.
        """
        # Stage 1: captions
        caption_results = []
        audio_batch_data: List[Dict] = []
        audio_schema_ready = False
        for track in tqdm(model_tracks, desc="Flamingo captions"):
            audio_path = self.music_root / track["path"]
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            canonical_name = self._canonical_name(track)
            try:
                description, audio_embedding = model.get_caption(
                    str(audio_path), canonical_name
                )
            except Exception as e:
                if is_oom_error(e):
                    raise
                # Store minimal fallback description and continue
                self.logger.warning(
                    "Caption failed for %s, using fallback: %s", canonical_name, e
                )
                description = f"Audio track titled '{canonical_name}'."
                audio_embedding = []

            if audio_embedding:
                if not audio_schema_ready:
                    try:
                        model.ensure_milvus_schemas(
                            client, audio_dim=len(audio_embedding)
                        )
                        model.ensure_milvus_index(client)
                        audio_schema_ready = True
                    except Exception as e:
                        self.logger.warning(
                            "Failed to ensure Flamingo audio schema/index: %s", e
                        )
                audio_batch_data.append(
                    {
                        "name": canonical_name,
                        "embedding": audio_embedding,
                        "offset": 0.0,
                        "model_id": model.caption_model_id,
                    }
                )
                if len(audio_batch_data) >= batch_size:
                    client.insert(
                        collection_name=audio_collection, data=audio_batch_data
                    )
                    audio_batch_data = []

            caption_results.append((canonical_name, description))

        # Free Flamingo before embedding
        model.unload_captioner()

        if audio_batch_data:
            client.insert(collection_name=audio_collection, data=audio_batch_data)

        # Stage 2: embeddings
        batch_data: List[Dict] = []
        operations_completed = 0
        for canonical_name, description in tqdm(
            caption_results, desc="Qwen3 embeddings"
        ):
            segment = model.embed_description(description, canonical_name)
            row = {
                "name": segment.title,
                "embedding": segment.description_embedding,
                "offset": segment.offset_seconds,
                "model_id": model.text_model_id,
                "description": segment.description,
            }
            batch_data.append(row)
            operations_completed += 1

            if len(batch_data) >= batch_size:
                client.insert(collection_name=collection, data=batch_data)
                batch_data = []

        if batch_data:
            client.insert(collection_name=collection, data=batch_data)

        return operations_completed

    def _process_flamingo_audio(
        self,
        model_tracks: List[Dict],
        model: DescriptionEmbeddingPipeline,
        collection: str,
        client,
        batch_size: int,
    ) -> int:
        """
        Generate Flamingo audio embeddings only (no Qwen text embeddings).

        Returns number of completed embedding operations.
        """
        batch_data: List[Dict] = []
        operations_completed = 0
        audio_schema_ready = False

        for track in tqdm(model_tracks, desc="Flamingo audio"):
            audio_path = self.music_root / track["path"]
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            canonical_name = self._canonical_name(track)
            try:
                _description, audio_embedding = model.get_caption(
                    str(audio_path), canonical_name
                )
            except Exception as e:
                if is_oom_error(e):
                    raise
                self.logger.warning(
                    "Audio embedding failed for %s, skipping: %s", canonical_name, e
                )
                audio_embedding = []

            if not audio_embedding:
                continue

            if not audio_schema_ready:
                try:
                    model.ensure_milvus_schemas(client, audio_dim=len(audio_embedding))
                    model.ensure_milvus_index(client)
                    audio_schema_ready = True
                except Exception as e:
                    self.logger.warning(
                        "Failed to ensure Flamingo audio schema/index: %s", e
                    )

            batch_data.append(
                {
                    "name": canonical_name,
                    "embedding": audio_embedding,
                    "offset": 0.0,
                    "model_id": model.caption_model_id,
                }
            )
            operations_completed += 1

            if len(batch_data) >= batch_size:
                client.insert(collection_name=collection, data=batch_data)
                batch_data = []

        if batch_data:
            client.insert(collection_name=collection, data=batch_data)

        return operations_completed

    def _canonical_name(self, track: Dict) -> str:
        artist = (
            str(track["artist"]).replace("•", "&").replace("/", "_").replace("\\", "_")
        )
        title = (
            str(track["title"]).replace("•", "&").replace("/", "_").replace("\\", "_")
        )
        return f"{artist} - {title}".strip()

    def _existing_names_for_collection(
        self, client: MilvusClient, collection: str, names: List[str]
    ) -> set:
        """
        Return a set of names already present in the given collection.
        Uses batch queries to avoid huge filters; assumes 'name' field exists.
        """
        existing = set()
        if not names:
            return existing

        BATCH = 256
        lite_mode = "://" not in self.milvus_uri or self.milvus_uri.startswith("file:")
        for i in range(0, len(names), BATCH):
            chunk = names[i : i + BATCH]
            try:
                if lite_mode:
                    filter_expr = f"name in {json.dumps(chunk)}"
                    rows = client.query(
                        collection_name=collection,
                        filter=filter_expr,
                        output_fields=["name"],
                    )
                else:
                    rows = client.query(
                        collection_name=collection,
                        filter="name in {names}",
                        filter_params={"names": chunk},
                        output_fields=["name"],
                    )
                for row in rows or []:
                    if "name" in row:
                        existing.add(str(row["name"]))
            except Exception:
                # Best-effort; if query fails, skip filtering for this batch
                self.logger.exception(
                    "Failed to query existing names for %s", collection
                )
        return existing

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
    gpu_settings: Optional[GPUSettings] = None,
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
        db_path, music_root, milvus_uri, checkpoint_interval, gpu_settings=gpu_settings
    )
    return _current_job


def get_current_job() -> Optional[BatchEmbeddingJob]:
    """Get the current batch embedding job (if any)."""
    return _current_job
