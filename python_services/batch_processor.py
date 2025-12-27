"""
Model-optimized batch processor for embedding requests.

Processes all requests with each model before switching to minimize GPU swaps.
Instead of processing each track through the full pipeline (MuQ → Flamingo → Qwen3)
before moving to the next track, this processor:

1. Processes ALL tracks with MuQ
2. Then processes ALL tracks with Flamingo
3. Then processes ALL tracks with Qwen3

This reduces GPU swaps from O(N*3) to O(3).
"""

import base64
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from batch_queue_manager import BatchResult

if TYPE_CHECKING:
    from description_pipeline import DescriptionEmbeddingPipeline
    from embedding_models import MuQEmbeddingModel


@dataclass
class TrackContext:
    """Intermediate state for a track being processed through the pipeline."""

    request_id: str
    music_file: str
    music_name: str
    payload: dict
    temp_paths: List[Path] = field(default_factory=list)

    # Intermediate results
    muq_result: Optional[dict] = None
    caption: Optional[str] = None
    audio_embedding: Optional[List[float]] = None
    description_embedding: Optional[List[float]] = None
    error: Optional[str] = None


class ModelFirstBatchProcessor:
    """
    Processes batches of embedding requests in model-first order:
    1. All tracks through MuQ (audio embedding)
    2. All tracks through Flamingo (caption + audio embedding)
    3. All tracks through Qwen3 (text embedding)

    This minimizes GPU model swaps from O(N*3) to O(3).
    """

    def __init__(
        self,
        *,
        muq_model: "MuQEmbeddingModel",
        description_pipeline: Optional["DescriptionEmbeddingPipeline"],
        logger: Optional[logging.Logger] = None,
    ):
        self.muq_model = muq_model
        self.description_pipeline = description_pipeline
        self.logger = logger or logging.getLogger(__name__)

    def process_batch(self, payloads: List[dict]) -> Dict[str, BatchResult]:
        """
        Process a batch of requests, returning results keyed by request_id.

        Args:
            payloads: List of embedding request payloads

        Returns:
            Dict mapping request_id to BatchResult
        """
        start_time = time.monotonic()

        # Prepare all track contexts
        contexts: List[TrackContext] = []
        for payload in payloads:
            try:
                ctx = self._prepare_context(payload)
                contexts.append(ctx)
            except Exception as e:
                # Immediate failure for this request
                request_id = payload.get("request_id", f"unknown-{id(payload)}")
                self.logger.error("Failed to prepare context for %s: %s", request_id, e)
                contexts.append(
                    TrackContext(
                        request_id=request_id,
                        music_file="",
                        music_name="",
                        payload=payload,
                        error=str(e),
                    )
                )

        results: Dict[str, BatchResult] = {}

        try:
            # Stage 1: MuQ audio embeddings for all tracks
            self._stage_muq(contexts)

            # Stage 2 & 3: Flamingo captions and Qwen3 embeddings
            if self.description_pipeline:
                self._stage_flamingo(contexts)
                self._stage_qwen3(contexts)

            # Build final results
            for ctx in contexts:
                results[ctx.request_id] = self._build_result(ctx)

        finally:
            # Cleanup temp files
            for ctx in contexts:
                for path in ctx.temp_paths:
                    try:
                        path.unlink(missing_ok=True)
                        if path.parent.name.startswith("navidrome-batch-"):
                            try:
                                path.parent.rmdir()
                            except OSError:
                                pass  # Directory not empty
                    except Exception:
                        pass

        elapsed = time.monotonic() - start_time
        self.logger.info(
            "Batch of %d tracks processed in %.1fs (%.2fs/track)",
            len(contexts),
            elapsed,
            elapsed / max(len(contexts), 1),
        )

        return results

    def _prepare_context(self, payload: dict) -> TrackContext:
        """Validate payload and prepare track context."""
        music_file = payload.get("music_file")
        if not music_file:
            raise ValueError("music_file is required")

        request_id = payload.get("request_id", f"batch-{id(payload)}")
        artist = str(payload.get("artist") or "")
        title = str(payload.get("title") or "")
        base_name = payload.get("name") or Path(str(music_file)).name

        # Compute canonical name (same logic as EmbedSocketServer)
        if artist or title:
            music_name = f"{artist} - {title}".strip(" -")
        else:
            music_name = str(base_name)

        temp_paths: List[Path] = []

        # Materialize base64 data if path not accessible
        music_file = self._ensure_local_file(
            music_file,
            payload.get("music_data_b64"),
            suggested_name=music_name,
            created_paths=temp_paths,
        )

        return TrackContext(
            request_id=request_id,
            music_file=music_file,
            music_name=music_name,
            payload=payload,
            temp_paths=temp_paths,
        )

    def _ensure_local_file(
        self,
        path: str,
        b64_data: Optional[str],
        *,
        suggested_name: Optional[str],
        created_paths: List[Path],
    ) -> str:
        """Ensure file is accessible locally, materializing from base64 if needed."""
        candidate = Path(path)
        if candidate.exists():
            return str(candidate)

        if not b64_data:
            raise FileNotFoundError(f"File not found: {candidate}")

        data = base64.b64decode(b64_data)
        suffix = Path(suggested_name or candidate.name or "upload").suffix or ".audio"
        temp_dir = Path(tempfile.mkdtemp(prefix="navidrome-batch-"))
        temp_path = temp_dir / f"upload{suffix}"
        temp_path.write_bytes(data)
        created_paths.append(temp_path)
        return str(temp_path)

    def _stage_muq(self, contexts: List[TrackContext]) -> None:
        """Process all tracks through MuQ model."""
        valid_contexts = [c for c in contexts if not c.error]
        if not valid_contexts:
            return

        self.logger.info("Stage 1/3: MuQ embeddings for %d tracks", len(valid_contexts))
        stage_start = time.monotonic()

        # Ensure model is loaded (claims GPU)
        self.muq_model.ensure_model_loaded()

        for i, ctx in enumerate(valid_contexts, 1):
            try:
                ctx.muq_result = self.muq_model.embed_music(
                    ctx.music_file,
                    ctx.music_name,
                )
                if i % 10 == 0 or i == len(valid_contexts):
                    self.logger.debug("MuQ progress: %d/%d", i, len(valid_contexts))
            except Exception as e:
                self.logger.error("MuQ failed for %s: %s", ctx.music_name, e)
                ctx.error = f"MuQ embedding failed: {e}"

        elapsed = time.monotonic() - stage_start
        self.logger.info(
            "Stage 1 complete: %d tracks in %.1fs", len(valid_contexts), elapsed
        )

    def _stage_flamingo(self, contexts: List[TrackContext]) -> None:
        """Process all tracks through Flamingo captioner."""
        if not self.description_pipeline:
            return

        valid_contexts = [c for c in contexts if not c.error]
        if not valid_contexts:
            return

        self.logger.info(
            "Stage 2/3: Flamingo captions for %d tracks", len(valid_contexts)
        )
        stage_start = time.monotonic()

        # Get captioner (claims GPU, offloads MuQ via GPU_COORDINATOR)
        captioner = self.description_pipeline._get_captioner()

        for i, ctx in enumerate(valid_contexts, 1):
            try:
                caption, audio_embedding = captioner.generate(ctx.music_file)
                ctx.caption = caption
                ctx.audio_embedding = audio_embedding

                # Log caption preview
                preview = caption[:200] + "..." if len(caption) > 200 else caption
                self.logger.debug(
                    "Caption for '%s': %s",
                    ctx.music_name[:50],
                    preview.replace("\n", " "),
                )

                if i % 10 == 0 or i == len(valid_contexts):
                    self.logger.debug(
                        "Flamingo progress: %d/%d", i, len(valid_contexts)
                    )
            except Exception as e:
                self.logger.error("Flamingo failed for %s: %s", ctx.music_name, e)
                # Use fallback caption - don't fail the track entirely
                ctx.caption = f"Audio track titled '{ctx.music_name}'."
                ctx.audio_embedding = []

        # Unload Flamingo before loading Qwen3
        self.description_pipeline.unload_captioner()

        elapsed = time.monotonic() - stage_start
        self.logger.info(
            "Stage 2 complete: %d tracks in %.1fs", len(valid_contexts), elapsed
        )

    def _stage_qwen3(self, contexts: List[TrackContext]) -> None:
        """Process all captions through Qwen3 embedder."""
        if not self.description_pipeline:
            return

        valid_contexts = [c for c in contexts if not c.error and c.caption]
        if not valid_contexts:
            return

        self.logger.info(
            "Stage 3/3: Qwen3 embeddings for %d tracks", len(valid_contexts)
        )
        stage_start = time.monotonic()

        # Get embedder (claims GPU)
        embedder = self.description_pipeline._get_embedder()

        for i, ctx in enumerate(valid_contexts, 1):
            try:
                embedding = embedder.embed_text(ctx.caption)
                ctx.description_embedding = embedding.cpu().tolist()

                if i % 10 == 0 or i == len(valid_contexts):
                    self.logger.debug("Qwen3 progress: %d/%d", i, len(valid_contexts))
            except Exception as e:
                self.logger.error("Qwen3 failed for %s: %s", ctx.music_name, e)
                # Non-fatal: track still has MuQ embedding and caption
                ctx.description_embedding = None

        elapsed = time.monotonic() - stage_start
        self.logger.info(
            "Stage 3 complete: %d tracks in %.1fs", len(valid_contexts), elapsed
        )

    def _build_result(self, ctx: TrackContext) -> BatchResult:
        """Build final result from track context."""
        if ctx.error:
            return BatchResult(
                request_id=ctx.request_id,
                success=False,
                payload={"status": "error", "message": ctx.error},
                error=ctx.error,
            )

        # Combine MuQ and description results
        combined_payload: dict = {
            "status": "ok",
            "music_name": ctx.music_name,
        }

        if ctx.muq_result:
            combined_payload["segments"] = ctx.muq_result.get("segments", [])
            combined_payload["model_id"] = ctx.muq_result.get("model_id", "")
            combined_payload["sample_rate"] = ctx.muq_result.get("sample_rate")
            combined_payload["window_seconds"] = ctx.muq_result.get("window_seconds")
            combined_payload["hop_seconds"] = ctx.muq_result.get("hop_seconds")

        if ctx.caption or ctx.audio_embedding or ctx.description_embedding:
            combined_payload["descriptions"] = [
                {
                    "title": ctx.music_name,
                    "description": ctx.caption or "",
                    "embedding": ctx.description_embedding or [],
                    "audio_embedding": ctx.audio_embedding or [],
                    "offset_seconds": 0.0,
                }
            ]

        return BatchResult(
            request_id=ctx.request_id,
            success=True,
            payload=combined_payload,
        )
