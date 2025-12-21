"""
Text Embedding Service

Provides REST API for embedding text queries into embedding spaces.
Supported models:
- qwen3 : Qwen3-Embedding-8B text embeddings (real model)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from description_pipeline import DescriptionEmbeddingPipeline, _HAS_MUSIC_FLAMINGO
from gpu_settings import GPUSettings, is_oom_error, load_gpu_settings
from stub_text_embedders import (  # noqa: E402
    get_stub_embedder,
    StubTextEmbedder,
)

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> str:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return "debug" if verbose else "info"


class TextEmbeddingRequest(BaseModel):
    """Request to embed text into audio space"""

    text: str = Field(..., description="Text query to embed", min_length=1)
    model: str = Field(
        default="qwen3",
        description="Embedding model to use",
        pattern="^(qwen3)$",
    )


class TextEmbeddingResponse(BaseModel):
    """Response containing embedded text"""

    embedding: List[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Model used for embedding")
    dimension: int = Field(..., description="Embedding dimensionality")
    is_stub: bool = Field(..., description="Whether stub embedder was used")


class ModelInfo(BaseModel):
    """Information about an available model"""

    name: str
    dimension: int
    status: str  # "available", "stub", "unavailable"
    description: str


class TextEmbeddingService:
    """Manages text embedding models for recommendation endpoints."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        use_stubs: bool = False,
        device: Optional[str] = None,
        gpu_settings: Optional[GPUSettings] = None,
    ):
        """
        Initialize text embedding service.

        Args:
            checkpoint_dir: Directory containing trained model checkpoints
            use_stubs: Force use of stub embedders even if real models available
            device: Device to use for models ('cuda', 'cpu', or None for auto)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.use_stubs = use_stubs
        self.gpu_settings = gpu_settings or load_gpu_settings()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.embedders: Dict[str, object] = {}
        self.description_pipeline: Optional[DescriptionEmbeddingPipeline] = None

        logger.info("Initialized TextEmbeddingService")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Force stubs: {self.use_stubs}")

    def get_embedder(self, model_name: str):
        """
        Get or create embedder for specified model.

        Args:
            model_name: Model name ("qwen3")

        Returns:
            Embedder instance
        """
        if model_name not in self.embedders:
            if self.use_stubs:
                logger.info("Using stub embedder for qwen3 (stubs enabled)")
                self.embedders[model_name] = get_stub_embedder("qwen3")
            else:
                try:
                    if self.description_pipeline is None:
                        self.description_pipeline = DescriptionEmbeddingPipeline(
                            device=self.device,
                            logger=logger,
                            gpu_settings=self.gpu_settings,
                        )
                    self.embedders[model_name] = self.description_pipeline
                except Exception as exc:
                    logger.warning("Falling back to stub qwen3 embedder: %s", exc)
                    self.embedders[model_name] = get_stub_embedder("qwen3")

        return self.embedders[model_name]

    def is_stub(self, model_name: str) -> bool:
        """Check if model is using stub implementation"""
        embedder = self.embedders.get(model_name)
        return isinstance(embedder, StubTextEmbedder)

    def embed_text(self, text: str, model: str = "qwen3") -> np.ndarray:
        """
        Embed text query into an embedding space.

        Args:
            text: Text query
            model: Model name ("qwen3")

        Returns:
            Embedding vector as numpy array

        Raises:
            ValueError: If model name is invalid
            RuntimeError: If embedding fails
        """
        if model not in ["qwen3"]:
            raise ValueError(f"Invalid model: {model}")

        try:
            embedder = self.get_embedder(model)
            if isinstance(embedder, DescriptionEmbeddingPipeline):
                return np.array(embedder.embed_text(text))
            embedding = embedder.embed_text(text)
            return embedding
        except Exception as e:
            if is_oom_error(e):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                message = (
                    f"CUDA out of memory while embedding with {model}. "
                    "Lower the GPU memory cap or enable CPU offload in batch settings."
                )
            else:
                message = f"Embedding failed: {str(e)}"
            logger.error(message)
            raise RuntimeError(message)

    def get_model_info(self) -> List[ModelInfo]:
        """Get information about all available models"""
        models = []

        qwen3_status = "stub" if self.use_stubs else "available"
        models.append(
            ModelInfo(
                name="qwen3",
                dimension=4096,
                status=qwen3_status,
                description="Qwen3-Embedding-8B text embeddings for caption search",
            )
        )

        return models


def build_text_embedding_router(service: TextEmbeddingService) -> APIRouter:
    router = APIRouter()

    @router.post("/embed_text", response_model=TextEmbeddingResponse)
    async def embed_text_endpoint(
        request: TextEmbeddingRequest,
    ) -> TextEmbeddingResponse:
        """
        Embed text into audio embedding space.

        This endpoint projects a text query into the audio embedding space,
        allowing similarity search against audio embeddings stored in Milvus.
        """
        try:
            embedding = service.embed_text(request.text, request.model)

            # Check if stub was used
            is_stub = service.is_stub(request.model)

            return TextEmbeddingResponse(
                embedding=embedding.tolist(),
                model=request.model,
                dimension=len(embedding),
                is_stub=is_stub,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/models", response_model=List[ModelInfo])
    async def list_models_endpoint() -> List[ModelInfo]:
        """
        List available text embedding models.

        Returns information about each model including:
        - Name
        - Embedding dimensionality
        - Status (available, stub, or unavailable)
        - Description
        """
        return service.get_model_info()

    @router.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "device": service.device,
            "checkpoint_dir": str(service.checkpoint_dir),
            "use_stubs": service.use_stubs,
            "gpu_settings": {
                "max_gpu_memory_gb": service.gpu_settings.max_gpu_memory_gb,
                "precision": service.gpu_settings.precision,
                "enable_cpu_offload": service.gpu_settings.enable_cpu_offload,
            },
        }

    return router


# Initialize service
checkpoint_dir = os.getenv("TEXT_EMBEDDING_CHECKPOINT_DIR", "checkpoints")
use_stubs_env = os.getenv("TEXT_EMBEDDING_USE_STUBS", "auto").lower()
if use_stubs_env == "true":
    use_stubs = True
elif use_stubs_env == "false":
    use_stubs = False
else:
    # Auto-mode: use stubs when heavy model dependencies are unavailable
    use_stubs = not _HAS_MUSIC_FLAMINGO
device = os.getenv("TEXT_EMBEDDING_DEVICE", None)
gpu_settings = load_gpu_settings()

text_service = TextEmbeddingService(
    checkpoint_dir=checkpoint_dir,
    use_stubs=use_stubs,
    device=device,
    gpu_settings=gpu_settings,
)

# FastAPI application
app = FastAPI(
    title="Text Embedding Service",
    description="Embed text queries with MuQ or Qwen3 embeddings",
    version="1.0.0",
)

app.include_router(build_text_embedding_router(text_service))


# Development/testing entry point
if __name__ == "__main__":
    import argparse
    import uvicorn

    port = int(os.getenv("TEXT_EMBEDDING_PORT", "9003"))
    parser = argparse.ArgumentParser(description="Navidrome Text Embedding Service")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args, _unknown = parser.parse_known_args()
    uvicorn_log_level = _configure_logging(args.verbose)

    logger.info(f"Starting Text Embedding Service on port {port}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Using stubs: {use_stubs}")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level=uvicorn_log_level)
