"""
Text Embedding Service

Provides REST API for embedding text queries into audio embedding spaces.
Supports multiple models (MuQ, MERT, Latent) and can use either:
1. Trained text-to-audio projection models (from inference.py)
2. Stub embedders for development/testing

The service automatically falls back to stubs when trained models are not available.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Try importing real models, fall back to stubs
try:
    from inference import TextToAudioEmbedder

    REAL_MODELS_AVAILABLE = True
except ImportError:
    logger.warning("inference.py not available, will use stub embedders only")
    REAL_MODELS_AVAILABLE = False

from stub_text_embedders import (  # noqa: E402
    get_stub_embedder,
    StubTextEmbedder,
)


class TextEmbeddingRequest(BaseModel):
    """Request to embed text into audio space"""

    text: str = Field(..., description="Text query to embed", min_length=1)
    model: str = Field(
        default="muq",
        description="Embedding model to use",
        pattern="^(muq|mert|latent)$",
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
    """
    Manages text-to-audio embedding models.

    Supports both trained projection models and stub embedders.
    Automatically falls back to stubs when trained models are unavailable.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        use_stubs: bool = False,
        device: Optional[str] = None,
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

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.embedders: Dict[str, object] = {}

        logger.info("Initialized TextEmbeddingService")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Force stubs: {self.use_stubs}")

    def get_embedder(self, model_name: str):
        """
        Get or create embedder for specified model.

        Tries to load real model from checkpoint, falls back to stub if unavailable.

        Args:
            model_name: Model name ("muq", "mert", or "latent")

        Returns:
            Embedder instance (TextToAudioEmbedder or StubTextEmbedder)
        """
        if model_name not in self.embedders:
            # Try loading real model first (unless forced to use stubs)
            if not self.use_stubs and REAL_MODELS_AVAILABLE:
                checkpoint_path = self.checkpoint_dir / f"{model_name}_best_r1.pt"

                if checkpoint_path.exists():
                    try:
                        logger.info(
                            f"Loading real model for {model_name} from "
                            f"{checkpoint_path}"
                        )
                        self.embedders[model_name] = TextToAudioEmbedder(
                            str(checkpoint_path), device=self.device
                        )
                        logger.info(f"Successfully loaded real {model_name} model")
                        return self.embedders[model_name]
                    except Exception as e:
                        logger.warning(
                            f"Failed to load real model for {model_name}: {e}. "
                            f"Falling back to stub."
                        )

            # Fall back to stub
            logger.info(f"Using stub embedder for {model_name}")
            self.embedders[model_name] = get_stub_embedder(model_name)

        return self.embedders[model_name]

    def is_stub(self, model_name: str) -> bool:
        """Check if model is using stub implementation"""
        embedder = self.embedders.get(model_name)
        return isinstance(embedder, StubTextEmbedder)

    def embed_text(self, text: str, model: str = "muq") -> np.ndarray:
        """
        Embed text query into audio space.

        Args:
            text: Text query
            model: Model name ("muq", "mert", or "latent")

        Returns:
            Embedding vector as numpy array

        Raises:
            ValueError: If model name is invalid
            RuntimeError: If embedding fails
        """
        if model not in ["muq", "mert", "latent"]:
            raise ValueError(f"Invalid model: {model}")

        try:
            embedder = self.get_embedder(model)
            embedding = embedder.embed_text(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed text with {model}: {e}")
            raise RuntimeError(f"Embedding failed: {str(e)}")

    def get_model_info(self) -> List[ModelInfo]:
        """Get information about all available models"""
        models = []

        for model_name in ["muq", "mert", "latent"]:
            # Check if real model checkpoint exists
            checkpoint_path = self.checkpoint_dir / f"{model_name}_best_r1.pt"
            has_checkpoint = checkpoint_path.exists()

            # Get dimension based on model
            dimensions = {"muq": 1536, "mert": 76_800, "latent": 576}
            dimension = dimensions[model_name]

            # Determine status
            if self.use_stubs:
                status = "stub"
            elif not REAL_MODELS_AVAILABLE:
                status = "stub"
            elif has_checkpoint:
                status = "available"
            else:
                status = "stub"

            descriptions = {
                "muq": "MuQ-MuLan model (default, balanced performance)",
                "mert": "MERT model (high dimensionality, detailed features)",
                "latent": "Music2Latent model (compact representation)",
            }

            models.append(
                ModelInfo(
                    name=model_name,
                    dimension=dimension,
                    status=status,
                    description=descriptions[model_name],
                )
            )

        return models


# FastAPI application
app = FastAPI(
    title="Text Embedding Service",
    description="Embed text queries into audio embedding spaces",
    version="1.0.0",
)

# Initialize service
checkpoint_dir = os.getenv("TEXT_EMBEDDING_CHECKPOINT_DIR", "checkpoints")
use_stubs = os.getenv("TEXT_EMBEDDING_USE_STUBS", "false").lower() == "true"
device = os.getenv("TEXT_EMBEDDING_DEVICE", None)

text_service = TextEmbeddingService(
    checkpoint_dir=checkpoint_dir, use_stubs=use_stubs, device=device
)


@app.post("/embed_text", response_model=TextEmbeddingResponse)
async def embed_text(request: TextEmbeddingRequest) -> TextEmbeddingResponse:
    """
    Embed text into audio embedding space.

    This endpoint projects a text query into the audio embedding space,
    allowing similarity search against audio embeddings stored in Milvus.
    """
    try:
        embedding = text_service.embed_text(request.text, request.model)

        # Check if stub was used
        is_stub = text_service.is_stub(request.model)

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


@app.get("/models", response_model=List[ModelInfo])
async def list_models() -> List[ModelInfo]:
    """
    List available text embedding models.

    Returns information about each model including:
    - Name
    - Embedding dimensionality
    - Status (available, stub, or unavailable)
    - Description
    """
    return text_service.get_model_info()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": text_service.device,
        "checkpoint_dir": str(text_service.checkpoint_dir),
        "use_stubs": text_service.use_stubs,
    }


# Development/testing entry point
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("TEXT_EMBEDDING_PORT", "9003"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Starting Text Embedding Service on port {port}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Using stubs: {use_stubs}")

    uvicorn.run(app, host="0.0.0.0", port=port)
