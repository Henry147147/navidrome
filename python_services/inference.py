#!/usr/bin/env python3
"""
inference.py - Inference script for trained text-to-audio projection models

This script provides utilities to:
1. Load trained projection models
2. Embed text queries into audio embedding spaces
3. Perform similarity search (optional Milvus integration)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """Text encoder (same as training)"""

    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, -1, :]

        return embeddings


class ProjectionHead(nn.Module):
    """Projection head (same as training)"""

    def __init__(self, text_dim: int, hidden_dim: int, audio_dim: int):
        super().__init__()
        self.text_dim = text_dim
        self.audio_dim = audio_dim

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, audio_dim),
            nn.BatchNorm1d(audio_dim),
        )

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        """Project and normalize text embeddings"""
        text_features = self.text_proj(text_emb)
        return F.normalize(text_features, p=2, dim=-1)


class TextToAudioEmbedder:
    """Main class for text-to-audio embedding"""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize embedder from checkpoint

        Args:
            checkpoint_path: Path to trained checkpoint (.pt file)
            device: Device to run on
            dtype: Data type for inference
        """
        self.device = device
        self.dtype = dtype

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]

        self.audio_encoder_name = config["audio_encoder"]
        self.audio_dim = config["audio_dim"]

        logger.info(f"Audio encoder: {self.audio_encoder_name}")
        logger.info(f"Audio dimension: {self.audio_dim}")

        # Initialize models
        logger.info("Loading text encoder...")
        self.text_encoder = TextEncoder(config["text_model_name"], device=device)
        self.text_encoder.model.load_state_dict(checkpoint["text_encoder_state_dict"])

        logger.info("Loading projection head...")
        self.projection = ProjectionHead(
            config["text_dim"], config["hidden_dim"], config["audio_dim"]
        ).to(device)
        self.projection.load_state_dict(checkpoint["projection_state_dict"])

        # Set to eval mode
        self.text_encoder.eval()
        self.projection.eval()

        # Store metrics
        self.metrics = checkpoint.get("metrics", {})
        logger.info(f"Model metrics: R@1={self.metrics.get('t2a_R@1', 0):.2f}%")

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text query

        Args:
            text: Text query string

        Returns:
            Audio space embedding (numpy array)
        """
        return self.embed_texts([text])[0]

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple text queries

        Args:
            texts: List of text query strings

        Returns:
            Audio space embeddings [N, audio_dim]
        """
        # Encode text
        text_emb = self.text_encoder(texts)

        # Project to audio space
        audio_features = self.projection(text_emb)

        return audio_features.cpu().numpy()

    @torch.no_grad()
    def embed_texts_batch(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed large number of texts in batches

        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            Audio space embeddings [N, audio_dim]
        """
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Embedding texts")

        for i in iterator:
            batch_texts = texts[i : i + batch_size]
            embeddings = self.embed_texts(batch_texts)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)

    def compute_similarity(self, text: str, audio_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity between text and audio embeddings

        Args:
            text: Text query
            audio_embeddings: Audio embeddings [N, audio_dim]

        Returns:
            Similarity scores [N]
        """
        text_emb = self.embed_text(text)

        # Normalize audio embeddings if not already
        audio_norms = np.linalg.norm(audio_embeddings, axis=1, keepdims=True)
        audio_embeddings_norm = audio_embeddings / (audio_norms + 1e-8)

        # Cosine similarity (dot product of normalized vectors)
        similarities = audio_embeddings_norm @ text_emb

        return similarities

    def retrieve_top_k(
        self,
        text: str,
        audio_embeddings: np.ndarray,
        audio_ids: Optional[List[str]] = None,
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-k most similar audio samples

        Args:
            text: Text query
            audio_embeddings: Audio embeddings [N, audio_dim]
            audio_ids: Optional list of audio IDs
            k: Number of results to return

        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = self.compute_similarity(text, audio_embeddings)

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_k_indices:
            score = similarities[idx]
            if audio_ids:
                results.append((audio_ids[idx], score))
            else:
                results.append((idx, score))

        return results


class MultiModelEmbedder:
    """Ensemble of all 3 models"""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda",
        use_best_r1: bool = True,
    ):
        """
        Load all 3 models

        Args:
            checkpoint_dir: Directory containing checkpoints
            device: Device to run on
            use_best_r1: Use best_r1 checkpoints (vs best_loss)
        """
        suffix = "best_r1" if use_best_r1 else "best_loss"

        self.embedders = {}

        for encoder in ["muq", "mert", "latent"]:
            checkpoint_path = f"{checkpoint_dir}/{encoder}_{suffix}.pt"
            logger.info(f"Loading {encoder} model...")
            self.embedders[encoder] = TextToAudioEmbedder(
                checkpoint_path, device=device
            )

    def embed_text(self, text: str, model: str = "muq") -> np.ndarray:
        """
        Embed text using specific model

        Args:
            text: Text query
            model: 'muq', 'mert', or 'latent'

        Returns:
            Embedding in audio space
        """
        return self.embedders[model].embed_text(text)

    def embed_text_all(self, text: str) -> Dict[str, np.ndarray]:
        """
        Embed text using all 3 models

        Args:
            text: Text query

        Returns:
            Dictionary mapping model name to embedding
        """
        return {
            name: embedder.embed_text(text) for name, embedder in self.embedders.items()
        }


def main():
    parser = argparse.ArgumentParser(description="Text-to-audio embedding inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--query", type=str, nargs="+", help="Text query/queries to embed"
    )
    parser.add_argument(
        "--query-file", type=str, help="File containing queries (one per line)"
    )
    parser.add_argument("--output", type=str, help="Output file for embeddings (.npy)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Load model
    embedder = TextToAudioEmbedder(args.checkpoint, device=args.device)

    # Get queries
    if args.query:
        queries = [" ".join(args.query)]  # Join multiple words
    elif args.query_file:
        with open(args.query_file, "r") as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        print("Interactive mode - enter queries (Ctrl+C to exit)")
        queries = []
        try:
            while True:
                query = input("\nQuery: ").strip()
                if query:
                    queries.append(query)
        except KeyboardInterrupt:
            print("\n")

    if not queries:
        logger.error("No queries provided!")
        return

    # Embed queries
    logger.info(f"Embedding {len(queries)} queries...")
    embeddings = embedder.embed_texts_batch(queries, show_progress=True)

    # Output
    if args.output:
        np.save(args.output, embeddings)
        logger.info(f"Embeddings saved to {args.output}")
    else:
        print("\nEmbeddings:")
        for i, (query, emb) in enumerate(zip(queries, embeddings)):
            print(f"\nQuery {i+1}: {query}")
            print(f"Embedding shape: {emb.shape}")
            print(f"Norm: {np.linalg.norm(emb):.4f}")
            print(f"First 10 dims: {emb[:10]}")


if __name__ == "__main__":
    main()
