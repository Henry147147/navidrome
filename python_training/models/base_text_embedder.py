"""Base class for text embedding models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional
import logging


class BaseTextEmbedder(nn.Module, ABC):
    """
    Abstract base class for text embedding models.

    All text embedders should:
    1. Take tokenized text as input
    2. Output embeddings matching the dimensionality of their corresponding audio embedder
    3. Support enrichment (mean, robust_sigma, dmean) to match audio embedding format
    """

    def __init__(
        self,
        target_dim: int,
        enrichment_enabled: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize base text embedder.

        Args:
            target_dim: Target embedding dimension (before enrichment)
            enrichment_enabled: Whether to apply enrichment (3x expansion)
            logger: Logger instance
        """
        super().__init__()
        self.target_dim = target_dim
        self.enrichment_enabled = enrichment_enabled
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        if enrichment_enabled:
            self.output_dim = target_dim * 3  # mean, robust_sigma, dmean
        else:
            self.output_dim = target_dim

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the text embedder.

        Args:
            input_ids: Tokenized input text [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Text embeddings [batch_size, output_dim]
        """
        pass

    def get_output_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.output_dim

    def save_pretrained(self, save_directory: str) -> None:
        """Save model weights and configuration."""
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save model state
        torch.save({
            'state_dict': self.state_dict(),
            'target_dim': self.target_dim,
            'enrichment_enabled': self.enrichment_enabled,
            'output_dim': self.output_dim,
            'model_class': self.__class__.__name__,
        }, os.path.join(save_directory, 'pytorch_model.bin'))

        self.logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load model from saved weights."""
        import os
        checkpoint_path = os.path.join(load_directory, 'pytorch_model.bin')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Create model instance
        model = cls(
            target_dim=checkpoint['target_dim'],
            enrichment_enabled=checkpoint['enrichment_enabled'],
            **kwargs
        )

        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])

        return model


def enrich_text_embedding(embedding: torch.Tensor, num_synthetic_timesteps: int = 10) -> torch.Tensor:
    """
    Apply enrichment to text embedding to match audio embedding format.

    Since text embeddings don't have a natural time dimension like audio,
    we create synthetic temporal statistics by:
    1. Mean: Use the embedding itself
    2. Robust sigma: Add small noise to simulate variation
    3. Dmean: Create synthetic temporal gradient

    Args:
        embedding: Text embedding [batch_size, D] or [D]
        num_synthetic_timesteps: Number of synthetic timesteps for gradient calculation

    Returns:
        Enriched embedding [batch_size, 3*D] or [3*D]
    """
    if embedding.dim() == 1:
        # Single embedding [D] -> [3*D]
        D = embedding.shape[0]
        device = embedding.device

        # Mean: Use the embedding itself
        mean = embedding

        # Robust sigma: Estimate from embedding statistics
        # Use absolute value as a proxy for spread
        robust = torch.abs(embedding) * 0.5

        # Dmean: Create synthetic gradient
        # Use small random perturbation
        dmean = torch.randn(D, device=device) * 0.01 * torch.abs(embedding)

        return torch.cat([mean, robust, dmean], dim=0)

    elif embedding.dim() == 2:
        # Batch of embeddings [batch_size, D] -> [batch_size, 3*D]
        batch_size, D = embedding.shape
        device = embedding.device

        # Mean: Use the embedding itself
        mean = embedding  # [batch_size, D]

        # Robust sigma: Estimate from embedding statistics
        robust = torch.abs(embedding) * 0.5  # [batch_size, D]

        # Dmean: Create synthetic gradient with small random perturbation
        dmean = torch.randn(batch_size, D, device=device) * 0.01 * torch.abs(embedding)

        return torch.cat([mean, robust, dmean], dim=1)  # [batch_size, 3*D]

    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {embedding.dim()}D")


__all__ = [
    'BaseTextEmbedder',
    'enrich_text_embedding',
]
