"""Text embedder for MuQ audio embeddings (1536D)."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_text_embedder import BaseTextEmbedder, enrich_text_embedding


class MuQTextEmbedder(BaseTextEmbedder):
    """
    Text embedding model that produces 1536D embeddings to match MuQ audio embeddings.

    Architecture:
    - Base: sentence-transformers/all-MiniLM-L6-v2 or similar
    - Hidden dim: 384
    - Projection: 384 -> 512
    - Enrichment: 512 -> 1536 (mean, robust_sigma, dmean)
    """

    def __init__(
        self,
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        target_dim: int = 512,  # Before enrichment
        enrichment_enabled: bool = True,
        dropout: float = 0.1,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize MuQ text embedder.

        Args:
            base_model: HuggingFace model name for the base transformer
            target_dim: Target dimension before enrichment (512)
            enrichment_enabled: Whether to apply enrichment
            dropout: Dropout rate
            logger: Logger instance
        """
        super().__init__(
            target_dim=target_dim,
            enrichment_enabled=enrichment_enabled,
            logger=logger
        )

        # Load base transformer
        self.transformer = AutoModel.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Get hidden dimension from transformer
        self.hidden_dim = self.transformer.config.hidden_size

        # Projection layers to target dimension
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, target_dim),
            nn.LayerNorm(target_dim),
        )

        self.logger.info(f"Initialized MuQ text embedder: {self.hidden_dim} -> {target_dim} -> {self.output_dim}")

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the text embedder.

        Args:
            input_ids: Tokenized input text [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Text embeddings [batch_size, output_dim]
        """
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            # Use pooler output if available
            hidden = outputs.pooler_output
        else:
            # Mean pooling over sequence
            hidden = outputs.last_hidden_state
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size())
                sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                hidden = sum_hidden / sum_mask
            else:
                hidden = hidden.mean(dim=1)

        # Project to target dimension
        embedding = self.projection(hidden)  # [batch_size, target_dim]

        # Apply enrichment if enabled
        if self.enrichment_enabled:
            embedding = enrich_text_embedding(embedding)  # [batch_size, 3*target_dim]

        return embedding

    def encode_texts(self, texts: list, batch_size: int = 32, device: str = "cuda") -> torch.Tensor:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            device: Device to use

        Returns:
            Embeddings tensor [num_texts, output_dim]
        """
        self.eval()
        self.to(device)

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )

                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)

                # Get embeddings
                embeddings = self.forward(input_ids, attention_mask)
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def save_pretrained(self, save_directory: str) -> None:
        """Save model and tokenizer."""
        super().save_pretrained(save_directory)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

        # Save additional config
        import os
        import json
        config = {
            'model_type': 'MuQTextEmbedder',
            'base_model': self.transformer.config.name_or_path,
            'target_dim': self.target_dim,
            'enrichment_enabled': self.enrichment_enabled,
            'output_dim': self.output_dim,
        }
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)


if __name__ == "__main__":
    # Test the model
    logging.basicConfig(level=logging.INFO)

    model = MuQTextEmbedder()
    print(f"Model output dim: {model.get_output_dim()}")

    # Test encoding
    texts = [
        "A upbeat pop song with electronic drums and synth",
        "Slow jazz ballad with piano and saxophone",
        "Heavy metal guitar riffs with aggressive vocals",
    ]

    embeddings = model.encode_texts(texts, device="cpu")
    print(f"Embeddings shape: {embeddings.shape}")  # Should be [3, 1536]
    print(f"Embedding norm: {torch.norm(embeddings[0]):.4f}")
