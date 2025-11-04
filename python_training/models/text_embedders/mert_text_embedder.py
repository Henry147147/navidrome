"""Text embedder for MERT audio embeddings (76,800D)."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_text_embedder import BaseTextEmbedder, enrich_text_embedding


class MERTTextEmbedder(BaseTextEmbedder):
    """
    Text embedding model that produces 76,800D embeddings to match MERT audio embeddings.

    MERT outputs 76,800D (25 layers × 1024 × 3 enrichment).

    Architecture:
    - Base: roberta-base or gpt2-medium
    - Hidden dim: 768 (RoBERTa) or 1024 (GPT-2)
    - Multi-layer extraction to mimic MERT's 25 layers
    - Projection: hidden_dim × num_layers -> 25,600
    - Enrichment: 25,600 -> 76,800
    """

    def __init__(
        self,
        base_model: str = "roberta-base",
        target_dim: int = 25600,  # Before enrichment (25 layers × 1024)
        enrichment_enabled: bool = True,
        num_layers_to_use: int = 12,  # Use all 12 layers from base model
        dropout: float = 0.1,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize MERT text embedder.

        Args:
            base_model: HuggingFace model name
            target_dim: Target dimension before enrichment (25,600)
            enrichment_enabled: Whether to apply enrichment
            num_layers_to_use: Number of transformer layers to extract
            dropout: Dropout rate
            logger: Logger instance
        """
        super().__init__(
            target_dim=target_dim,
            enrichment_enabled=enrichment_enabled,
            logger=logger
        )

        # Load base transformer
        self.transformer = AutoModel.from_pretrained(base_model, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Get hidden dimension from transformer
        self.hidden_dim = self.transformer.config.hidden_size
        self.num_layers_to_use = num_layers_to_use

        # Calculate combined dimension from all layers
        self.combined_dim = self.hidden_dim * num_layers_to_use

        # Projection to target dimension
        # We need to go from (hidden_dim × num_layers) to 25,600
        self.projection = nn.Sequential(
            nn.Linear(self.combined_dim, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4096, target_dim),
            nn.LayerNorm(target_dim),
        )

        self.logger.info(
            f"Initialized MERT text embedder: "
            f"{self.hidden_dim}×{num_layers_to_use} -> {target_dim} -> {self.output_dim}"
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the text embedder.

        Args:
            input_ids: Tokenized input text [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Text embeddings [batch_size, output_dim]
        """
        # Get transformer outputs with all hidden states
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Extract hidden states from multiple layers
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Each tensor is [batch_size, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states

        # Use the last num_layers_to_use layers
        selected_layers = hidden_states[-self.num_layers_to_use:]

        # Pool each layer (mean pooling over sequence)
        pooled_layers = []
        for layer_hidden in selected_layers:
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden.size())
                sum_hidden = torch.sum(layer_hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                pooled = layer_hidden.mean(dim=1)
            pooled_layers.append(pooled)

        # Concatenate all layer representations
        combined = torch.cat(pooled_layers, dim=1)  # [batch_size, hidden_dim × num_layers]

        # Project to target dimension
        embedding = self.projection(combined)  # [batch_size, target_dim]

        # Apply enrichment if enabled
        if self.enrichment_enabled:
            embedding = enrich_text_embedding(embedding)  # [batch_size, 3*target_dim]

        return embedding

    def encode_texts(self, texts: list, batch_size: int = 16, device: str = "cuda") -> torch.Tensor:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding (smaller due to large model)
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
            'model_type': 'MERTTextEmbedder',
            'base_model': self.transformer.config.name_or_path,
            'target_dim': self.target_dim,
            'enrichment_enabled': self.enrichment_enabled,
            'output_dim': self.output_dim,
            'num_layers_to_use': self.num_layers_to_use,
        }
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)


if __name__ == "__main__":
    # Test the model
    logging.basicConfig(level=logging.INFO)

    model = MERTTextEmbedder()
    print(f"Model output dim: {model.get_output_dim()}")

    # Test encoding
    texts = [
        "A upbeat pop song with electronic drums and synth",
        "Slow jazz ballad with piano and saxophone",
    ]

    embeddings = model.encode_texts(texts, device="cpu")
    print(f"Embeddings shape: {embeddings.shape}")  # Should be [2, 76800]
    print(f"Embedding norm: {torch.norm(embeddings[0]):.4f}")
