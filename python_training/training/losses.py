"""Loss functions for music-text embedding training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.

    This loss encourages matching pairs (text, audio) to have high similarity
    while non-matching pairs have low similarity.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter for scaling similarities
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            text_embeddings: Text embeddings [batch_size, embedding_dim]
            audio_embeddings: Audio embeddings [batch_size, embedding_dim]

        Returns:
            Loss value (scalar tensor)
        """
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device

        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        audio_embeddings = F.normalize(audio_embeddings, p=2, dim=1)

        # Compute similarity matrix
        # [batch_size, batch_size] where sim[i,j] = similarity(text[i], audio[j])
        similarity_matrix = torch.matmul(text_embeddings, audio_embeddings.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=device)

        # Compute loss in both directions (text->audio and audio->text)
        loss_text_to_audio = F.cross_entropy(similarity_matrix, labels)
        loss_audio_to_text = F.cross_entropy(similarity_matrix.T, labels)

        # Average the two directions
        loss = (loss_text_to_audio + loss_audio_to_text) / 2.0

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: InfoNCE + MSE for direct embedding matching.

    InfoNCE: Contrastive learning for cross-modal alignment
    MSE: Direct regression to match audio embedding values
    """

    def __init__(
        self,
        temperature: float = 0.07,
        infonce_weight: float = 1.0,
        mse_weight: float = 0.1,
    ):
        """
        Initialize combined loss.

        Args:
            temperature: Temperature for InfoNCE
            infonce_weight: Weight for InfoNCE loss
            mse_weight: Weight for MSE loss
        """
        super().__init__()
        self.infonce = InfoNCELoss(temperature=temperature)
        self.mse = nn.MSELoss()
        self.infonce_weight = infonce_weight
        self.mse_weight = mse_weight

    def forward(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
    ) -> tuple:
        """
        Compute combined loss.

        Args:
            text_embeddings: Text embeddings [batch_size, embedding_dim]
            audio_embeddings: Audio embeddings [batch_size, embedding_dim]

        Returns:
            Tuple of (total_loss, infonce_loss, mse_loss)
        """
        # InfoNCE loss
        infonce_loss = self.infonce(text_embeddings, audio_embeddings)

        # MSE loss for direct matching
        mse_loss = self.mse(text_embeddings, audio_embeddings)

        # Combined loss
        total_loss = (
            self.infonce_weight * infonce_loss +
            self.mse_weight * mse_loss
        )

        return total_loss, infonce_loss, mse_loss


class TripletLoss(nn.Module):
    """
    Triplet loss for music-text embedding.

    Encourages text embedding to be closer to its matching audio
    than to non-matching audio.
    """

    def __init__(self, margin: float = 0.2):
        """
        Initialize triplet loss.

        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings_positive: torch.Tensor,
        audio_embeddings_negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            text_embeddings: Anchor embeddings [batch_size, embedding_dim]
            audio_embeddings_positive: Positive embeddings [batch_size, embedding_dim]
            audio_embeddings_negative: Negative embeddings [batch_size, embedding_dim]

        Returns:
            Loss value
        """
        # Normalize
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        audio_embeddings_positive = F.normalize(audio_embeddings_positive, p=2, dim=1)
        audio_embeddings_negative = F.normalize(audio_embeddings_negative, p=2, dim=1)

        # Compute distances
        positive_distance = 1.0 - F.cosine_similarity(text_embeddings, audio_embeddings_positive)
        negative_distance = 1.0 - F.cosine_similarity(text_embeddings, audio_embeddings_negative)

        # Triplet loss
        loss = F.relu(positive_distance - negative_distance + self.margin)

        return loss.mean()


__all__ = [
    'InfoNCELoss',
    'CombinedLoss',
    'TripletLoss',
]
