"""Data loader for music-text embedding training."""

import h5py
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List
import logging


class MusicTextDataset(Dataset):
    """
    Dataset for music-text embedding training.

    Loads pre-computed audio embeddings from HDF5 files and corresponding
    text descriptions from metadata.
    """

    def __init__(
        self,
        embeddings_file: Path,
        metadata_file: Path,
        tokenizer,
        max_text_length: int = 512,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize dataset.

        Args:
            embeddings_file: HDF5 file containing audio embeddings
            metadata_file: JSON file containing text metadata
            tokenizer: Tokenizer for text processing
            max_text_length: Maximum length for tokenized text
            logger: Logger instance
        """
        self.embeddings_file = Path(embeddings_file)
        self.metadata_file = Path(metadata_file)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.logger = logger or logging.getLogger("MusicTextDataset")

        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        # Open HDF5 file
        self.hdf5_file = h5py.File(embeddings_file, 'r')
        self.embeddings = self.hdf5_file['embeddings']
        self.file_paths = self.hdf5_file['file_paths']

        # Create mapping from file path to index
        self.path_to_idx = {}
        for idx in range(len(self.file_paths)):
            path = self.file_paths[idx]
            if isinstance(path, bytes):
                path = path.decode('utf-8')
            self.path_to_idx[path] = idx

        self.logger.info(f"Loaded dataset with {len(self.metadata)} samples")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns:
            Tuple of (input_ids, attention_mask, audio_embedding)
        """
        # Get metadata
        sample = self.metadata[idx]

        # Extract text description
        text = self._extract_text(sample)

        # Tokenize text
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # Get audio embedding
        # Find the corresponding embedding by matching file path
        sample_id = sample.get('id', sample.get('track_id', idx))
        audio_embedding = self._get_audio_embedding(sample_id, idx)

        return input_ids, attention_mask, audio_embedding

    def _extract_text(self, sample: dict) -> str:
        """Extract text description from sample metadata."""
        # Try different text fields
        for key in ['caption', 'text', 'description']:
            if key in sample and sample[key]:
                return str(sample[key])

        # Fallback: combine genres, instruments, moods
        text_parts = []

        if 'genres' in sample:
            genres = sample['genres']
            if isinstance(genres, list):
                text_parts.append("Genres: " + ", ".join(genres))
            elif genres:
                text_parts.append(f"Genre: {genres}")

        if 'instruments' in sample:
            instruments = sample['instruments']
            if isinstance(instruments, list):
                text_parts.append("Instruments: " + ", ".join(instruments))

        if 'moods' in sample or 'mood' in sample:
            mood = sample.get('moods', sample.get('mood'))
            if isinstance(mood, list):
                text_parts.append("Mood: " + ", ".join(mood))
            elif mood:
                text_parts.append(f"Mood: {mood}")

        if 'artist' in sample and sample['artist']:
            text_parts.append(f"Artist: {sample['artist']}")

        if 'title' in sample and sample['title']:
            text_parts.append(f"Title: {sample['title']}")

        if text_parts:
            return ". ".join(text_parts)

        # Last resort: return placeholder
        return "Music audio"

    def _get_audio_embedding(self, sample_id, fallback_idx: int) -> torch.Tensor:
        """Get audio embedding for a sample."""
        # Try to find by ID in path mapping
        # Sample IDs might be formatted differently, try a few variants
        possible_paths = [
            f"audio/{sample_id:06d}.wav",
            f"audio/{sample_id:06d}.mp3",
            f"audio/{sample_id}.wav",
            f"audio/{sample_id}.mp3",
            str(sample_id),
        ]

        for path in possible_paths:
            if path in self.path_to_idx:
                idx = self.path_to_idx[path]
                embedding = self.embeddings[idx]
                return torch.from_numpy(embedding).float()

        # Fallback to using the sample index directly
        if fallback_idx < len(self.embeddings):
            embedding = self.embeddings[fallback_idx]
            return torch.from_numpy(embedding).float()

        # Should not reach here, but return zeros as last resort
        self.logger.warning(f"Could not find embedding for sample {sample_id}")
        embedding_dim = self.embeddings.shape[1]
        return torch.zeros(embedding_dim, dtype=torch.float32)

    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()

    def __del__(self):
        """Cleanup when dataset is deleted."""
        self.close()


def create_dataloader(
    embeddings_file: Path,
    metadata_file: Path,
    tokenizer,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    max_text_length: int = 512,
) -> DataLoader:
    """
    Create a DataLoader for music-text training.

    Args:
        embeddings_file: HDF5 file with audio embeddings
        metadata_file: JSON file with text metadata
        tokenizer: Text tokenizer
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        max_text_length: Maximum text length

    Returns:
        DataLoader instance
    """
    dataset = MusicTextDataset(
        embeddings_file=embeddings_file,
        metadata_file=metadata_file,
        tokenizer=tokenizer,
        max_text_length=max_text_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


__all__ = [
    'MusicTextDataset',
    'create_dataloader',
]
