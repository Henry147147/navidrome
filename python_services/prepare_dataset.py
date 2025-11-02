#!/usr/bin/env python3
"""
prepare_dataset.py - MusicBench Dataset Preparation for Audio-Text Embedding Training

This script:
1. Loads MusicBench dataset from HuggingFace
2. Processes audio through 3 embedding models (MuQ, MERT, Music2Latent)
3. Applies enrichment to create final embeddings
4. Saves pre-computed embeddings to HDF5 for efficient training
5. Handles both main and alt captions for data augmentation
"""

import os
import sys
import argparse
import json
import logging
import signal
import tempfile
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

import h5py
import numpy as np
import torch
import soundfile as sf
import torchaudio
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator
from sklearn.model_selection import train_test_split as sk_train_test_split

# Import existing embedding models
from embedding_models import (
    MuQEmbeddingModel,
    MertModel,
    MusicLatentSpaceModel,
    enrich_embedding
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prepare_dataset.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle SIGINT and SIGTERM for graceful shutdown"""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    logger.warning(f"\n{sig_name} received. Saving checkpoint and exiting gracefully...")
    _shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@dataclass
class CheckpointState:
    """State for resuming dataset preparation"""
    split: str  # 'train', 'val', or 'test'
    model_pass: str  # 'muq', 'mert', or 'latent'
    processed_audio_files: Set[str] = field(default_factory=set)
    failed_audio_files: Set[str] = field(default_factory=set)
    batch_index: int = 0
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            'split': self.split,
            'model_pass': self.model_pass,
            'processed_audio_files': list(self.processed_audio_files),
            'failed_audio_files': list(self.failed_audio_files),
            'batch_index': self.batch_index,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CheckpointState':
        return cls(
            split=data['split'],
            model_pass=data['model_pass'],
            processed_audio_files=set(data.get('processed_audio_files', [])),
            failed_audio_files=set(data.get('failed_audio_files', [])),
            batch_index=data.get('batch_index', 0),
            timestamp=data.get('timestamp', '')
        )


class CheckpointManager:
    """Manages checkpoint save/load/validation for dataset preparation"""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def save(self, state: CheckpointState) -> None:
        """Save checkpoint atomically"""
        try:
            from datetime import datetime
            state.timestamp = datetime.now().isoformat()

            # Write to temporary file first
            temp_path = self.checkpoint_path + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)

            # Atomic rename
            shutil.move(temp_path, self.checkpoint_path)
            logger.info(f"Checkpoint saved: {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def load(self) -> Optional[CheckpointState]:
        """Load checkpoint if it exists and is valid"""
        if not os.path.exists(self.checkpoint_path):
            return None

        try:
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f)
            state = CheckpointState.from_dict(data)
            logger.info(f"Loaded checkpoint from {state.timestamp}")
            logger.info(f"  Split: {state.split}, Model: {state.model_pass}")
            logger.info(f"  Processed: {len(state.processed_audio_files)} files")
            logger.info(f"  Failed: {len(state.failed_audio_files)} files")
            return state
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            return None

    def delete(self) -> None:
        """Delete checkpoint file"""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            logger.info(f"Checkpoint deleted: {self.checkpoint_path}")


# Batch processing helpers
def load_single_audio(audio_path: str) -> Tuple[Optional[torch.Tensor], int, str]:
    """
    Load a single audio file and convert to tensor.

    Returns:
        (waveform, sample_rate, audio_path) or (None, 0, audio_path) if failed
    """
    try:
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            return None, 0, audio_path

        # Load audio file using soundfile
        audio_data, sample_rate = sf.read(audio_path, dtype='float32')

        # Convert to torch tensor and ensure correct shape [channels, samples]
        if len(audio_data.shape) == 1:
            # Mono audio
            waveform = torch.from_numpy(audio_data).unsqueeze(0)
        else:
            # Stereo or multi-channel: transpose from [samples, channels] to [channels, samples]
            waveform = torch.from_numpy(audio_data.T)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform, sample_rate, audio_path
    except Exception as e:
        logger.error(f"Failed to load {audio_path}: {e}")
        return None, 0, audio_path


def load_audio_batch(audio_paths: List[str], num_workers: int = 4) -> List[Tuple[Optional[torch.Tensor], int, str]]:
    """
    Load multiple audio files in parallel using ThreadPoolExecutor.

    Args:
        audio_paths: List of paths to audio files
        num_workers: Number of parallel workers

    Returns:
        List of (waveform, sample_rate, audio_path) tuples
    """
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(load_single_audio, audio_paths))
    return results


def collate_waveforms(
    waveforms: List[torch.Tensor],
    sample_rates: List[int],
    target_sr: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Collate variable-length waveforms into a batched tensor.

    Args:
        waveforms: List of waveforms with shape [channels, samples]
        sample_rates: List of sample rates
        target_sr: Target sample rate for resampling
        device: Target device
        dtype: Target dtype

    Returns:
        Batched tensor of shape [batch_size, channels, max_length]
    """
    # Resample all to target rate if needed
    resampled = []
    for waveform, sr in zip(waveforms, sample_rates):
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        resampled.append(waveform)

    # Find max length
    max_len = max(w.shape[-1] for w in resampled)

    # Pad to max length and move to device
    padded = []
    for waveform in resampled:
        if waveform.shape[-1] < max_len:
            pad_amount = max_len - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        padded.append(waveform.to(device).to(dtype))

    # Stack into batch
    return torch.stack(padded, dim=0)  # [batch_size, channels, max_len]


class AudioBatchLoader:
    """Iterator that yields batches of audio samples from dataset."""

    def __init__(self, dataset, batch_size: int = 32):
        """
        Args:
            dataset: HuggingFace dataset
            batch_size: Number of samples per batch
        """
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[List]:
        """Yield batches of samples."""
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # Yield remaining samples
        if batch:
            yield batch

    def __len__(self) -> int:
        """Return number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# ============================================================================
# Unified Dataset Loader (HuggingFace Dataset-compatible interface)
# ============================================================================

class UnifiedDatasetSubset:
    """Subset of unified dataset for train/val/test splits"""

    def __init__(self, parent_loader: 'UnifiedDatasetLoader', indices: List[int]):
        """
        Initialize subset.

        Args:
            parent_loader: Parent UnifiedDatasetLoader
            indices: List of indices for this subset
        """
        self.parent = parent_loader
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        """Get item by subset index"""
        actual_idx = self.indices[idx]
        return self.parent[actual_idx]

    def train_test_split(self, test_size: float, seed: int) -> Dict[str, 'UnifiedDatasetSubset']:
        """
        Split this subset further (for creating val/test from temp split).

        Args:
            test_size: Fraction for test set
            seed: Random seed

        Returns:
            Dict with 'train' and 'test' subsets
        """
        train_indices, test_indices = sk_train_test_split(
            self.indices,
            test_size=test_size,
            random_state=seed
        )

        return {
            'train': UnifiedDatasetSubset(self.parent, train_indices),
            'test': UnifiedDatasetSubset(self.parent, test_indices)
        }


class UnifiedDatasetLoader:
    """
    Loader for unified JSON dataset with HuggingFace Dataset-compatible interface.

    This allows seamless replacement of MusicBench dataset loading with unified
    multi-dataset JSON format.
    """

    def __init__(
        self,
        dataset_file: str,
        sample_size: Optional[int] = None,
        min_duration: float = 0.0,
        max_duration: float = float('inf'),
        random_seed: int = 42
    ):
        """
        Initialize unified dataset loader.

        Args:
            dataset_file: Path to unified JSON file
            sample_size: Optional sample size limit
            min_duration: Minimum audio duration (seconds)
            max_duration: Maximum audio duration (seconds)
            random_seed: Random seed for sampling
        """
        logger.info(f"Loading unified dataset from {dataset_file}...")

        # Load JSON
        with open(dataset_file, 'r') as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} tracks from unified dataset")

        # Validate required fields
        if self.data:
            required_fields = ['id', 'audio_path', 'caption', 'duration']
            missing_fields = [f for f in required_fields if f not in self.data[0]]
            if missing_fields:
                raise ValueError(f"Unified dataset missing required fields: {missing_fields}")

        # Filter by duration
        if min_duration > 0 or max_duration < float('inf'):
            original_count = len(self.data)
            self.data = [
                item for item in self.data
                if min_duration <= item.get('duration', 0) <= max_duration
            ]
            logger.info(f"Filtered by duration ({min_duration}s - {max_duration}s): "
                       f"{original_count} → {len(self.data)} tracks")

        # Sample if specified
        if sample_size and sample_size < len(self.data):
            random.seed(random_seed)
            self.data = random.sample(self.data, sample_size)
            logger.info(f"Sampled {sample_size} tracks")

        # Dataset statistics
        by_dataset = defaultdict(int)
        for item in self.data:
            by_dataset[item.get('dataset', 'unknown')] += 1

        logger.info(f"Dataset composition:")
        for dataset, count in sorted(by_dataset.items()):
            logger.info(f"  {dataset}: {count} tracks")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get item in format compatible with MusicBench expectations.

        Maps unified format to expected format:
        - audio_path → location
        - caption → main_caption
        - alt_captions[0] → alt_caption (if available)
        """
        item = self.data[idx]

        # Map to expected format
        result = {
            'location': item['audio_path'],
            'main_caption': item['caption'],
            'alt_caption': '',
            'id': item['id']
        }

        # Add alt caption if available
        if item.get('alt_captions') and len(item['alt_captions']) > 0:
            result['alt_caption'] = item['alt_captions'][0]

        return result

    def train_test_split(self, test_size: float, seed: int) -> Dict[str, UnifiedDatasetSubset]:
        """
        Split dataset into train and test subsets.

        Args:
            test_size: Fraction for test set
            seed: Random seed

        Returns:
            Dict with 'train' and 'test' UnifiedDatasetSubset instances
        """
        indices = list(range(len(self.data)))
        train_indices, test_indices = sk_train_test_split(
            indices,
            test_size=test_size,
            random_state=seed
        )

        return {
            'train': UnifiedDatasetSubset(self, train_indices),
            'test': UnifiedDatasetSubset(self, test_indices)
        }


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation"""
    # Input dataset
    dataset_file: str = "data/unified_dataset.json"  # Path to unified JSON
    use_musicbench: bool = False  # If True, load MusicBench instead of unified dataset

    # Sampling and filtering
    sample_size: Optional[int] = None  # Limit total samples processed
    min_duration: float = 5.0  # Minimum audio duration (seconds)
    max_duration: float = 600.0  # Maximum audio duration (seconds)

    # Output
    output_dir: str = "data/processed_embeddings"
    hdf5_filename: str = "embeddings.h5"
    checkpoint_filename: str = "prepare_checkpoint.json"

    # Data splitting
    train_val_split: float = 0.9  # 90% train, 10% val (for MusicBench only)

    # Processing
    compression: str = "gzip"
    compression_level: int = 4
    batch_size: int = 32  # Batch size for parallel processing
    num_workers: int = 4  # Number of parallel workers for audio loading
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"
    random_seed: int = 42
    continue_from_checkpoint: bool = False


class AudioEmbeddingExtractor:
    """Manages all 3 audio embedding models and extraction (one at a time for memory efficiency)"""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype

        # Don't load models yet - load them on-demand to save VRAM
        self.muq_model = None
        self.mert_model = None
        self.latent_model = None

        logger.info("Audio embedding extractor initialized (models will be loaded on-demand)")

    def extract_embeddings(self, audio_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract embeddings from all 3 models for a single audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with embeddings or None if processing fails
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                return None

            # Load audio file using soundfile
            audio_data, sample_rate = sf.read(audio_path, dtype='float32')

            # Convert to torch tensor and ensure correct shape [channels, samples]
            if len(audio_data.shape) == 1:
                # Mono audio
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
            else:
                # Stereo or multi-channel: transpose from [samples, channels] to [channels, samples]
                waveform = torch.from_numpy(audio_data.T)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.to(self.device).to(self.dtype)

            embeddings = {}

            # Extract MuQ embedding (1536D after enrichment)
            try:
                # Get raw embeddings without enrichment
                muq_emb = self.muq_model.embed_audio_tensor(
                    waveform, sample_rate, apply_enrichment=False
                )
                if muq_emb is not None and len(muq_emb.shape) == 2:
                    # Apply enrichment
                    muq_enriched = enrich_embedding(muq_emb)
                    embeddings['muq'] = muq_enriched.cpu().numpy()
                else:
                    logger.warning(f"MuQ embedding failed for {audio_path}")
                    return None
            except Exception as e:
                logger.error(f"MuQ extraction failed for {audio_path}: {e}")
                return None

            # Extract MERT embedding (76,800D after enrichment)
            try:
                # Get raw embeddings without enrichment
                mert_emb = self.mert_model.embed_audio_tensor(
                    waveform, sample_rate, apply_enrichment=False
                )
                if mert_emb is not None and len(mert_emb.shape) == 2:
                    # Apply enrichment
                    mert_enriched = enrich_embedding(mert_emb)
                    embeddings['mert'] = mert_enriched.cpu().numpy()
                else:
                    logger.warning(f"MERT embedding failed for {audio_path}")
                    return None
            except Exception as e:
                logger.error(f"MERT extraction failed for {audio_path}: {e}")
                return None

            # Extract Music2Latent embedding (576D after enrichment)
            try:
                # Get raw embeddings without enrichment
                latent_emb = self.latent_model.embed_audio_tensor(
                    waveform, sample_rate, apply_enrichment=False
                )
                if latent_emb is not None and len(latent_emb.shape) == 2:
                    # Apply enrichment
                    latent_enriched = enrich_embedding(latent_emb)
                    embeddings['latent'] = latent_enriched.cpu().numpy()
                else:
                    logger.warning(f"Latent embedding failed for {audio_path}")
                    return None
            except Exception as e:
                logger.error(f"Latent extraction failed for {audio_path}: {e}")
                return None

            return embeddings

        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {e}")
            return None

    def extract_embeddings_batch(
        self,
        audio_paths: List[str],
        num_workers: int = 4
    ) -> List[Optional[Dict[str, np.ndarray]]]:
        """
        Extract embeddings from all 3 models for multiple audio files in batch.

        Args:
            audio_paths: List of paths to audio files
            num_workers: Number of parallel workers for audio loading

        Returns:
            List of dictionaries with embeddings (or None for failed samples)
        """
        # Load all audio files in parallel
        logger.debug(f"Loading {len(audio_paths)} audio files in parallel...")
        audio_results = load_audio_batch(audio_paths, num_workers=num_workers)

        # Separate successful and failed loads
        valid_waveforms = []
        valid_sample_rates = []
        valid_indices = []

        for idx, (waveform, sample_rate, audio_path) in enumerate(audio_results):
            if waveform is not None:
                # Move to device
                waveform = waveform.to(self.device).to(self.dtype)
                valid_waveforms.append(waveform)
                valid_sample_rates.append(sample_rate)
                valid_indices.append(idx)

        # Initialize results list with None for all samples
        results = [None] * len(audio_paths)

        if not valid_waveforms:
            logger.warning("No valid audio files in batch")
            return results

        try:
            # Batch process through all 3 models
            logger.debug(f"Processing {len(valid_waveforms)} tracks through MuQ...")
            muq_embeddings = self.muq_model.embed_audio_tensor_batch(
                valid_waveforms, valid_sample_rates, apply_enrichment=False
            )

            # Apply enrichment to MuQ
            muq_enriched = [enrich_embedding(emb).cpu().numpy() for emb in muq_embeddings]

            logger.debug(f"Processing {len(valid_waveforms)} tracks through MERT...")
            mert_embeddings = self.mert_model.embed_audio_tensor_batch(
                valid_waveforms, valid_sample_rates, apply_enrichment=False
            )

            # Apply enrichment to MERT
            mert_enriched = [enrich_embedding(emb).cpu().numpy() for emb in mert_embeddings]

            logger.debug(f"Processing {len(valid_waveforms)} tracks through Music2Latent...")
            latent_embeddings = self.latent_model.embed_audio_tensor_batch(
                valid_waveforms, valid_sample_rates, apply_enrichment=False
            )

            # Apply enrichment to Latent
            latent_enriched = [enrich_embedding(emb).cpu().numpy() for emb in latent_embeddings]

            # Combine results
            for i, idx in enumerate(valid_indices):
                results[idx] = {
                    'muq': muq_enriched[i],
                    'mert': mert_enriched[i],
                    'latent': latent_enriched[i]
                }

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Return None for all samples in case of batch failure
            return results

        return results

    def extract_muq_embeddings_batch(
        self,
        audio_paths: List[str],
        num_workers: int = 4
    ) -> List[Optional[np.ndarray]]:
        """Extract only MuQ embeddings for a batch of audio files."""
        # Load model if not already loaded
        if self.muq_model is None:
            logger.info("Loading MuQ-MuLan model...")
            self.muq_model = MuQEmbeddingModel(device=self.device)

        # Load audio files
        audio_results = load_audio_batch(audio_paths, num_workers=num_workers)

        # Separate successful and failed loads
        valid_waveforms = []
        valid_sample_rates = []
        valid_indices = []

        for idx, (waveform, sample_rate, audio_path) in enumerate(audio_results):
            if waveform is not None:
                waveform = waveform.to(self.device).to(self.dtype)
                valid_waveforms.append(waveform)
                valid_sample_rates.append(sample_rate)
                valid_indices.append(idx)

        results = [None] * len(audio_paths)

        if not valid_waveforms:
            return results

        try:
            # Batch process through MuQ
            muq_embeddings = self.muq_model.embed_audio_tensor_batch(
                valid_waveforms, valid_sample_rates, apply_enrichment=False
            )

            # Apply enrichment
            muq_enriched = [enrich_embedding(emb).cpu().numpy() for emb in muq_embeddings]

            for i, idx in enumerate(valid_indices):
                results[idx] = muq_enriched[i]

        except Exception as e:
            logger.error(f"MuQ batch processing failed: {e}")

        return results

    def extract_mert_embeddings_batch(
        self,
        audio_paths: List[str],
        num_workers: int = 4
    ) -> List[Optional[np.ndarray]]:
        """Extract only MERT embeddings for a batch of audio files."""
        # Load model if not already loaded
        if self.mert_model is None:
            logger.info("Loading MERT-v1-330M model...")
            self.mert_model = MertModel(device=self.device)

        # Load audio files
        audio_results = load_audio_batch(audio_paths, num_workers=num_workers)

        valid_waveforms = []
        valid_sample_rates = []
        valid_indices = []

        for idx, (waveform, sample_rate, audio_path) in enumerate(audio_results):
            if waveform is not None:
                waveform = waveform.to(self.device).to(self.dtype)
                valid_waveforms.append(waveform)
                valid_sample_rates.append(sample_rate)
                valid_indices.append(idx)

        results = [None] * len(audio_paths)

        if not valid_waveforms:
            return results

        try:
            # Batch process through MERT
            mert_embeddings = self.mert_model.embed_audio_tensor_batch(
                valid_waveforms, valid_sample_rates, apply_enrichment=False
            )

            # Apply enrichment
            mert_enriched = [enrich_embedding(emb).cpu().numpy() for emb in mert_embeddings]

            for i, idx in enumerate(valid_indices):
                results[idx] = mert_enriched[i]

        except Exception as e:
            logger.error(f"MERT batch processing failed: {e}")

        return results

    def extract_latent_embeddings_batch(
        self,
        audio_paths: List[str],
        num_workers: int = 4
    ) -> List[Optional[np.ndarray]]:
        """Extract only Music2Latent embeddings for a batch of audio files."""
        # Load model if not already loaded
        if self.latent_model is None:
            logger.info("Loading Music2Latent model...")
            self.latent_model = MusicLatentSpaceModel(device=self.device)

        # Load audio files
        audio_results = load_audio_batch(audio_paths, num_workers=num_workers)

        valid_waveforms = []
        valid_sample_rates = []
        valid_indices = []

        for idx, (waveform, sample_rate, audio_path) in enumerate(audio_results):
            if waveform is not None:
                waveform = waveform.to(self.device).to(self.dtype)
                valid_waveforms.append(waveform)
                valid_sample_rates.append(sample_rate)
                valid_indices.append(idx)

        results = [None] * len(audio_paths)

        if not valid_waveforms:
            return results

        try:
            # Batch process through Latent
            latent_embeddings = self.latent_model.embed_audio_tensor_batch(
                valid_waveforms, valid_sample_rates, apply_enrichment=False
            )

            # Apply enrichment
            latent_enriched = [enrich_embedding(emb).cpu().numpy() for emb in latent_embeddings]

            for i, idx in enumerate(valid_indices):
                results[idx] = latent_enriched[i]

        except Exception as e:
            logger.error(f"Latent batch processing failed: {e}")

        return results

    def unload_model(self, model_name: str):
        """Unload a specific model to free VRAM."""
        import gc

        if model_name == 'muq' and self.muq_model is not None:
            logger.info("Unloading MuQ model...")
            del self.muq_model
            self.muq_model = None
        elif model_name == 'mert' and self.mert_model is not None:
            logger.info("Unloading MERT model...")
            del self.mert_model
            self.mert_model = None
        elif model_name == 'latent' and self.latent_model is not None:
            logger.info("Unloading Music2Latent model...")
            del self.latent_model
            self.latent_model = None

        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Freed CUDA memory. Current allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


class MusicBenchProcessor:
    """Processes MusicBench dataset and saves embeddings"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.extractor = AudioEmbeddingExtractor(
            device=config.device,
            dtype=getattr(torch, config.dtype.replace('float', 'float'))
        )

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Initialize checkpoint manager
        checkpoint_path = os.path.join(config.output_dir, config.checkpoint_filename)
        self.checkpoint_manager = CheckpointManager(checkpoint_path)
        self.checkpoint_state: Optional[CheckpointState] = None

        # Load checkpoint if continuing
        if config.continue_from_checkpoint:
            self.checkpoint_state = self.checkpoint_manager.load()
            if self.checkpoint_state is None:
                logger.warning("No checkpoint found. Starting from beginning.")

    def load_and_split_dataset(self):
        """Load dataset and create train/val/test splits"""

        # Choose dataset source
        if self.config.use_musicbench:
            return self._load_musicbench_dataset()
        else:
            return self._load_unified_dataset()

    def _load_musicbench_dataset(self):
        """Load MusicBench dataset from HuggingFace (original implementation)"""
        logger.info("Loading MusicBench dataset from HuggingFace...")

        dataset = load_dataset("amaai-lab/MusicBench")

        logger.info(f"Original train samples: {len(dataset['train'])}")
        logger.info(f"Test samples: {len(dataset['test'])}")

        # Validate that audio files exist
        self._validate_audio_files(dataset['train'])

        # Split training data into train/val
        train_val_split = dataset['train'].train_test_split(
            test_size=1 - self.config.train_val_split,
            seed=self.config.random_seed
        )

        splits = {
            'train': train_val_split['train'],
            'val': train_val_split['test'],
            'test': dataset['test']
        }

        logger.info(f"Final splits:")
        logger.info(f"  Train: {len(splits['train'])} samples")
        logger.info(f"  Val: {len(splits['val'])} samples")
        logger.info(f"  Test: {len(splits['test'])} samples")

        return splits

    def _load_unified_dataset(self):
        """Load unified dataset from JSON"""
        logger.info(f"Loading unified dataset from {self.config.dataset_file}...")

        # Load unified dataset
        dataset = UnifiedDatasetLoader(
            dataset_file=self.config.dataset_file,
            sample_size=self.config.sample_size,
            min_duration=self.config.min_duration,
            max_duration=self.config.max_duration,
            random_seed=self.config.random_seed
        )

        # Validate audio files
        if len(dataset) > 0:
            self._validate_audio_files(dataset)

        # Create train/val/test splits (80/10/10)
        # First split: 80% train, 20% temp
        train_temp_split = dataset.train_test_split(
            test_size=0.2,
            seed=self.config.random_seed
        )

        # Second split: split temp into 50/50 for val and test
        val_test_split = train_temp_split['test'].train_test_split(
            test_size=0.5,
            seed=self.config.random_seed
        )

        splits = {
            'train': train_temp_split['train'],
            'val': val_test_split['train'],
            'test': val_test_split['test']
        }

        logger.info(f"Final splits:")
        logger.info(f"  Train: {len(splits['train'])} samples")
        logger.info(f"  Val: {len(splits['val'])} samples")
        logger.info(f"  Test: {len(splits['test'])} samples")

        return splits

    def _validate_audio_files(self, dataset_split):
        """Validate that audio files exist"""
        logger.info("Validating audio file availability...")

        if len(dataset_split) == 0:
            logger.warning("Dataset is empty!")
            return

        sample = dataset_split[0]
        audio_path = sample['location']

        if not os.path.exists(audio_path):
            logger.error("=" * 80)
            logger.error("AUDIO FILES NOT FOUND")
            logger.error("=" * 80)
            logger.error(f"Expected audio file at: {audio_path}")
            logger.error("")
            logger.error("Please ensure:")
            logger.error("1. Audio files have been downloaded")
            logger.error("2. Unified dataset JSON contains correct absolute paths")
            logger.error("3. The paths are accessible from this machine")
            logger.error("=" * 80)
            raise FileNotFoundError(
                f"Audio files not found. Expected first file at: {audio_path}"
            )

        logger.info("✓ Audio files found! Proceeding with dataset preparation...")

    def process_split(self, split_name: str, split_data, hdf5_file: h5py.File):
        """
        Process a single split and save to HDF5

        Args:
            split_name: 'train', 'val', or 'test'
            split_data: Dataset split
            hdf5_file: Open HDF5 file handle
        """
        global _shutdown_requested

        logger.info(f"\nProcessing {split_name} split...")

        # Check if we should skip this split based on checkpoint
        if self.checkpoint_state is not None:
            if split_name < self.checkpoint_state.split:
                logger.info(f"Skipping {split_name} split (already completed)")
                return self._get_cached_stats(hdf5_file, split_name)
            elif split_name > self.checkpoint_state.split:
                logger.info(f"Checkpoint is for {self.checkpoint_state.split}, but we're on {split_name}. Starting fresh for this split.")
                self.checkpoint_state = None

        # Calculate total samples (2x for main + alt captions)
        total_samples = len(split_data) * 2

        # Create or get group (may already exist when resuming)
        if split_name in hdf5_file:
            split_group = hdf5_file[split_name]
            logger.info(f"Resuming existing split group: {split_name}")
        else:
            split_group = hdf5_file.create_group(split_name)

        # Pre-allocate or get existing datasets
        dt = h5py.special_dtype(vlen=str)

        if 'muq_embeddings' in split_group:
            muq_ds = split_group['muq_embeddings']
            mert_ds = split_group['mert_embeddings']
            latent_ds = split_group['latent_embeddings']
            captions_ds = split_group['captions']
            audio_ids_ds = split_group['audio_ids']
        else:
            muq_ds = split_group.create_dataset(
                'muq_embeddings',
                shape=(total_samples, 1536),
                dtype='float32',
                compression=self.config.compression,
                compression_opts=self.config.compression_level
            )

            mert_ds = split_group.create_dataset(
                'mert_embeddings',
                shape=(total_samples, 76800),
                dtype='float32',
                compression=self.config.compression,
                compression_opts=self.config.compression_level
            )

            latent_ds = split_group.create_dataset(
                'latent_embeddings',
                shape=(total_samples, 576),
                dtype='float32',
                compression=self.config.compression,
                compression_opts=self.config.compression_level
            )

            captions_ds = split_group.create_dataset(
                'captions',
                shape=(total_samples,),
                dtype=dt
            )

            audio_ids_ds = split_group.create_dataset(
                'audio_ids',
                shape=(total_samples,),
                dtype=dt
            )

        # Statistics
        successful = 0
        failed = 0
        failed_ids = []

        # First pass: Collect sample metadata and save captions/IDs
        logger.info(f"Collecting metadata for {len(split_data)} samples...")
        sample_metadata = []  # List of (audio_path, main_caption, alt_caption, audio_id)
        idx = 0

        # Check if captions already exist (resuming)
        existing_captions = idx < len(captions_ds) and captions_ds[0]

        for sample_idx, sample in enumerate(split_data):
            audio_path = sample['location']
            main_caption = sample.get('main_caption', sample.get('caption', ''))
            alt_caption = sample.get('alt_caption', '')
            audio_id = sample.get('id', f'{split_name}_{sample_idx}')

            # Save metadata
            if main_caption:
                sample_metadata.append((audio_path, main_caption, audio_id, 'main', idx))
                if not existing_captions or idx >= len(captions_ds):
                    captions_ds[idx] = main_caption
                    audio_ids_ds[idx] = f"{audio_id}_main"
                idx += 1

            if alt_caption:
                sample_metadata.append((audio_path, alt_caption, audio_id, 'alt', idx))
                if not existing_captions or idx >= len(captions_ds):
                    captions_ds[idx] = alt_caption
                    audio_ids_ds[idx] = f"{audio_id}_alt"
                idx += 1

        actual_samples = idx
        logger.info(f"Total caption pairs to process: {actual_samples}")

        # Build audio_to_indices mapping using defaultdict for efficiency
        audio_to_indices = defaultdict(list)
        for audio_path, _, _, _, hdf5_idx in sample_metadata:
            audio_to_indices[audio_path].append(hdf5_idx)

        # Determine which model passes to run based on checkpoint
        model_passes = ['muq', 'mert', 'latent']
        if self.checkpoint_state is not None and self.checkpoint_state.split == split_name:
            # Skip completed passes
            pass_idx = model_passes.index(self.checkpoint_state.model_pass)
            model_passes = model_passes[pass_idx:]
            logger.info(f"Resuming from {self.checkpoint_state.model_pass} pass")

        # Process each model separately to allow larger batch sizes
        for pass_idx, model_name in enumerate(model_passes):
            if _shutdown_requested:
                logger.warning("Shutdown requested. Saving checkpoint...")
                self._save_checkpoint(split_name, model_name, audio_to_indices)
                return None

            logger.info(f"\n{'='*60}")
            logger.info(f"PASS {pass_idx+1}/3: Extracting {model_name.upper()} embeddings...")
            logger.info(f"{'='*60}")

            # Get dataset for this model
            if model_name == 'muq':
                ds = muq_ds
                extract_func = self.extractor.extract_muq_embeddings_batch
            elif model_name == 'mert':
                ds = mert_ds
                extract_func = self.extractor.extract_mert_embeddings_batch
            else:  # latent
                ds = latent_ds
                extract_func = self.extractor.extract_latent_embeddings_batch

            self._process_model_pass(
                split_name=split_name,
                split_data=split_data,
                model_name=model_name,
                ds=ds,
                extract_func=extract_func,
                audio_to_indices=audio_to_indices
            )

            # Unload model to free VRAM
            self.extractor.unload_model(model_name)

        # Finalize: successful count (approximate)
        successful = len(split_data) - failed

        # Resize datasets if actual samples differ from allocated
        if actual_samples < total_samples:
            logger.info(f"Resizing datasets from {total_samples} to {actual_samples}")
            muq_ds.resize((actual_samples,) + muq_ds.shape[1:])
            mert_ds.resize((actual_samples,) + mert_ds.shape[1:])
            latent_ds.resize((actual_samples,) + latent_ds.shape[1:])
            captions_ds.resize((actual_samples,))
            audio_ids_ds.resize((actual_samples,))

        # Save statistics
        stats = {
            'total_audio_samples': len(split_data),
            'total_embedding_pairs': actual_samples,
            'successful': successful,
            'failed': failed,
            'failed_ids': failed_ids
        }

        split_group.attrs['stats'] = json.dumps(stats)

        logger.info(f"\n{split_name} split complete:")
        logger.info(f"  Successful: {successful}/{len(split_data)}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total embedding pairs: {actual_samples}")

        return stats

    def _process_model_pass(
        self,
        split_name: str,
        split_data,
        model_name: str,
        ds: h5py.Dataset,
        extract_func,
        audio_to_indices: Dict[str, List[int]]
    ):
        """Process a single model pass with checkpointing"""
        global _shutdown_requested

        batch_loader = AudioBatchLoader(split_data, batch_size=self.config.batch_size)

        # Determine which batches to process
        start_batch = 0
        if (self.checkpoint_state is not None and
            self.checkpoint_state.split == split_name and
            self.checkpoint_state.model_pass == model_name):
            start_batch = self.checkpoint_state.batch_index
            logger.info(f"Resuming from batch {start_batch}")

        # Get processed files set from checkpoint
        processed_files = set()
        if self.checkpoint_state and self.checkpoint_state.split == split_name:
            processed_files = self.checkpoint_state.processed_audio_files

        batch_count = 0
        for batch in tqdm(batch_loader, desc=f"{model_name.upper()} {split_name}", total=len(batch_loader)):
            # Skip already-processed batches
            if batch_count < start_batch:
                batch_count += 1
                continue

            # Check for shutdown signal
            if _shutdown_requested:
                logger.warning("Shutdown requested during batch processing")
                self._save_checkpoint(split_name, model_name, processed_files, batch_count)
                raise KeyboardInterrupt("Graceful shutdown")

            audio_paths = [sample['location'] for sample in batch]

            # Filter out already-processed files
            paths_to_process = [p for p in audio_paths if p not in processed_files]

            if not paths_to_process:
                batch_count += 1
                continue

            # Extract embeddings
            embeddings = extract_func(
                paths_to_process,
                num_workers=self.config.num_workers
            )

            # Save embeddings
            for audio_path, emb in zip(paths_to_process, embeddings):
                if emb is not None:
                    for hdf5_idx in audio_to_indices.get(audio_path, []):
                        ds[hdf5_idx] = emb
                    processed_files.add(audio_path)
                else:
                    logger.warning(f"Failed to process {audio_path}")

            batch_count += 1

            # Save checkpoint and log memory every 10 batches
            if batch_count % 10 == 0:
                self._save_checkpoint(split_name, model_name, processed_files, batch_count)

                # Log CUDA memory if available
                if torch.cuda.is_available():
                    allocated_gb = torch.cuda.memory_allocated() / 1e9
                    reserved_gb = torch.cuda.memory_reserved() / 1e9
                    logger.info(f"CUDA Memory: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved")

        # Save final checkpoint for this pass
        self._save_checkpoint(split_name, model_name, processed_files, batch_count)

    def _save_checkpoint(
        self,
        split_name: str,
        model_name: str,
        processed_files: Set[str],
        batch_index: int = 0
    ):
        """Save current processing state"""
        state = CheckpointState(
            split=split_name,
            model_pass=model_name,
            processed_audio_files=processed_files,
            batch_index=batch_index
        )
        self.checkpoint_manager.save(state)

    def _get_cached_stats(self, hdf5_file: h5py.File, split_name: str) -> Dict:
        """Get statistics from already-processed split"""
        if split_name not in hdf5_file:
            return {'total_audio_samples': 0, 'total_embedding_pairs': 0, 'successful': 0, 'failed': 0}

        split_group = hdf5_file[split_name]
        if 'stats' in split_group.attrs:
            return json.loads(split_group.attrs['stats'])

        # Infer stats from data
        return {
            'total_audio_samples': len(split_group['audio_ids']),
            'total_embedding_pairs': len(split_group['audio_ids']),
            'successful': len(split_group['audio_ids']),
            'failed': 0
        }

    def process_all(self):
        """Main processing pipeline"""
        global _shutdown_requested

        logger.info("Starting MusicBench dataset preparation...")
        if self.checkpoint_state:
            logger.info(f"Resuming from checkpoint:")
            logger.info(f"  Split: {self.checkpoint_state.split}")
            logger.info(f"  Model pass: {self.checkpoint_state.model_pass}")
            logger.info(f"  Processed files: {len(self.checkpoint_state.processed_audio_files)}")

        logger.info(f"Configuration:")
        logger.info(f"  Device: {self.config.device}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Num workers: {self.config.num_workers}")
        logger.info(f"  Data type: {self.config.dtype}")

        # Validate output directory is writable
        hdf5_path = os.path.join(self.config.output_dir, self.config.hdf5_filename)
        if os.path.exists(hdf5_path) and not os.access(hdf5_path, os.W_OK):
            raise PermissionError(f"HDF5 file exists but is not writable: {hdf5_path}")
        if not os.access(self.config.output_dir, os.W_OK):
            raise PermissionError(f"Output directory is not writable: {self.config.output_dir}")

        try:
            # Load and split dataset
            splits = self.load_and_split_dataset()

            # Create HDF5 file
            hdf5_path = os.path.join(self.config.output_dir, self.config.hdf5_filename)
            logger.info(f"Creating HDF5 file: {hdf5_path}")

            all_stats = {}

            # Open HDF5 with appropriate mode
            file_mode = 'a' if self.config.continue_from_checkpoint else 'w'
            with h5py.File(hdf5_path, file_mode) as hdf5_file:
                # Save config as attributes (only if creating new file)
                if file_mode == 'w':
                    hdf5_file.attrs['config'] = json.dumps(self.config.__dict__)
                    hdf5_file.attrs['embedding_dimensions'] = json.dumps({
                        'muq': 1536,
                        'mert': 76800,
                        'latent': 576
                    })

                # Process each split
                for split_name in ['train', 'val', 'test']:
                    if _shutdown_requested:
                        logger.warning("Shutdown requested. Exiting...")
                        return

                    stats = self.process_split(split_name, splits[split_name], hdf5_file)
                    if stats is not None:
                        all_stats[split_name] = stats

            # Save summary
            summary_path = os.path.join(self.config.output_dir, 'preparation_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(all_stats, f, indent=2)

            # Delete checkpoint on successful completion
            self.checkpoint_manager.delete()

            logger.info(f"\n{'='*80}")
            logger.info(f"DATASET PREPARATION COMPLETE!")
            logger.info(f"{'='*80}")
            logger.info(f"HDF5 file saved to: {hdf5_path}")
            logger.info(f"Summary saved to: {summary_path}")

            # Print final statistics
            if all_stats:
                total_pairs = sum(stats.get('total_embedding_pairs', 0) for stats in all_stats.values())
                total_failed = sum(stats.get('failed', 0) for stats in all_stats.values())
                total_audio = sum(stats.get('total_audio_samples', 0) for stats in all_stats.values())

                logger.info(f"\nFinal Statistics:")
                logger.info(f"  Total audio files: {total_audio}")
                logger.info(f"  Total embedding pairs (with augmentation): {total_pairs}")
                logger.info(f"  Failed samples: {total_failed}")
                if total_audio > 0:
                    success_rate = (1 - total_failed/total_audio)*100
                    logger.info(f"  Success rate: {success_rate:.2f}%")

                logger.info(f"\nSplit breakdown:")
                for split_name, stats in all_stats.items():
                    logger.info(f"  {split_name}: {stats.get('total_audio_samples', 0)} audio files, "
                              f"{stats.get('total_embedding_pairs', 0)} embedding pairs")

                # Display file size
                if os.path.exists(hdf5_path):
                    file_size_mb = os.path.getsize(hdf5_path) / (1024 * 1024)
                    logger.info(f"\nHDF5 file size: {file_size_mb:.2f} MB")

            logger.info(f"{'='*80}\n")

        except KeyboardInterrupt:
            logger.warning("\nGraceful shutdown completed. Resume with --continue flag.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during processing: {e}", exc_info=True)
            logger.info("Checkpoint saved. Resume with --continue flag.")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare music dataset for training (supports unified JSON or MusicBench)"
    )

    # Dataset selection
    parser.add_argument('--dataset-file', type=str, default='data/unified_dataset.json',
                       help='Path to unified dataset JSON file (default: data/unified_dataset.json)')
    parser.add_argument('--use-musicbench', action='store_true',
                       help='Use MusicBench from HuggingFace instead of unified dataset')

    # Sampling and filtering
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Limit total number of samples to process (default: all)')
    parser.add_argument('--min-duration', type=float, default=5.0,
                       help='Minimum audio duration in seconds (default: 5.0)')
    parser.add_argument('--max-duration', type=float, default=600.0,
                       help='Maximum audio duration in seconds (default: 600.0)')

    # Output
    parser.add_argument('--output-dir', type=str, default='data/processed_embeddings',
                       help='Output directory for embeddings (default: data/processed_embeddings)')

    # Processing
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for embedding extraction')
    parser.add_argument('--dtype', type=str, default='float32',
                       help='Data type for embeddings')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for parallel processing (default: 32)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of parallel workers for audio loading (default: 4)')

    # Splitting (for MusicBench only)
    parser.add_argument('--train-val-split', type=float, default=0.9,
                       help='Fraction of training data for training, rest for validation (MusicBench only)')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--continue', dest='continue_from_checkpoint', action='store_true',
                       help='Continue from previous checkpoint if it exists')

    args = parser.parse_args()

    # Create config
    config = DatasetConfig(
        dataset_file=args.dataset_file,
        use_musicbench=args.use_musicbench,
        sample_size=args.sample_size,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        train_val_split=args.train_val_split,
        random_seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        continue_from_checkpoint=args.continue_from_checkpoint
    )

    # Process dataset
    processor = MusicBenchProcessor(config)
    processor.process_all()


if __name__ == '__main__':
    main()
