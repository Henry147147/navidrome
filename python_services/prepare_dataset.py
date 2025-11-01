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
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm

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


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation"""
    output_dir: str = "data/musicbench_embeddings"
    hdf5_filename: str = "embeddings.h5"
    train_val_split: float = 0.9  # 90% train, 10% val
    compression: str = "gzip"
    compression_level: int = 4
    batch_size: int = 1  # Process one at a time due to variable lengths
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"
    random_seed: int = 42


class AudioEmbeddingExtractor:
    """Manages all 3 audio embedding models and extraction"""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype

        logger.info("Initializing audio embedding models...")

        # Initialize MuQ model
        logger.info("Loading MuQ-MuLan model...")
        self.muq_model = MuQEmbeddingModel(device=device, dtype=dtype)

        # Initialize MERT model
        logger.info("Loading MERT-v1-330M model...")
        self.mert_model = MertModel(device=device, dtype=dtype)

        # Initialize Music2Latent model
        logger.info("Loading Music2Latent model...")
        self.latent_model = MusicLatentSpaceModel(device=device, dtype=dtype)

        logger.info("All models loaded successfully!")

    def extract_embeddings(self, audio_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract embeddings from all 3 models for a single audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with embeddings or None if processing fails
        """
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.to(self.device).to(self.dtype)

            embeddings = {}

            # Extract MuQ embedding (1536D after enrichment)
            try:
                muq_emb = self.muq_model.embed_audio(waveform, sample_rate)
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
                mert_emb = self.mert_model.embed_audio(waveform, sample_rate)
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
                latent_emb = self.latent_model.embed_audio(waveform, sample_rate)
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

    def load_and_split_dataset(self):
        """Load MusicBench and create train/val/test splits"""
        logger.info("Loading MusicBench dataset...")

        dataset = load_dataset("amaai-lab/MusicBench")

        logger.info(f"Original train samples: {len(dataset['train'])}")
        logger.info(f"Test samples: {len(dataset['test'])}")

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

    def process_split(self, split_name: str, split_data, hdf5_file: h5py.File):
        """
        Process a single split and save to HDF5

        Args:
            split_name: 'train', 'val', or 'test'
            split_data: Dataset split
            hdf5_file: Open HDF5 file handle
        """
        logger.info(f"\nProcessing {split_name} split...")

        # Calculate total samples (2x for main + alt captions)
        total_samples = len(split_data) * 2

        # Create groups
        split_group = hdf5_file.create_group(split_name)

        # Pre-allocate datasets
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

        # Create variable-length string dataset for captions
        dt = h5py.special_dtype(vlen=str)
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

        # Process each sample
        idx = 0
        for sample_idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            # Get audio path (MusicBench provides 'location' or 'audio' field)
            audio_info = sample.get('audio', {})

            # Handle different possible audio formats
            if isinstance(audio_info, dict) and 'path' in audio_info:
                audio_path = audio_info['path']
            elif isinstance(audio_info, dict) and 'array' in audio_info:
                # Audio is provided as array, need to save temporarily
                audio_array = audio_info['array']
                sample_rate = audio_info.get('sampling_rate', 44100)

                # Create temp file
                temp_path = f"/tmp/musicbench_temp_{sample_idx}.wav"
                waveform_tensor = torch.tensor(audio_array, dtype=torch.float32)
                if len(waveform_tensor.shape) == 1:
                    waveform_tensor = waveform_tensor.unsqueeze(0)
                torchaudio.save(temp_path, waveform_tensor, sample_rate)
                audio_path = temp_path
            else:
                logger.warning(f"Cannot extract audio path for sample {sample_idx}")
                failed += 1
                failed_ids.append(sample.get('id', f'unknown_{sample_idx}'))
                continue

            # Extract embeddings
            embeddings = self.extractor.extract_embeddings(audio_path)

            # Clean up temp file if created
            if 'temp_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)

            if embeddings is None:
                failed += 1
                failed_ids.append(sample.get('id', f'unknown_{sample_idx}'))
                continue

            # Get captions
            main_caption = sample.get('caption', '')
            alt_caption = sample.get('alt_caption', '')
            audio_id = sample.get('id', f'{split_name}_{sample_idx}')

            # Save main caption version
            if main_caption:
                muq_ds[idx] = embeddings['muq']
                mert_ds[idx] = embeddings['mert']
                latent_ds[idx] = embeddings['latent']
                captions_ds[idx] = main_caption
                audio_ids_ds[idx] = f"{audio_id}_main"
                idx += 1

            # Save alt caption version
            if alt_caption:
                muq_ds[idx] = embeddings['muq']
                mert_ds[idx] = embeddings['mert']
                latent_ds[idx] = embeddings['latent']
                captions_ds[idx] = alt_caption
                audio_ids_ds[idx] = f"{audio_id}_alt"
                idx += 1

            successful += 1

        # Resize datasets if some samples failed
        if idx < total_samples:
            logger.info(f"Resizing datasets from {total_samples} to {idx} (removed failed samples)")
            muq_ds.resize((idx,) + muq_ds.shape[1:])
            mert_ds.resize((idx,) + mert_ds.shape[1:])
            latent_ds.resize((idx,) + latent_ds.shape[1:])
            captions_ds.resize((idx,))
            audio_ids_ds.resize((idx,))

        # Save statistics
        stats = {
            'total_audio_samples': len(split_data),
            'total_embedding_pairs': idx,
            'successful': successful,
            'failed': failed,
            'failed_ids': failed_ids
        }

        split_group.attrs['stats'] = json.dumps(stats)

        logger.info(f"{split_name} split complete:")
        logger.info(f"  Successful: {successful}/{len(split_data)}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total embedding pairs: {idx}")

        return stats

    def process_all(self):
        """Main processing pipeline"""
        logger.info("Starting MusicBench dataset preparation...")

        # Load and split dataset
        splits = self.load_and_split_dataset()

        # Create HDF5 file
        hdf5_path = os.path.join(self.config.output_dir, self.config.hdf5_filename)
        logger.info(f"Creating HDF5 file: {hdf5_path}")

        all_stats = {}

        with h5py.File(hdf5_path, 'w') as hdf5_file:
            # Save config as attributes
            hdf5_file.attrs['config'] = json.dumps(self.config.__dict__)
            hdf5_file.attrs['embedding_dimensions'] = json.dumps({
                'muq': 1536,
                'mert': 76800,
                'latent': 576
            })

            # Process each split
            for split_name in ['train', 'val', 'test']:
                stats = self.process_split(split_name, splits[split_name], hdf5_file)
                all_stats[split_name] = stats

        # Save summary
        summary_path = os.path.join(self.config.output_dir, 'preparation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_stats, f, indent=2)

        logger.info(f"\nDataset preparation complete!")
        logger.info(f"HDF5 file saved to: {hdf5_path}")
        logger.info(f"Summary saved to: {summary_path}")

        # Print final statistics
        total_pairs = sum(stats['total_embedding_pairs'] for stats in all_stats.values())
        total_failed = sum(stats['failed'] for stats in all_stats.values())

        logger.info(f"\nFinal Statistics:")
        logger.info(f"  Total embedding pairs: {total_pairs}")
        logger.info(f"  Total failed samples: {total_failed}")
        logger.info(f"  Success rate: {(1 - total_failed/sum(stats['total_audio_samples'] for stats in all_stats.values()))*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Prepare MusicBench dataset for training")
    parser.add_argument('--output-dir', type=str, default='data/musicbench_embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for embedding extraction')
    parser.add_argument('--dtype', type=str, default='float32',
                       help='Data type for embeddings')
    parser.add_argument('--train-val-split', type=float, default=0.9,
                       help='Fraction of training data to use for training (rest for validation)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Create config
    config = DatasetConfig(
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        train_val_split=args.train_val_split,
        random_seed=args.seed
    )

    # Process dataset
    processor = MusicBenchProcessor(config)
    processor.process_all()


if __name__ == '__main__':
    main()
