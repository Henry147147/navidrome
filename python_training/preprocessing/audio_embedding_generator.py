"""Generate audio embeddings using existing models from python_services."""

import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import json
import h5py
import numpy as np
import torch
from tqdm import tqdm

# Add python_services to path to import existing embedding models
_python_services_dir = str(Path(__file__).parent.parent.parent / "python_services")
if _python_services_dir not in sys.path:
    sys.path.insert(0, _python_services_dir)

from embedding_models import MuQEmbeddingModel, MertModel, MusicLatentSpaceModel


class AudioEmbeddingGenerator:
    """
    Generate audio embeddings for training music-text alignment models.

    Uses the existing audio embedding models from python_services to generate
    embeddings for all audio files in a dataset.
    """

    def __init__(
        self,
        model_name: str = "muq",
        device: str = "cuda",
        batch_size: int = 8,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize audio embedding generator.

        Args:
            model_name: Name of the audio model ('muq', 'mert', or 'music2latent')
            device: Device to use for inference
            batch_size: Batch size for processing
            logger: Logger instance
        """
        self.model_name = model_name.lower()
        self.device = device
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger("AudioEmbeddingGenerator")

        # Load the appropriate audio embedding model
        self.model = self._load_model()
        self.embedding_dim = self._get_embedding_dim()

        self.logger.info(f"Loaded {model_name} model with embedding dim: {self.embedding_dim}")

    def _load_model(self):
        """Load the audio embedding model."""
        if self.model_name == "muq":
            return MuQEmbeddingModel(
                device=self.device,
                timeout_seconds=3600,  # 1 hour
            )
        elif self.model_name == "mert":
            return MertModel(
                device=self.device,
                timeout_seconds=3600,
            )
        elif self.model_name == "music2latent":
            return MusicLatentSpaceModel(
                device=self.device,
                timeout_seconds=3600,
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _get_embedding_dim(self) -> int:
        """Get the embedding dimension for the model."""
        if self.model_name == "muq":
            return 1536
        elif self.model_name == "mert":
            return 76800
        elif self.model_name == "music2latent":
            return 576
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def generate_embeddings_for_dataset(
        self,
        dataset_dir: Path,
        output_file: Path,
        max_samples: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Generate embeddings for all audio files in a dataset.

        Args:
            dataset_dir: Directory containing the dataset
            output_file: HDF5 file to save embeddings
            max_samples: Maximum number of samples to process

        Returns:
            Dictionary with generation statistics
        """
        dataset_dir = Path(dataset_dir)
        audio_dir = dataset_dir / "audio"
        metadata_dir = dataset_dir / "metadata"

        if not audio_dir.exists():
            raise ValueError(f"Audio directory not found: {audio_dir}")

        self.logger.info(f"Generating embeddings for dataset: {dataset_dir}")
        self.logger.info(f"Output file: {output_file}")

        # Find all audio files
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        audio_files = sorted([
            f for f in audio_dir.rglob('*')
            if f.is_file() and f.suffix.lower() in audio_extensions
        ])

        if max_samples:
            audio_files = audio_files[:max_samples]

        self.logger.info(f"Found {len(audio_files)} audio files")

        # Load metadata if available
        metadata = self._load_metadata(metadata_dir)

        # Generate embeddings
        stats = {
            'total_files': len(audio_files),
            'successful': 0,
            'failed': 0,
            'failed_files': [],
        }

        # Create HDF5 file for storing embeddings
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_file, 'w') as hf:
            # Create datasets
            embeddings_ds = hf.create_dataset(
                'embeddings',
                shape=(len(audio_files), self.embedding_dim),
                dtype='float32',
                chunks=True,
                compression='gzip',
            )

            # Create metadata datasets
            file_paths_ds = hf.create_dataset(
                'file_paths',
                shape=(len(audio_files),),
                dtype=h5py.string_dtype(encoding='utf-8'),
            )

            # Process files in batches
            for i in tqdm(range(0, len(audio_files), self.batch_size), desc="Processing batches"):
                batch_files = audio_files[i:i+self.batch_size]
                batch_embeddings = []
                batch_paths = []

                for audio_file in batch_files:
                    try:
                        # Generate embedding
                        embedding = self._generate_embedding(audio_file)
                        batch_embeddings.append(embedding)
                        batch_paths.append(str(audio_file.relative_to(dataset_dir)))
                        stats['successful'] += 1

                    except Exception as e:
                        self.logger.warning(f"Failed to process {audio_file}: {e}")
                        stats['failed'] += 1
                        stats['failed_files'].append(str(audio_file))
                        # Add zero embedding as placeholder
                        batch_embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                        batch_paths.append(str(audio_file.relative_to(dataset_dir)))

                # Save batch
                start_idx = i
                end_idx = min(i + len(batch_embeddings), len(audio_files))
                embeddings_ds[start_idx:end_idx] = np.array(batch_embeddings)
                file_paths_ds[start_idx:end_idx] = batch_paths

            # Save metadata
            hf.attrs['model_name'] = self.model_name
            hf.attrs['embedding_dim'] = self.embedding_dim
            hf.attrs['num_samples'] = len(audio_files)
            hf.attrs['dataset_dir'] = str(dataset_dir)

        self.logger.info(f"Embeddings saved to {output_file}")
        self.logger.info(f"Success rate: {stats['successful']}/{stats['total_files']}")

        return stats

    def _generate_embedding(self, audio_file: Path) -> np.ndarray:
        """Generate embedding for a single audio file."""
        import torchaudio

        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_file))

        # Generate embedding using the model's embed_audio_tensor method
        with torch.no_grad():
            embedding_tensor = self.model.embed_audio_tensor(
                waveform=waveform,
                sample_rate=sample_rate,
                apply_enrichment=True,
            )

        # Convert to numpy
        embedding = embedding_tensor.cpu().numpy()

        # Ensure correct shape
        if embedding.ndim > 1:
            embedding = embedding.flatten()

        assert embedding.shape[0] == self.embedding_dim, \
            f"Expected embedding dim {self.embedding_dim}, got {embedding.shape[0]}"

        return embedding.astype(np.float32)

    def _load_metadata(self, metadata_dir: Path) -> Optional[Dict]:
        """Load metadata from the dataset."""
        if not metadata_dir.exists():
            return None

        # Try to load different metadata files
        for filename in ['samples.json', 'processed_tracks.json', 'processed_metadata.json']:
            filepath = metadata_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load {filepath}: {e}")

        return None


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate audio embeddings for training')
    parser.add_argument('dataset_dir', type=str, help='Directory containing the dataset')
    parser.add_argument('output_file', type=str, help='Output HDF5 file for embeddings')
    parser.add_argument(
        '--model',
        type=str,
        default='muq',
        choices=['muq', 'mert', 'music2latent'],
        help='Audio embedding model to use'
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--max-samples', type=int, help='Maximum number of samples to process')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create generator
    generator = AudioEmbeddingGenerator(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Generate embeddings
    stats = generator.generate_embeddings_for_dataset(
        dataset_dir=Path(args.dataset_dir),
        output_file=Path(args.output_file),
        max_samples=args.max_samples,
    )

    print("\nGeneration complete!")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    if stats['failed_files']:
        print(f"Failed files saved to: {Path(args.output_file).parent / 'failed_files.json'}")
        with open(Path(args.output_file).parent / 'failed_files.json', 'w') as f:
            json.dump(stats['failed_files'], f, indent=2)


if __name__ == "__main__":
    main()
