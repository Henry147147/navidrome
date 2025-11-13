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
        pause_handler=None,
        state_manager=None,
    ):
        """
        Initialize audio embedding generator.

        Args:
            model_name: Name of the audio model ('muq', 'mert', or 'music2latent')
            device: Device to use for inference
            batch_size: Batch size for processing
            logger: Logger instance
            pause_handler: Handler for pause requests
            state_manager: Manager for saving/loading state
        """
        self.model_name = model_name.lower()
        self.device = device
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger("AudioEmbeddingGenerator")
        self.pause_handler = pause_handler
        self.state_manager = state_manager

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
        dataset_name: str,
        max_samples: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Generate embeddings for all audio files in a dataset.

        Args:
            dataset_dir: Directory containing the dataset
            output_file: HDF5 file to save embeddings
            dataset_name: Name of the dataset (for state tracking)
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

        # Check if we should resume from a previous run
        start_index = 0
        file_mode = 'w'
        if self.state_manager:
            should_resume, resume_index = self.state_manager.should_resume(dataset_name, self.model_name)
            if should_resume and output_file.exists():
                start_index = resume_index
                file_mode = 'r+'  # Read/write mode for existing file
                self.logger.info(f"Resuming from track {start_index}")

        # Load metadata if available
        metadata = self._load_metadata(metadata_dir)

        # Generate embeddings
        stats = {
            'total_files': len(audio_files),
            'successful': 0,
            'failed': 0,
            'failed_files': [],
            'paused': False,
        }

        # Create HDF5 file for storing embeddings
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_file, file_mode) as hf:
            # Create or access datasets
            if file_mode == 'w':
                embeddings_ds = hf.create_dataset(
                    'embeddings',
                    shape=(len(audio_files), self.embedding_dim),
                    dtype='float32',
                    chunks=True,
                    compression='gzip',
                )

                file_paths_ds = hf.create_dataset(
                    'file_paths',
                    shape=(len(audio_files),),
                    dtype=h5py.string_dtype(encoding='utf-8'),
                )
            else:
                # Access existing datasets (when resuming)
                embeddings_ds = hf['embeddings']
                file_paths_ds = hf['file_paths']

            # Process files one at a time to allow pause after each track
            for i in tqdm(range(start_index, len(audio_files)), desc="Processing tracks", initial=start_index, total=len(audio_files)):
                audio_file = audio_files[i]

                try:
                    # Generate embedding
                    embedding = self._generate_embedding(audio_file)
                    embeddings_ds[i] = embedding
                    file_paths_ds[i] = str(audio_file.relative_to(dataset_dir))
                    stats['successful'] += 1

                except Exception as e:
                    self.logger.warning(f"Failed to process {audio_file}: {e}")
                    stats['failed'] += 1
                    stats['failed_files'].append(str(audio_file))
                    # Add zero embedding as placeholder
                    embeddings_ds[i] = np.zeros(self.embedding_dim, dtype=np.float32)
                    file_paths_ds[i] = str(audio_file.relative_to(dataset_dir))

                # Update progress in state manager
                if self.state_manager:
                    self.state_manager.update_progress(i)

                # Check for pause after each track
                if self.pause_handler and self.pause_handler.is_paused():
                    self.logger.warning(f"Paused after processing track {i+1}/{len(audio_files)}")
                    stats['paused'] = True
                    break

            # Save metadata only if this is a new file or we completed all tracks
            if file_mode == 'w' or (not stats['paused']):
                hf.attrs['model_name'] = self.model_name
                hf.attrs['embedding_dim'] = self.embedding_dim
                hf.attrs['num_samples'] = len(audio_files)
                hf.attrs['dataset_dir'] = str(dataset_dir)

        if not stats['paused']:
            self.logger.info(f"Embeddings saved to {output_file}")
            self.logger.info(f"Success rate: {stats['successful']}/{stats['total_files']}")
        else:
            self.logger.warning(f"Paused - Progress saved. Processed {stats['successful']}/{stats['total_files']} tracks")

        return stats

    def _generate_embedding(self, audio_file: Path) -> np.ndarray:
        """Generate embedding for a single audio file."""
        import soundfile as sf

        # Load audio using soundfile (more reliable than torchaudio)
        audio_data, sample_rate = sf.read(str(audio_file), dtype='float32')

        # Convert to torch tensor
        # If stereo, convert to mono by averaging channels
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Convert to torch tensor and add batch dimension
        waveform = torch.from_numpy(audio_data).unsqueeze(0)

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
