"""Downloader for MusicBench dataset."""

from datetime import datetime
from pathlib import Path
from typing import Optional
import logging
import json
import soundfile as sf
import numpy as np
from tqdm import tqdm
import sys
import os

from ..base_downloader import BaseDownloader, DatasetMetadata


def _get_hf_load_dataset():
    """
    Import and return HuggingFace's load_dataset function.

    This needs to be done carefully to avoid conflicts with the local 'datasets' package.
    We keep the HuggingFace datasets module in sys.modules during usage to avoid
    pickle/multiprocessing errors.
    """
    _training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Temporarily filter out python_training from sys.path
    _filtered_path = [p for p in sys.path if os.path.abspath(p) != _training_dir and 'python_training' not in p and not p.startswith('.')]

    # Save the current state
    _original_path = sys.path[:]
    _original_datasets = sys.modules.get('datasets', None)

    # Temporarily update sys.path and remove local datasets from sys.modules
    sys.path[:] = _filtered_path
    if 'datasets' in sys.modules:
        del sys.modules['datasets']

    try:
        # Import HuggingFace datasets
        import datasets as hf_datasets
        # Keep the HuggingFace datasets module in sys.modules
        # Don't restore the original yet - we'll use it
        sys.path[:] = _original_path
        return hf_datasets.load_dataset
    except Exception as e:
        # Restore original state on error
        sys.path[:] = _original_path
        if _original_datasets:
            sys.modules['datasets'] = _original_datasets
        raise


class MusicBenchDownloader(BaseDownloader):
    """
    Downloader for MusicBench dataset from HuggingFace.

    MusicBench is a dataset for music generation evaluation containing:
    - ~50,000 music samples
    - Detailed text descriptions/prompts
    - Various music styles and genres
    """

    DATASET_NAME = "amaai-lab/MusicBench"

    def __init__(
        self,
        output_dir: Path,
        max_samples: Optional[int] = None,
        audio_format: str = "wav",
        sample_rate: int = 44100,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize MusicBench downloader.

        Args:
            output_dir: Directory where dataset will be saved
            max_samples: Maximum number of samples to download (None for all)
            audio_format: Audio format (wav, mp3, flac)
            sample_rate: Target sample rate
            logger: Logger instance
        """
        super().__init__(output_dir, logger)
        self.max_samples = max_samples
        self.audio_format = audio_format
        self.sample_rate = sample_rate

    def download(self) -> DatasetMetadata:
        """Download MusicBench dataset from HuggingFace."""
        self.logger.info(f"Loading MusicBench dataset from HuggingFace...")

        try:
            # Get HuggingFace load_dataset function
            load_dataset = _get_hf_load_dataset()

            # Load dataset from HuggingFace
            dataset = load_dataset(self.DATASET_NAME, split="train", streaming=False)

            if self.max_samples:
                dataset = dataset.select(range(min(self.max_samples, len(dataset))))

            self.logger.info(f"Found {len(dataset)} samples to download")

            # Process samples
            metadata_list = []
            total_duration = 0.0

            for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
                sample_id = f"{idx:06d}"

                # Extract audio
                audio_data = sample.get('audio', None)
                if audio_data is None:
                    self.logger.warning(f"No audio data for sample {idx}")
                    continue

                # audio_data is typically a dict with 'array' and 'sampling_rate'
                if isinstance(audio_data, dict):
                    waveform = audio_data['array']
                    source_sr = audio_data['sampling_rate']
                else:
                    waveform = audio_data
                    source_sr = self.sample_rate

                # Resample if needed
                if source_sr != self.sample_rate:
                    waveform = self._resample_audio(waveform, source_sr, self.sample_rate)

                # Save audio file
                audio_file = self.audio_dir / f"{sample_id}.{self.audio_format}"
                sf.write(str(audio_file), waveform, self.sample_rate)

                # Extract text description
                text = sample.get('caption', sample.get('text', sample.get('description', '')))

                # Calculate duration
                duration = len(waveform) / self.sample_rate
                total_duration += duration

                # Save metadata
                metadata_list.append({
                    'id': sample_id,
                    'text': text,
                    'duration': duration,
                    'sample_rate': self.sample_rate,
                    'genre': sample.get('genre', ''),
                    'mood': sample.get('mood', ''),
                    'tempo': sample.get('tempo', ''),
                })

            # Save metadata JSON
            with open(self.metadata_dir / "samples.json", 'w') as f:
                json.dump(metadata_list, f, indent=2)

            # Calculate total size
            total_size = sum(
                (self.audio_dir / f"{meta['id']}.{self.audio_format}").stat().st_size
                for meta in metadata_list
            )

            # Create dataset metadata
            metadata = DatasetMetadata(
                name="MusicBench",
                version="1.0",
                num_samples=len(metadata_list),
                total_size_bytes=total_size,
                download_date=datetime.now().isoformat(),
                source_url=f"https://huggingface.co/datasets/{self.DATASET_NAME}",
                format=self.audio_format,
                audio_duration_total_seconds=total_duration,
                additional_info={
                    'sample_rate': self.sample_rate,
                    'avg_duration': total_duration / len(metadata_list) if metadata_list else 0,
                }
            )

            self.save_metadata(metadata)
            return metadata

        except Exception as e:
            self.logger.error(f"Error downloading MusicBench: {e}")
            raise

    def _resample_audio(self, waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio using librosa."""
        import librosa
        return librosa.resample(waveform, orig_sr=source_sr, target_sr=target_sr)

    def verify(self) -> bool:
        """Verify the downloaded dataset."""
        metadata = self.load_metadata()
        if metadata is None:
            self.logger.error("No metadata found")
            return False

        # Check metadata file
        samples_file = self.metadata_dir / "samples.json"
        if not samples_file.exists():
            self.logger.error("samples.json not found")
            return False

        # Load samples
        with open(samples_file, 'r') as f:
            samples = json.load(f)

        # Verify audio files exist
        missing_files = []
        for sample in samples:
            audio_file = self.audio_dir / f"{sample['id']}.{self.audio_format}"
            if not audio_file.exists():
                missing_files.append(sample['id'])

        if missing_files:
            self.logger.error(f"Missing {len(missing_files)} audio files")
            return False

        self.logger.info("Dataset verification passed")
        return True


if __name__ == "__main__":
    # Test the downloader
    logging.basicConfig(level=logging.INFO)
    output_dir = Path("data/musicbench")
    downloader = MusicBenchDownloader(output_dir, max_samples=100)

    if not downloader.is_downloaded():
        metadata = downloader.download()
        print(f"Downloaded {metadata.num_samples} samples")

    if downloader.verify():
        print("Dataset verified successfully")
    else:
        print("Dataset verification failed")
