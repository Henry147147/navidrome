"""Base class for dataset downloaders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import logging
import json
from tqdm import tqdm


@dataclass
class DatasetMetadata:
    """Metadata for a downloaded dataset."""
    name: str
    version: str
    num_samples: int
    total_size_bytes: int
    download_date: str
    source_url: str
    format: str
    audio_duration_total_seconds: float
    additional_info: Dict[str, any] = None


class BaseDownloader(ABC):
    """Abstract base class for dataset downloaders."""

    def __init__(
        self,
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the downloader.

        Args:
            output_dir: Directory where dataset will be saved
            logger: Logger instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Create subdirectories
        self.audio_dir = self.output_dir / "audio"
        self.metadata_dir = self.output_dir / "metadata"
        self.audio_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

    @abstractmethod
    def download(self) -> DatasetMetadata:
        """
        Download the dataset.

        Returns:
            DatasetMetadata object with information about the downloaded dataset
        """
        pass

    @abstractmethod
    def verify(self) -> bool:
        """
        Verify the downloaded dataset is complete and valid.

        Returns:
            True if dataset is valid, False otherwise
        """
        pass

    def save_metadata(self, metadata: DatasetMetadata) -> None:
        """Save dataset metadata to JSON file."""
        metadata_path = self.metadata_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'name': metadata.name,
                'version': metadata.version,
                'num_samples': metadata.num_samples,
                'total_size_bytes': metadata.total_size_bytes,
                'download_date': metadata.download_date,
                'source_url': metadata.source_url,
                'format': metadata.format,
                'audio_duration_total_seconds': metadata.audio_duration_total_seconds,
                'additional_info': metadata.additional_info or {},
            }, f, indent=2)
        self.logger.info(f"Saved metadata to {metadata_path}")

    def load_metadata(self) -> Optional[DatasetMetadata]:
        """Load dataset metadata from JSON file."""
        metadata_path = self.metadata_dir / "dataset_metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            data = json.load(f)

        return DatasetMetadata(**data)

    def is_downloaded(self) -> bool:
        """Check if dataset has already been downloaded."""
        metadata = self.load_metadata()
        if metadata is None:
            return False

        # Basic check - can be overridden for more thorough checks
        return self.audio_dir.exists() and len(list(self.audio_dir.iterdir())) > 0

    def get_download_progress(self) -> Dict[str, any]:
        """Get current download progress."""
        if not self.audio_dir.exists():
            return {'downloaded': 0, 'total': 0, 'complete': False}

        metadata = self.load_metadata()
        if metadata is None:
            return {'downloaded': len(list(self.audio_dir.iterdir())), 'total': 0, 'complete': False}

        downloaded = len(list(self.audio_dir.iterdir()))
        return {
            'downloaded': downloaded,
            'total': metadata.num_samples,
            'complete': downloaded >= metadata.num_samples,
        }
