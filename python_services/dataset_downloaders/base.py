"""
Base downloader class with common functionality for all dataset downloaders.
"""

import os
import json
import logging
import hashlib
import tempfile
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime

import requests
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass
class DownloadState:
    """State for resuming downloads"""
    dataset_name: str
    downloaded_files: Set[str] = field(default_factory=set)
    failed_files: Set[str] = field(default_factory=set)
    total_files: int = 0
    total_size_bytes: int = 0
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            'dataset_name': self.dataset_name,
            'downloaded_files': list(self.downloaded_files),
            'failed_files': list(self.failed_files),
            'total_files': self.total_files,
            'total_size_bytes': self.total_size_bytes,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DownloadState':
        return cls(
            dataset_name=data['dataset_name'],
            downloaded_files=set(data.get('downloaded_files', [])),
            failed_files=set(data.get('failed_files', [])),
            total_files=data.get('total_files', 0),
            total_size_bytes=data.get('total_size_bytes', 0),
            timestamp=data.get('timestamp', '')
        )


class BaseDownloader(ABC):
    """
    Base class for dataset downloaders with common functionality.
    """

    def __init__(
        self,
        output_dir: str,
        resume: bool = True,
        verify: bool = True,
        dry_run: bool = False
    ):
        """
        Initialize downloader.

        Args:
            output_dir: Directory to save downloaded files
            resume: Whether to resume interrupted downloads
            verify: Whether to verify downloaded files
            dry_run: If True, don't actually download, just show what would be downloaded
        """
        self.output_dir = Path(output_dir)
        self.resume = resume
        self.verify = verify
        self.dry_run = dry_run

        # Create output directory
        if not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint management
        self.checkpoint_file = self.output_dir / '.download_state.json'
        self.state: Optional[DownloadState] = None

        # Load state if resuming
        if resume and self.checkpoint_file.exists():
            self.load_state()
        else:
            self.state = DownloadState(dataset_name=self.get_name())

        logger.info(f"Initialized {self.get_name()} downloader")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Resume: {resume}")
        logger.info(f"  Dry run: {dry_run}")

    @abstractmethod
    def get_name(self) -> str:
        """Return dataset name"""
        pass

    @abstractmethod
    def download(self) -> bool:
        """
        Download the dataset.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def get_metadata_path(self) -> Path:
        """Return path to dataset metadata file/directory"""
        pass

    def save_state(self):
        """Save download state to checkpoint file"""
        if self.dry_run:
            return

        self.state.timestamp = datetime.now().isoformat()

        try:
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
            shutil.move(temp_file, self.checkpoint_file)
            logger.debug(f"Saved checkpoint for {self.get_name()}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_state(self):
        """Load download state from checkpoint file"""
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            self.state = DownloadState.from_dict(data)
            logger.info(f"Loaded checkpoint from {self.state.timestamp}")
            logger.info(f"  Downloaded: {len(self.state.downloaded_files)} files")
            logger.info(f"  Failed: {len(self.state.failed_files)} files")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            self.state = DownloadState(dataset_name=self.get_name())

    def delete_state(self):
        """Delete checkpoint file"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint deleted")

    def download_file(
        self,
        url: str,
        output_path: Path,
        description: str = "",
        chunk_size: int = 8192
    ) -> bool:
        """
        Download a single file with progress bar.

        Args:
            url: URL to download from
            output_path: Path to save file
            description: Description for progress bar
            chunk_size: Download chunk size in bytes

        Returns:
            True if successful, False otherwise
        """
        # Check if already downloaded
        file_id = str(output_path.relative_to(self.output_dir))
        if file_id in self.state.downloaded_files:
            logger.debug(f"Skipping {file_id} (already downloaded)")
            return True

        if self.dry_run:
            logger.info(f"[DRY RUN] Would download: {url} -> {output_path}")
            return True

        try:
            # Create parent directory
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Download with progress bar
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=description or file_id,
                    leave=False
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Mark as downloaded
            self.state.downloaded_files.add(file_id)
            self.state.total_size_bytes += output_path.stat().st_size
            self.save_state()

            return True

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            self.state.failed_files.add(file_id)
            self.save_state()

            # Clean up partial download
            if output_path.exists():
                output_path.unlink()

            return False

    def verify_file(self, file_path: Path, expected_hash: Optional[str] = None) -> bool:
        """
        Verify a downloaded file.

        Args:
            file_path: Path to file
            expected_hash: Expected MD5 hash (optional)

        Returns:
            True if file exists and hash matches (if provided)
        """
        if not file_path.exists():
            return False

        if expected_hash is None:
            return True

        # Compute MD5 hash
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)

        actual_hash = md5.hexdigest()
        return actual_hash == expected_hash

    def get_download_summary(self) -> Dict:
        """Get summary of download progress"""
        return {
            'dataset': self.get_name(),
            'total_files': self.state.total_files,
            'downloaded': len(self.state.downloaded_files),
            'failed': len(self.state.failed_files),
            'total_size_mb': self.state.total_size_bytes / (1024 * 1024),
            'success_rate': len(self.state.downloaded_files) / max(self.state.total_files, 1) * 100
        }

    def print_summary(self):
        """Print download summary"""
        summary = self.get_download_summary()

        logger.info(f"\n{'='*60}")
        logger.info(f"{self.get_name()} Download Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Total files: {summary['total_files']}")
        logger.info(f"Downloaded: {summary['downloaded']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Success rate: {summary['success_rate']:.2f}%")
        logger.info(f"Total size: {summary['total_size_mb']:.2f} MB")
        logger.info(f"{'='*60}\n")

    def check_disk_space(self, required_bytes: int) -> bool:
        """
        Check if enough disk space is available.

        Args:
            required_bytes: Required space in bytes

        Returns:
            True if enough space available
        """
        if self.dry_run:
            return True

        stat = shutil.disk_usage(self.output_dir)
        available = stat.free

        if available < required_bytes:
            required_gb = required_bytes / (1024**3)
            available_gb = available / (1024**3)
            logger.error(f"Insufficient disk space!")
            logger.error(f"  Required: {required_gb:.2f} GB")
            logger.error(f"  Available: {available_gb:.2f} GB")
            return False

        return True
