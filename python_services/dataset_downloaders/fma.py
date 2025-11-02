"""
Free Music Archive (FMA) Dataset downloader.

Dataset: 106,574 tracks with rich metadata
Source: https://github.com/mdeff/fma
Paper: https://arxiv.org/abs/1612.01840
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .base import BaseDownloader


logger = logging.getLogger(__name__)


class FMADownloader(BaseDownloader):
    """Downloader for FMA dataset"""

    DATASET_NAME = "fma"

    # Download URLs from https://github.com/mdeff/fma
    URLS = {
        'metadata': 'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip',
        'small': 'https://os.unil.cloud.switch.ch/fma/fma_small.zip',  # 8K tracks, 7.2 GB
        'medium': 'https://os.unil.cloud.switch.ch/fma/fma_medium.zip',  # 25K tracks, 22 GB
        'large': 'https://os.unil.cloud.switch.ch/fma/fma_large.zip',  # 106K tracks, 93 GB
        'full': 'https://os.unil.cloud.switch.ch/fma/fma_full.zip',  # 106K tracks, 879 GB (lossless)
    }

    SIZES_GB = {
        'metadata': 0.5,
        'small': 7.2,
        'medium': 22,
        'large': 93,
        'full': 879
    }

    def __init__(self, output_dir: str, size: str = 'small', **kwargs):
        """
        Initialize FMA downloader.

        Args:
            output_dir: Output directory
            size: Dataset size ('small', 'medium', 'large', or 'full')
            **kwargs: Additional arguments for BaseDownloader
        """
        # Set size BEFORE calling super().__init__() because get_name() needs it
        if size not in self.URLS:
            raise ValueError(f"Invalid size: {size}. Choose from: {list(self.URLS.keys())}")

        self.size = size

        super().__init__(output_dir, **kwargs)

        logger.info(f"FMA dataset size: {size} (~{self.SIZES_GB[size]} GB)")

    def get_name(self) -> str:
        return f"{self.DATASET_NAME}_{self.size}"

    def get_metadata_path(self) -> Path:
        return self.output_dir / "fma_metadata"

    def download(self) -> bool:
        """Download FMA dataset"""
        logger.info(f"Starting FMA ({self.size}) download...")

        # Check disk space
        required_bytes = self.SIZES_GB['metadata'] * 1024**3
        if self.size != 'metadata':
            required_bytes += self.SIZES_GB[self.size] * 1024**3

        if not self.check_disk_space(int(required_bytes * 1.1)):  # 10% buffer
            return False

        # Download metadata
        if not self._download_metadata():
            return False

        # Download audio (if not just metadata)
        if self.size != 'metadata' and not self._download_audio():
            return False

        # Create README
        self._create_readme()

        # Print summary
        self.print_summary()

        # Clean up checkpoint
        if len(self.state.failed_files) == 0:
            self.delete_state()

        return True

    def _download_metadata(self) -> bool:
        """Download metadata ZIP file"""
        metadata_zip = self.output_dir / "fma_metadata.zip"

        if metadata_zip.exists() and self.resume:
            logger.info("Metadata already downloaded")
            return True

        logger.info("Downloading FMA metadata...")

        success = self.download_file(
            self.URLS['metadata'],
            metadata_zip,
            description="FMA Metadata"
        )

        if success and not self.dry_run:
            # Extract ZIP
            logger.info("Extracting metadata...")
            import zipfile
            with zipfile.ZipFile(metadata_zip, 'r') as zip_ref:
                zip_ref.extractall(self.output_dir)

            logger.info("Metadata extracted")

        return success

    def _download_audio(self) -> bool:
        """Download audio ZIP file"""
        audio_zip = self.output_dir / f"fma_{self.size}.zip"

        if audio_zip.exists() and self.resume:
            logger.info(f"Audio ({self.size}) already downloaded")
            return True

        logger.info(f"Downloading FMA audio ({self.size})...")
        logger.info(f"This will download ~{self.SIZES_GB[self.size]} GB")

        success = self.download_file(
            self.URLS[self.size],
            audio_zip,
            description=f"FMA {self.size.title()}"
        )

        if success and not self.dry_run:
            # Extract ZIP
            logger.info("Extracting audio files...")
            logger.info("This may take a while...")

            import zipfile
            with zipfile.ZipFile(audio_zip, 'r') as zip_ref:
                # Extract with progress
                members = zip_ref.namelist()
                for member in tqdm(members, desc="Extracting"):
                    zip_ref.extract(member, self.output_dir)

            logger.info("Audio extracted")

        return success

    def _create_readme(self):
        """Create README file"""
        readme_path = self.output_dir / "README.md"

        if self.dry_run:
            return

        readme_content = f"""# Free Music Archive (FMA) Dataset

## Overview
Large-scale dataset of music tracks with rich metadata.

- **Size**: {self.size} ({self.SIZES_GB.get(self.size, 0)} GB)
- **Metadata**: Genre, artist, album, tags, audio features
- **License**: Creative Commons
- **Source**: https://github.com/mdeff/fma

## Dataset Sizes
- Small: 8,000 tracks (7.2 GB) - balanced genres
- Medium: 25,000 tracks (22 GB) - unbalanced genres
- Large: 106,574 tracks (93 GB) - all tracks, lossy quality
- Full: 106,574 tracks (879 GB) - all tracks, lossless quality

## Directory Structure
```
fma/
├── fma_metadata/     # CSV files with metadata
│   ├── tracks.csv    # Track metadata
│   ├── genres.csv    # Genre hierarchy
│   ├── features.csv  # Audio features
│   └── ...
├── fma_{self.size}/  # MP3 files organized by ID
└── README.md
```

## Citation
```bibtex
@inproceedings{{fma_dataset,
  title={{FMA: A Dataset for Music Analysis}},
  author={{Defferrard, Michaël and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier}},
  booktitle={{18th International Society for Music Information Retrieval Conference}},
  year={{2017}}
}}
```

## Download Info
- Downloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Downloader: dataset_downloaders/fma.py
"""

        with open(readme_path, 'w') as f:
            f.write(readme_content)
