"""
Clotho Dataset downloader.

Dataset: ~6,000 audio clips with captions (15-30s, general audio)
Source: https://zenodo.org/records/3490684
"""

import logging
from pathlib import Path
from .base import BaseDownloader

logger = logging.getLogger(__name__)


class ClothoDownloader(BaseDownloader):
    """Downloader for Clotho dataset"""

    DATASET_NAME = "clotho"
    ZENODO_RECORD_ID = "3490684"

    def get_name(self) -> str:
        return self.DATASET_NAME

    def get_metadata_path(self) -> Path:
        return self.output_dir / "metadata"

    def download(self) -> bool:
        """Download Clotho dataset"""
        logger.info("Clotho downloader - Implementation pending")
        logger.info(f"Download from: https://zenodo.org/records/{self.ZENODO_RECORD_ID}")
        return False
