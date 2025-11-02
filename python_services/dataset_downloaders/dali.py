"""
DALI Dataset downloader.

Dataset: 5,358 songs with time-aligned lyrics
Source: https://github.com/gabolsgabs/DALI
"""

import logging
from pathlib import Path
from .base import BaseDownloader

logger = logging.getLogger(__name__)


class DALIDownloader(BaseDownloader):
    """Downloader for DALI dataset"""

    DATASET_NAME = "dali"

    def get_name(self) -> str:
        return self.DATASET_NAME

    def get_metadata_path(self) -> Path:
        return self.output_dir / "annotations"

    def download(self) -> bool:
        """Download DALI dataset"""
        logger.info("DALI downloader - Implementation pending")
        logger.info("Use: git clone https://github.com/gabolsgabs/DALI.git")
        return False
