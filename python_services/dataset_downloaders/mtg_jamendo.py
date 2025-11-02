"""
MTG-Jamendo Dataset downloader.

Dataset: 55,000 tracks with multi-label tags
Source: https://mtg.github.io/mtg-jamendo-dataset/
"""

import logging
from pathlib import Path
from .base import BaseDownloader

logger = logging.getLogger(__name__)


class MTGJamendoDownloader(BaseDownloader):
    """Downloader for MTG-Jamendo dataset"""

    DATASET_NAME = "mtg_jamendo"
    ZENODO_RECORD_ID = "3826813"

    def get_name(self) -> str:
        return self.DATASET_NAME

    def get_metadata_path(self) -> Path:
        return self.output_dir / "metadata"

    def download(self) -> bool:
        """Download MTG-Jamendo dataset"""
        logger.info("MTG-Jamendo downloader - Implementation pending")
        logger.info("Use: git clone https://github.com/MTG/mtg-jamendo-dataset.git")
        logger.info("Then follow their download instructions")
        return False
