"""
Song Describer Dataset downloader.

Dataset: ~1,100 songs with human-written captions
Source: https://zenodo.org/records/10072001
Paper: https://arxiv.org/abs/2311.10057
"""

import csv
import json
import logging
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

from .base import BaseDownloader


logger = logging.getLogger(__name__)


class SongDescriberDownloader(BaseDownloader):
    """Downloader for Song Describer dataset"""

    ZENODO_RECORD_ID = "10072001"
    DATASET_NAME = "song_describer"

    def get_name(self) -> str:
        return self.DATASET_NAME

    def get_metadata_path(self) -> Path:
        return self.output_dir / "metadata.csv"

    def download(self) -> bool:
        """
        Download Song Describer dataset.

        Returns:
            True if successful
        """
        logger.info(f"Starting {self.DATASET_NAME} download...")

        # Step 1: Download metadata from Zenodo
        if not self._download_metadata():
            return False

        # Step 2: Download audio from YouTube
        if not self._download_audio():
            return False

        # Step 3: Create README
        self._create_readme()

        # Print summary
        self.print_summary()

        # Clean up checkpoint on success
        if len(self.state.failed_files) == 0:
            self.delete_state()

        return True

    def _download_metadata(self) -> bool:
        """Download metadata CSV from Zenodo"""
        metadata_path = self.get_metadata_path()

        # Check if metadata exists and is valid
        if metadata_path.exists() and self.resume:
            try:
                # Validate CSV is parseable and has content
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    # Check for expected columns
                    if 'caption_id' in header or 'track_id' in header:
                        logger.info("Metadata already downloaded and valid")
                        return True
                    else:
                        logger.warning(f"Metadata file has unexpected format: {header}")
                        logger.info("Re-downloading metadata...")
            except (csv.Error, StopIteration, IOError) as e:
                logger.warning(f"Existing metadata file is corrupted: {e}")
                logger.info("Re-downloading metadata...")
                # Continue to re-download

        if self.dry_run:
            logger.info("[DRY RUN] Would download metadata from Zenodo")
            return True

        logger.info("Downloading metadata from Zenodo...")

        try:
            # Get Zenodo record information
            zenodo_api_url = f"https://zenodo.org/api/records/{self.ZENODO_RECORD_ID}"
            response = requests.get(zenodo_api_url, timeout=30)
            response.raise_for_status()

            record = response.json()

            # Find the metadata file in the record
            files = record.get('files', [])
            metadata_file = None

            # Look specifically for song_describer.csv
            for file in files:
                if file['key'] == 'song_describer.csv':
                    metadata_file = file
                    break

            if not metadata_file:
                logger.error("song_describer.csv not found in Zenodo record")
                logger.error("Available files:")
                for file in files:
                    logger.error(f"  - {file['key']}")
                return False

            # Download metadata file
            metadata_url = metadata_file['links']['self']
            success = self.download_file(
                metadata_url,
                metadata_path,
                description="Metadata"
            )

            if success:
                logger.info(f"Metadata saved to {metadata_path}")

            return success

        except Exception as e:
            logger.error(f"Failed to download metadata: {e}")
            return False

    def _download_audio(self) -> bool:
        """Download audio files from Zenodo audio.zip"""
        metadata_path = self.get_metadata_path()

        if not metadata_path.exists():
            if self.dry_run:
                logger.info("[DRY RUN] Metadata not found, but would download ~1,100 audio tracks")
                return True
            else:
                logger.error("Metadata file not found. Download metadata first.")
                return False

        # Load metadata with error handling
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                tracks = list(reader)
        except Exception as e:
            logger.error(f"Failed to read metadata CSV: {e}")
            logger.error("Please delete the corrupted file and re-run the download:")
            logger.error(f"  rm {metadata_path}")
            logger.error(f"  python download_datasets.py --datasets song_describer")
            return False

        self.state.total_files = len(tracks)
        logger.info(f"Found {len(tracks)} tracks in metadata")

        # Download audio.zip from Zenodo
        audio_dir = self.output_dir / "audio"
        if not self.dry_run:
            audio_dir.mkdir(exist_ok=True)

        # Get Zenodo record to find audio.zip
        try:
            zenodo_api_url = f"https://zenodo.org/api/records/{self.ZENODO_RECORD_ID}"
            response = requests.get(zenodo_api_url, timeout=30)
            response.raise_for_status()
            record = response.json()

            # Find audio.zip
            files = record.get('files', [])
            audio_zip_file = None
            for file in files:
                if file['key'] == 'audio.zip':
                    audio_zip_file = file
                    break

            if not audio_zip_file:
                logger.error("audio.zip not found in Zenodo record")
                return False

            # Download audio.zip
            audio_zip_path = self.output_dir / "audio.zip"
            audio_zip_url = audio_zip_file['links']['self']

            # Check if we need to download
            needs_download = True
            if audio_zip_path.exists() and self.resume:
                # Validate the zip file
                try:
                    with zipfile.ZipFile(audio_zip_path, 'r') as zf:
                        # Test if it's a valid zip
                        if len(zf.filelist) > 0:
                            logger.info("audio.zip already downloaded and valid")
                            needs_download = False
                        else:
                            logger.warning("audio.zip is empty, re-downloading...")
                            audio_zip_path.unlink()
                except zipfile.BadZipFile:
                    logger.warning("audio.zip is corrupted, re-downloading...")
                    audio_zip_path.unlink()
                except Exception as e:
                    logger.warning(f"Could not validate audio.zip: {e}, re-downloading...")
                    audio_zip_path.unlink()

            if needs_download:
                logger.info(f"Downloading audio.zip ({audio_zip_file['size'] / (1024**3):.2f} GB)...")
                success = self.download_file(
                    audio_zip_url,
                    audio_zip_path,
                    description="Audio archive"
                )
                if not success:
                    logger.error("Failed to download audio.zip")
                    return False

            # Extract audio.zip
            if self.dry_run:
                logger.info("[DRY RUN] Would extract audio.zip")
                return True

            # Check if already extracted
            extracted_marker = audio_dir / ".extracted"
            if extracted_marker.exists():
                logger.info("Audio files already extracted")
            else:
                logger.info("Extracting audio files...")
                with zipfile.ZipFile(audio_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(audio_dir)

                # Create marker file
                extracted_marker.touch()
                logger.info(f"Extracted audio files to {audio_dir}")

                # Clean up zip file after extraction
                audio_zip_path.unlink()
                logger.info("Removed audio.zip after extraction")

            return True

        except Exception as e:
            logger.error(f"Failed to download audio: {e}", exc_info=True)
            return False

    def _check_ytdlp(self) -> bool:
        """Check if yt-dlp is installed"""
        try:
            result = subprocess.run(
                ['yt-dlp', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _download_youtube_audio(self, youtube_id: str, output_file: Path) -> bool:
        """
        Download audio from YouTube using yt-dlp.

        Args:
            youtube_id: YouTube video ID
            output_file: Path to save audio file

        Returns:
            True if successful
        """
        url = f"https://www.youtube.com/watch?v={youtube_id}"

        try:
            # yt-dlp command
            cmd = [
                'yt-dlp',
                '--extract-audio',
                '--audio-format', 'mp3',
                '--audio-quality', '0',  # Best quality
                '--output', str(output_file.with_suffix('')),  # yt-dlp adds .mp3
                '--no-playlist',
                '--quiet',
                '--no-warnings',
                url
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120  # 2 minute timeout
            )

            if result.returncode == 0 and output_file.exists():
                return True
            else:
                logger.warning(f"Failed to download {youtube_id}: {result.stderr.decode()}")
                return False

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout downloading {youtube_id}")
            return False
        except Exception as e:
            logger.warning(f"Error downloading {youtube_id}: {e}")
            return False

    def _create_readme(self):
        """Create README file with dataset information"""
        readme_path = self.output_dir / "README.md"

        if self.dry_run:
            return

        readme_content = f"""# Song Describer Dataset

## Overview
Human-written captions for music tracks (~2 minute clips).

- **Size**: ~1,100 songs
- **Captions**: 5 human-written descriptions per song
- **Duration**: ~2 minutes per clip
- **Source**: Zenodo Record {self.ZENODO_RECORD_ID}
- **Paper**: https://arxiv.org/abs/2311.10057

## Directory Structure
```
{self.DATASET_NAME}/
├── metadata.csv      # Song metadata and captions (CSV format)
├── audio/            # MP3 files from Zenodo
│   ├── <track_id>.mp3
│   └── ...
└── README.md         # This file
```

## Metadata Format (CSV)
Each row contains:
- `caption_id`: Unique caption identifier
- `track_id`: Track identifier
- `caption`: Human-written description
- `is_valid_subset`: Whether track is in validation set
- `familiarity`: Familiarity rating (0-2)
- `artist_id`: Artist identifier
- `album_id`: Album identifier
- `path`: Relative path to audio file
- `duration`: Track duration in seconds

## Citation
```bibtex
@article{{song_describer_2023,
  title={{Song Describer Dataset: A Corpus for Generating Descriptive Text About Music}},
  author={{...}},
  journal={{arXiv preprint arXiv:2311.10057}},
  year={{2023}}
}}
```

## License
Check individual track licenses. Audio sourced from MTG-Jamendo dataset via Zenodo.

## Download Info
- Downloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Downloader: dataset_downloaders/song_describer.py
"""

        with open(readme_path, 'w') as f:
            f.write(readme_content)

        logger.info(f"README created at {readme_path}")
