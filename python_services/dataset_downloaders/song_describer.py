"""
Song Describer Dataset downloader.

Dataset: ~1,100 songs with human-written captions
Source: https://zenodo.org/records/10072001
Paper: https://arxiv.org/abs/2311.10057
"""

import json
import logging
import subprocess
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
        return self.output_dir / "metadata.json"

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
        """Download metadata JSON from Zenodo"""
        metadata_path = self.get_metadata_path()

        # Check if metadata exists and is valid
        if metadata_path.exists() and self.resume:
            try:
                # Validate JSON is parseable
                with open(metadata_path, 'r') as f:
                    json.load(f)
                logger.info("Metadata already downloaded and valid")
                return True
            except (json.JSONDecodeError, IOError) as e:
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

            for file in files:
                if 'metadata' in file['key'].lower() or file['key'].endswith('.json'):
                    metadata_file = file
                    break

            if not metadata_file:
                logger.error("Metadata file not found in Zenodo record")
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
        """Download audio files from YouTube using yt-dlp"""
        metadata_path = self.get_metadata_path()

        if not metadata_path.exists():
            if self.dry_run:
                logger.info("[DRY RUN] Metadata not found, but would download ~1,100 audio tracks from YouTube")
                return True
            else:
                logger.error("Metadata file not found. Download metadata first.")
                return False

        # Load metadata with error handling
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Metadata file is corrupted: {e}")
            logger.error("Please delete the corrupted file and re-run the download:")
            logger.error(f"  rm {metadata_path}")
            logger.error(f"  python download_datasets.py --datasets song_describer")
            return False
        except IOError as e:
            logger.error(f"Failed to read metadata file: {e}")
            return False

        # Extract YouTube IDs
        audio_dir = self.output_dir / "audio"
        if not self.dry_run:
            audio_dir.mkdir(exist_ok=True)

        # Get list of tracks
        tracks = metadata if isinstance(metadata, list) else metadata.get('tracks', [])
        self.state.total_files = len(tracks)

        logger.info(f"Downloading audio for {len(tracks)} tracks from YouTube...")

        # Check yt-dlp availability
        if not self._check_ytdlp():
            logger.error("yt-dlp not found. Install with: pip install yt-dlp")
            return False

        # Download each track
        for track in tqdm(tracks, desc="Downloading audio"):
            youtube_id = track.get('youtube_id') or track.get('ytid') or track.get('id')

            if not youtube_id:
                logger.warning(f"No YouTube ID found for track: {track}")
                continue

            output_file = audio_dir / f"{youtube_id}.mp3"

            # Skip if already downloaded
            file_id = str(output_file.relative_to(self.output_dir))
            if file_id in self.state.downloaded_files:
                continue

            if self.dry_run:
                logger.info(f"[DRY RUN] Would download: {youtube_id}")
                continue

            # Download using yt-dlp
            success = self._download_youtube_audio(youtube_id, output_file)

            if success:
                self.state.downloaded_files.add(file_id)
                self.state.total_size_bytes += output_file.stat().st_size
            else:
                self.state.failed_files.add(file_id)

            # Save checkpoint every 10 files
            if len(self.state.downloaded_files) % 10 == 0:
                self.save_state()

        # Final save
        self.save_state()

        return True

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
├── metadata.json      # Song metadata and captions
├── audio/            # MP3 files (YouTube sourced)
│   ├── <youtube_id>.mp3
│   └── ...
└── README.md         # This file
```

## Metadata Format
Each entry contains:
- `youtube_id`: YouTube video ID
- `title`: Song title
- `artist`: Artist name
- `captions`: List of 5 human-written descriptions
- `duration`: Clip duration in seconds

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
Check individual track licenses. YouTube sourced content.

## Download Info
- Downloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Downloader: dataset_downloaders/song_describer.py
"""

        with open(readme_path, 'w') as f:
            f.write(readme_content)

        logger.info(f"README created at {readme_path}")
