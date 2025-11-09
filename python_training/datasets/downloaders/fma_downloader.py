"""Downloader for Free Music Archive (FMA) dataset."""

import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging
import json
import requests
import pandas as pd
from tqdm import tqdm

from ..base_downloader import BaseDownloader, DatasetMetadata


class FMADownloader(BaseDownloader):
    """
    Downloader for Free Music Archive (FMA) dataset.

    FMA is a dataset for music analysis containing 106,574 tracks from 16,341 artists
    and 14,854 albums, arranged in a hierarchical taxonomy of 161 genres.
    """

    BASE_URL = "https://os.unil.cloud.switch.ch/fma"
    DATASETS = {
        'small': {
            'size': '8000 tracks of 30s, 8 balanced genres (7.2 GiB)',
            'url': f'{BASE_URL}/fma_small.zip',
            'metadata': f'{BASE_URL}/fma_metadata.zip',
        },
        'medium': {
            'size': '25000 tracks of 30s, 16 unbalanced genres (22 GiB)',
            'url': f'{BASE_URL}/fma_medium.zip',
            'metadata': f'{BASE_URL}/fma_metadata.zip',
        },
        'large': {
            'size': '106574 tracks of 30s, 161 unbalanced genres (93 GiB)',
            'url': f'{BASE_URL}/fma_large.zip',
            'metadata': f'{BASE_URL}/fma_metadata.zip',
        },
        'full': {
            'size': '106574 untrimmed tracks, 161 unbalanced genres (~900 GiB)',
            'url': f'{BASE_URL}/fma_full.zip',
            'metadata': f'{BASE_URL}/fma_metadata.zip',
        },
    }

    def __init__(
        self,
        output_dir: Path,
        dataset_size: str = 'large',
        extract_clips: bool = True,
        clips_per_track: int = 3,
        clip_duration: int = 30,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize FMA downloader.

        Args:
            output_dir: Directory where dataset will be saved
            dataset_size: Size of dataset ('small', 'medium', 'large', 'full')
            extract_clips: Whether to extract multiple clips per track (for full dataset)
            clips_per_track: Number of clips to extract per track (if full dataset)
            clip_duration: Duration of each clip in seconds
            logger: Logger instance
        """
        super().__init__(output_dir, logger)
        self.dataset_size = dataset_size
        self.extract_clips = extract_clips
        self.clips_per_track = clips_per_track
        self.clip_duration = clip_duration

        if dataset_size not in self.DATASETS:
            raise ValueError(f"Invalid dataset size. Choose from: {list(self.DATASETS.keys())}")

        self.dataset_info = self.DATASETS[dataset_size]
        self.zip_dir = self.output_dir / "zip_files"
        self.zip_dir.mkdir(exist_ok=True)

    def download(self) -> DatasetMetadata:
        """Download FMA dataset."""
        self.logger.info(f"Downloading FMA {self.dataset_size} dataset...")
        self.logger.info(f"Size: {self.dataset_info['size']}")

        # Download metadata first
        metadata_zip = self.zip_dir / "fma_metadata.zip"
        if not metadata_zip.exists():
            self.logger.info("Downloading metadata...")
            self._download_file(self.dataset_info['metadata'], metadata_zip)
            self._extract_zip(metadata_zip, self.metadata_dir)

        # Download audio
        audio_zip = self.zip_dir / f"fma_{self.dataset_size}.zip"
        if not audio_zip.exists():
            self.logger.info("Downloading audio files...")
            self._download_file(self.dataset_info['url'], audio_zip)
            self._extract_zip(audio_zip, self.audio_dir)

        # Process metadata
        self.logger.info("Processing metadata...")
        tracks_df = self._load_metadata()

        # If full dataset and extract_clips is True, extract multiple clips
        if self.dataset_size == 'full' and self.extract_clips:
            self.logger.info("Extracting clips from full tracks...")
            num_clips = self._extract_clips_from_full(tracks_df)
            audio_duration = num_clips * self.clip_duration
        else:
            # For pre-segmented datasets (small, medium, large)
            num_clips = len(tracks_df)
            audio_duration = num_clips * 30  # All are 30s clips

        # Calculate total size
        total_size = sum(
            f.stat().st_size
            for f in self.audio_dir.rglob('*.mp3')
        )

        # Create dataset metadata
        metadata = DatasetMetadata(
            name=f"FMA_{self.dataset_size}",
            version="1.0",
            num_samples=num_clips,
            total_size_bytes=total_size,
            download_date=datetime.now().isoformat(),
            source_url=self.dataset_info['url'],
            format="mp3",
            audio_duration_total_seconds=audio_duration,
            additional_info={
                'dataset_size': self.dataset_size,
                'num_tracks': len(tracks_df),
                'clips_per_track': self.clips_per_track if self.extract_clips else 1,
                'clip_duration': self.clip_duration if self.extract_clips else 30,
            }
        )

        self.save_metadata(metadata)
        return metadata

    def _download_file(self, url: str, output_path: Path) -> None:
        """Download a file with progress bar."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            # Use a temporary file to avoid corruption
            temp_path = output_path.with_suffix('.tmp')

            with open(temp_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Verify downloaded size matches expected size
            downloaded_size = temp_path.stat().st_size
            if total_size > 0 and downloaded_size < total_size:
                raise ValueError(f"Incomplete download: {downloaded_size}/{total_size} bytes")

            # Move temp file to final location
            temp_path.replace(output_path)

        except Exception as e:
            # Clean up partial download
            if temp_path.exists():
                temp_path.unlink()
            raise Exception(f"Failed to download {url}: {e}") from e

    def _extract_zip(self, zip_path: Path, extract_to: Path) -> None:
        """Extract a zip file."""
        self.logger.info(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    def _load_metadata(self) -> pd.DataFrame:
        """Load and process FMA metadata."""
        tracks_csv = self.metadata_dir / "fma_metadata" / "tracks.csv"

        if not tracks_csv.exists():
            # Try alternative path
            tracks_csv = self.metadata_dir / "tracks.csv"

        # FMA metadata has multi-index columns
        tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

        # Save processed metadata as JSON
        metadata_list = []
        for track_id, row in tracks.iterrows():
            try:
                metadata_list.append({
                    'track_id': int(track_id),
                    'title': str(row[('track', 'title')]) if ('track', 'title') in row.index else '',
                    'artist': str(row[('artist', 'name')]) if ('artist', 'name') in row.index else '',
                    'album': str(row[('album', 'title')]) if ('album', 'title') in row.index else '',
                    'genre_top': str(row[('track', 'genre_top')]) if ('track', 'genre_top') in row.index else '',
                    'genres': str(row[('track', 'genres')]) if ('track', 'genres') in row.index else '[]',
                    'date_created': str(row[('track', 'date_created')]) if ('track', 'date_created') in row.index else '',
                })
            except Exception as e:
                self.logger.warning(f"Error processing track {track_id}: {e}")
                continue

        with open(self.metadata_dir / "processed_tracks.json", 'w') as f:
            json.dump(metadata_list, f, indent=2)

        return tracks

    def _extract_clips_from_full(self, tracks_df: pd.DataFrame) -> int:
        """Extract multiple clips from full-length tracks."""
        import subprocess
        from pydub import AudioSegment

        clips_dir = self.output_dir / "clips"
        clips_dir.mkdir(exist_ok=True)

        clip_metadata = []
        clip_count = 0

        for track_id, row in tqdm(list(tracks_df.iterrows()), desc="Extracting clips"):
            # Find the audio file
            # FMA organizes files in subdirectories: fma_full/000/000002.mp3
            track_id_str = f"{track_id:06d}"
            subdir = track_id_str[:3]
            audio_file = self.audio_dir / f"fma_full" / subdir / f"{track_id_str}.mp3"

            if not audio_file.exists():
                continue

            try:
                # Load audio to get duration
                audio = AudioSegment.from_mp3(str(audio_file))
                duration_seconds = len(audio) / 1000.0

                # Calculate clip positions
                if duration_seconds < self.clip_duration:
                    # Track too short, skip
                    continue

                # Extract evenly spaced clips
                interval = duration_seconds / (self.clips_per_track + 1)
                for i in range(self.clips_per_track):
                    start_time = interval * (i + 1)
                    clip_id = f"{track_id:06d}_{i:02d}"
                    output_file = clips_dir / f"{clip_id}.mp3"

                    # Use ffmpeg to extract clip
                    subprocess.run([
                        'ffmpeg',
                        '-i', str(audio_file),
                        '-ss', str(start_time),
                        '-t', str(self.clip_duration),
                        '-y',
                        str(output_file)
                    ], check=True, capture_output=True)

                    clip_metadata.append({
                        'clip_id': clip_id,
                        'track_id': int(track_id),
                        'clip_index': i,
                        'start_time': start_time,
                        'duration': self.clip_duration,
                    })
                    clip_count += 1

            except Exception as e:
                self.logger.warning(f"Error processing track {track_id}: {e}")
                continue

        # Save clip metadata
        with open(self.metadata_dir / "clips.json", 'w') as f:
            json.dump(clip_metadata, f, indent=2)

        return clip_count

    def verify(self) -> bool:
        """Verify the downloaded dataset."""
        metadata = self.load_metadata()
        if metadata is None:
            self.logger.error("No metadata found")
            return False

        # Check if audio directory exists and has files
        audio_files = list(self.audio_dir.rglob('*.mp3'))
        if not audio_files:
            self.logger.error("No audio files found")
            return False

        self.logger.info(f"Found {len(audio_files)} audio files")

        # Check metadata
        metadata_file = self.metadata_dir / "processed_tracks.json"
        if not metadata_file.exists():
            self.logger.error("Processed metadata not found")
            return False

        self.logger.info("Dataset verification passed")
        return True


if __name__ == "__main__":
    # Test the downloader
    logging.basicConfig(level=logging.INFO)
    output_dir = Path("data/fma")
    downloader = FMADownloader(output_dir, dataset_size='small')

    if not downloader.is_downloaded():
        metadata = downloader.download()
        print(f"Downloaded {metadata.num_samples} samples")

    if downloader.verify():
        print("Dataset verified successfully")
    else:
        print("Dataset verification failed")
