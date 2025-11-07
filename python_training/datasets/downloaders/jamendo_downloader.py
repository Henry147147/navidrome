"""Downloader for MTG-Jamendo dataset."""

import os
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging
import json
import requests
import subprocess
from tqdm import tqdm
import csv

from ..base_downloader import BaseDownloader, DatasetMetadata


class JamendoDownloader(BaseDownloader):
    """
    Downloader for MTG-Jamendo dataset.

    MTG-Jamendo is a dataset for automatic music tagging with:
    - 55,000 full audio tracks from Jamendo
    - Multiple tags: genre, instrument, mood/theme
    - High quality metadata
    """

    BASE_URL = "https://essentia.upf.edu/documentation/datasets/mtg-jamendo/"
    AUDIO_URL = "http://mtg.upf.edu/download/datasets/mtg-jamendo/raw_30s.zip"
    METADATA_URL_BASE = "https://github.com/MTG/mtg-jamendo-dataset/raw/master/data"

    def __init__(
        self,
        output_dir: Path,
        max_samples: Optional[int] = None,
        clip_duration: int = 30,
        sample_rate: int = 44100,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize MTG-Jamendo downloader.

        Args:
            output_dir: Directory where dataset will be saved
            max_samples: Maximum number of samples to download (None for all)
            clip_duration: Duration of clips in seconds (30s pre-segmented)
            sample_rate: Target sample rate
            logger: Logger instance
        """
        super().__init__(output_dir, logger)
        self.max_samples = max_samples
        self.clip_duration = clip_duration
        self.sample_rate = sample_rate
        self.zip_dir = self.output_dir / "zip_files"
        self.zip_dir.mkdir(exist_ok=True)

    def download(self) -> DatasetMetadata:
        """Download MTG-Jamendo dataset."""
        self.logger.info("Downloading MTG-Jamendo dataset...")

        # Download metadata files
        self.logger.info("Downloading metadata...")
        self._download_metadata()

        # Download audio (30s clips)
        # Note: The full dataset is very large, we might want to download selectively
        self.logger.info("Downloading audio clips...")
        audio_metadata = self._download_audio_selective()

        # Process and merge metadata
        self.logger.info("Processing metadata...")
        track_metadata = self._process_metadata(audio_metadata)

        # Calculate statistics
        num_samples = len(track_metadata)
        total_size = sum(
            (self.audio_dir / f"{meta['track_id']}.mp3").stat().st_size
            for meta in track_metadata
            if (self.audio_dir / f"{meta['track_id']}.mp3").exists()
        )

        # Create dataset metadata
        metadata = DatasetMetadata(
            name="MTG_Jamendo",
            version="1.0",
            num_samples=num_samples,
            total_size_bytes=total_size,
            download_date=datetime.now().isoformat(),
            source_url=self.BASE_URL,
            format="mp3",
            audio_duration_total_seconds=num_samples * self.clip_duration,
            additional_info={
                'sample_rate': self.sample_rate,
                'clip_duration': self.clip_duration,
                'tags_types': ['genre', 'instrument', 'mood_theme'],
            }
        )

        self.save_metadata(metadata)
        return metadata

    def _download_metadata(self) -> None:
        """Download metadata CSV files from GitHub."""
        metadata_files = [
            'tags.tsv',
            'splits/split-0/autotagging_genre-train.tsv',
            'splits/split-0/autotagging_genre-validation.tsv',
            'splits/split-0/autotagging_genre-test.tsv',
            'splits/split-0/autotagging_instrument-train.tsv',
            'splits/split-0/autotagging_instrument-validation.tsv',
            'splits/split-0/autotagging_moodtheme-train.tsv',
        ]

        for file_path in metadata_files:
            url = f"{self.METADATA_URL_BASE}/{file_path}"
            output_path = self.metadata_dir / file_path.replace('/', '_')
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if not output_path.exists():
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    self.logger.info(f"Downloaded {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to download {file_path}: {e}")

    def _download_audio_selective(self) -> list:
        """
        Download audio files selectively from Jamendo API.

        Since the full dataset is very large (~200GB), we download tracks
        directly from Jamendo based on the track IDs in the metadata.
        """
        # Load track IDs from metadata
        track_ids = set()
        # Updated glob pattern to match the saved filenames
        metadata_files = list(self.metadata_dir.glob("*autotagging_*-train.tsv"))

        for meta_file in metadata_files:
            with open(meta_file, 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    if 'TRACK_ID' in row:
                        track_ids.add(row['TRACK_ID'])
                    elif 'track_id' in row:
                        track_ids.add(row['track_id'])

        # Limit if max_samples is set
        if self.max_samples:
            track_ids = list(track_ids)[:self.max_samples]

        self.logger.info(f"Found {len(track_ids)} tracks to download")

        # Download from Jamendo
        # Note: This requires Jamendo API access or using the direct download link
        # For simplicity, we'll create placeholder logic
        downloaded_tracks = []

        for track_id in tqdm(list(track_ids), desc="Downloading tracks"):
            output_file = self.audio_dir / f"{track_id}.mp3"

            if output_file.exists():
                downloaded_tracks.append({'track_id': track_id})
                continue

            try:
                # Jamendo audio URL format: https://mp3d.jamendo.com/?trackid={id}&format=mp32
                # Note: This is an example - actual API access may differ
                url = f"https://mp3d.jamendo.com/?trackid={track_id}&format=mp32"

                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(output_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    # Convert to target sample rate if needed
                    if self.sample_rate != 44100:
                        self._resample_audio(output_file)

                    downloaded_tracks.append({'track_id': track_id})
                else:
                    self.logger.warning(f"Failed to download track {track_id}: HTTP {response.status_code}")

            except Exception as e:
                self.logger.warning(f"Error downloading track {track_id}: {e}")
                continue

        self.logger.info(f"Successfully downloaded {len(downloaded_tracks)}/{len(track_ids)} tracks")
        return downloaded_tracks

    def _resample_audio(self, audio_file: Path) -> None:
        """Resample audio file to target sample rate."""
        temp_file = audio_file.with_suffix('.temp.mp3')
        try:
            subprocess.run([
                'ffmpeg',
                '-i', str(audio_file),
                '-ar', str(self.sample_rate),
                '-ac', '1',
                '-y',
                str(temp_file)
            ], check=True, capture_output=True)

            # Replace original with resampled
            temp_file.replace(audio_file)
        except Exception as e:
            self.logger.warning(f"Error resampling {audio_file}: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def _process_metadata(self, audio_metadata: list) -> list:
        """Process and merge all metadata sources."""
        track_ids = {meta['track_id'] for meta in audio_metadata}

        # Load tags from metadata files
        track_tags = {}
        for track_id in track_ids:
            track_tags[track_id] = {
                'genres': [],
                'instruments': [],
                'moods': [],
            }

        # Load genre tags
        genre_files = list(self.metadata_dir.glob("*genre*.tsv"))
        self.logger.debug(f"Found genre files: {genre_files}")
        for genre_file in genre_files:
            try:
                with open(genre_file, 'r') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    for row in reader:
                        track_id = row.get('TRACK_ID') or row.get('track_id')
                        if track_id in track_tags:
                            tags = row.get('TAGS', '').split(',')
                            track_tags[track_id]['genres'].extend(tags)
            except Exception as e:
                self.logger.warning(f"Error processing {genre_file}: {e}")

        # Load instrument tags
        instrument_files = list(self.metadata_dir.glob("*instrument*.tsv"))
        self.logger.debug(f"Found instrument files: {instrument_files}")
        for instrument_file in instrument_files:
            try:
                with open(instrument_file, 'r') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    for row in reader:
                        track_id = row.get('TRACK_ID') or row.get('track_id')
                        if track_id in track_tags:
                            tags = row.get('TAGS', '').split(',')
                            track_tags[track_id]['instruments'].extend(tags)
            except Exception as e:
                self.logger.warning(f"Error processing {instrument_file}: {e}")

        # Load mood tags
        mood_files = list(self.metadata_dir.glob("*moodtheme*.tsv"))
        self.logger.debug(f"Found mood files: {mood_files}")
        for mood_file in mood_files:
            try:
                with open(mood_file, 'r') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    for row in reader:
                        track_id = row.get('TRACK_ID') or row.get('track_id')
                        if track_id in track_tags:
                            tags = row.get('TAGS', '').split(',')
                            track_tags[track_id]['moods'].extend(tags)
            except Exception as e:
                self.logger.warning(f"Error processing {mood_file}: {e}")

        # Create final metadata list
        final_metadata = []
        for track_id in track_ids:
            tags = track_tags.get(track_id, {})
            final_metadata.append({
                'track_id': track_id,
                'genres': list(set(tags.get('genres', []))),
                'instruments': list(set(tags.get('instruments', []))),
                'moods': list(set(tags.get('moods', []))),
            })

        # Save processed metadata
        with open(self.metadata_dir / "processed_metadata.json", 'w') as f:
            json.dump(final_metadata, f, indent=2)

        return final_metadata

    def verify(self) -> bool:
        """Verify the downloaded dataset."""
        metadata = self.load_metadata()
        if metadata is None:
            self.logger.error("No metadata found")
            return False

        # Check processed metadata
        processed_file = self.metadata_dir / "processed_metadata.json"
        if not processed_file.exists():
            self.logger.error("Processed metadata not found")
            return False

        # Load and verify audio files
        with open(processed_file, 'r') as f:
            tracks = json.load(f)

        missing_count = 0
        for track in tracks:
            audio_file = self.audio_dir / f"{track['track_id']}.mp3"
            if not audio_file.exists():
                missing_count += 1

        if missing_count > 0:
            self.logger.warning(f"Missing {missing_count}/{len(tracks)} audio files")
            # Accept if we have at least 80% of files
            success_rate = 1.0 - (missing_count / len(tracks))
            return success_rate >= 0.8

        self.logger.info("Dataset verification passed")
        return True


if __name__ == "__main__":
    # Test the downloader
    logging.basicConfig(level=logging.INFO)
    output_dir = Path("data/jamendo")
    downloader = JamendoDownloader(output_dir, max_samples=100)

    if not downloader.is_downloaded():
        metadata = downloader.download()
        print(f"Downloaded {metadata.num_samples} samples")

    if downloader.verify():
        print("Dataset verified successfully")
    else:
        print("Dataset verification failed")
