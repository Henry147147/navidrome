"""Downloader for Google's MusicCaps dataset."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging
import json
import subprocess
import time

from datasets import load_dataset
from tqdm import tqdm
import yt_dlp

from ..base_downloader import BaseDownloader, DatasetMetadata


class MusicCapsDownloader(BaseDownloader):
    """
    Downloader for Google's MusicCaps dataset.

    MusicCaps contains ~5,500 music clips (10 seconds each) from YouTube
    with detailed text captions written by musicians.
    """

    DATASET_NAME = "google/MusicCaps"
    CLIP_DURATION = 10  # seconds

    def __init__(
        self,
        output_dir: Path,
        max_samples: Optional[int] = None,
        num_workers: int = 4,
        audio_format: str = "wav",
        sample_rate: int = 44100,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize MusicCaps downloader.

        Args:
            output_dir: Directory where dataset will be saved
            max_samples: Maximum number of samples to download (None for all)
            num_workers: Number of parallel download workers
            audio_format: Audio format (wav, mp3, flac)
            sample_rate: Target sample rate in Hz
            logger: Logger instance
        """
        super().__init__(output_dir, logger)
        self.max_samples = max_samples
        self.num_workers = num_workers
        self.audio_format = audio_format
        self.sample_rate = sample_rate

    def download(self) -> DatasetMetadata:
        """Download MusicCaps dataset from HuggingFace and audio from YouTube."""
        self.logger.info("Loading MusicCaps dataset from HuggingFace...")

        # Load the dataset metadata
        dataset = load_dataset("google/MusicCaps", split="train")

        if self.max_samples:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))

        self.logger.info(f"Found {len(dataset)} samples to download")

        # Save text metadata
        metadata_list = []
        for idx, sample in enumerate(dataset):
            metadata_list.append({
                'id': idx,
                'ytid': sample['ytid'],
                'start_s': sample['start_s'],
                'end_s': sample['end_s'],
                'caption': sample['caption'],
                'aspect_list': sample['aspect_list'],
                'audioset_positive_labels': sample['audioset_positive_labels'],
                'is_balanced_subset': sample['is_balanced_subset'],
                'is_audioset_eval': sample['is_audioset_eval'],
            })

        # Save metadata JSON
        with open(self.metadata_dir / "samples.json", 'w') as f:
            json.dump(metadata_list, f, indent=2)

        # Download audio from YouTube
        self.logger.info("Downloading audio from YouTube...")
        successful_downloads = self._download_audio(metadata_list)

        # Calculate total size
        total_size = sum(
            (self.audio_dir / f"{meta['id']:06d}.{self.audio_format}").stat().st_size
            for meta in metadata_list
            if (self.audio_dir / f"{meta['id']:06d}.{self.audio_format}").exists()
        )

        # Create dataset metadata
        metadata = DatasetMetadata(
            name="MusicCaps",
            version="1.0",
            num_samples=successful_downloads,
            total_size_bytes=total_size,
            download_date=datetime.now().isoformat(),
            source_url="https://huggingface.co/datasets/google/MusicCaps",
            format=self.audio_format,
            audio_duration_total_seconds=successful_downloads * self.CLIP_DURATION,
            additional_info={
                'sample_rate': self.sample_rate,
                'clip_duration': self.CLIP_DURATION,
                'failed_downloads': len(metadata_list) - successful_downloads,
            }
        )

        self.save_metadata(metadata)
        return metadata

    def _download_audio(self, metadata_list: list) -> int:
        """
        Download audio clips from YouTube.

        Args:
            metadata_list: List of metadata dictionaries

        Returns:
            Number of successful downloads
        """
        successful = 0
        failed_ids = []

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(self.audio_dir / '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': self.audio_format,
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'extract_audio': True,
            'audio-format': self.audio_format,
            'audio-quality': 0,  # Best quality
            'postprocessor_args': [
                '-ar', str(self.sample_rate),  # Sample rate
                '-ac', '1',  # Mono
            ],
        }

        for meta in tqdm(metadata_list, desc="Downloading audio"):
            output_file = self.audio_dir / f"{meta['id']:06d}.{self.audio_format}"

            # Skip if already downloaded
            if output_file.exists():
                successful += 1
                continue

            try:
                # Build YouTube URL
                ytid = meta['ytid']
                url = f"https://www.youtube.com/watch?v={ytid}"

                # Download with yt-dlp
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Download full video
                    info = ydl.extract_info(url, download=True)
                    temp_file = self.audio_dir / f"{ytid}.{self.audio_format}"

                    # Extract the specific time segment using ffmpeg
                    if temp_file.exists():
                        start_time = meta['start_s']
                        duration = self.CLIP_DURATION

                        subprocess.run([
                            'ffmpeg',
                            '-i', str(temp_file),
                            '-ss', str(start_time),
                            '-t', str(duration),
                            '-ar', str(self.sample_rate),
                            '-ac', '1',
                            '-y',  # Overwrite
                            str(output_file)
                        ], check=True, capture_output=True)

                        # Remove temp file
                        temp_file.unlink()
                        successful += 1
                    else:
                        self.logger.warning(f"Could not find temp file for {ytid}")
                        failed_ids.append(meta['id'])

            except Exception as e:
                self.logger.warning(f"Failed to download {meta['ytid']}: {str(e)}")
                failed_ids.append(meta['id'])
                continue

            # Rate limiting to avoid YouTube throttling
            time.sleep(0.5)

        if failed_ids:
            with open(self.metadata_dir / "failed_downloads.json", 'w') as f:
                json.dump(failed_ids, f, indent=2)

        self.logger.info(f"Successfully downloaded {successful}/{len(metadata_list)} samples")
        return successful

    def verify(self) -> bool:
        """Verify the downloaded dataset."""
        metadata = self.load_metadata()
        if metadata is None:
            self.logger.error("No metadata found")
            return False

        # Check metadata file
        samples_file = self.metadata_dir / "samples.json"
        if not samples_file.exists():
            self.logger.error("samples.json not found")
            return False

        # Load samples
        with open(samples_file, 'r') as f:
            samples = json.load(f)

        # Verify audio files exist
        missing_files = []
        for sample in samples:
            audio_file = self.audio_dir / f"{sample['id']:06d}.{self.audio_format}"
            if not audio_file.exists():
                missing_files.append(sample['id'])

        if missing_files:
            self.logger.warning(f"Missing {len(missing_files)} audio files")
            # This is expected for YouTube downloads
            # Consider it valid if we have at least 90% of files
            success_rate = 1.0 - (len(missing_files) / len(samples))
            return success_rate >= 0.9

        self.logger.info("Dataset verification passed")
        return True


if __name__ == "__main__":
    # Test the downloader
    logging.basicConfig(level=logging.INFO)
    output_dir = Path("data/musiccaps")
    downloader = MusicCapsDownloader(output_dir, max_samples=10)

    if not downloader.is_downloaded():
        metadata = downloader.download()
        print(f"Downloaded {metadata.num_samples} samples")

    if downloader.verify():
        print("Dataset verified successfully")
    else:
        print("Dataset verification failed")
