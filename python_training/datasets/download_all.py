"""Master script to download all datasets for music-text embedding training."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.downloaders.musiccaps_downloader import MusicCapsDownloader
from datasets.downloaders.fma_downloader import FMADownloader
from datasets.downloaders.jamendo_downloader import JamendoDownloader
from datasets.downloaders.musicbench_downloader import MusicBenchDownloader


class DatasetManager:
    """Manager for downloading and organizing multiple datasets."""

    def __init__(self, base_output_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize dataset manager.

        Args:
            base_output_dir: Base directory for all datasets
            logger: Logger instance
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("DatasetManager")

        # Dataset configurations
        self.datasets = {
            'musiccaps': {
                'class': MusicCapsDownloader,
                'output_dir': self.base_output_dir / 'musiccaps',
                'kwargs': {
                    'max_samples': None,  # Download all
                    'audio_format': 'wav',
                    'sample_rate': 44100,
                },
                'description': 'MusicCaps: ~5.5k clips with detailed captions (~50GB)',
                'phase': 1,  # Phase 1: Large-scale short clips
            },
            # NOTE: MusicBench disabled - HuggingFace dataset doesn't include audio files
            # 'musicbench': {
            #     'class': MusicBenchDownloader,
            #     'output_dir': self.base_output_dir / 'musicbench',
            #     'kwargs': {
            #         'max_samples': None,
            #         'audio_format': 'wav',
            #         'sample_rate': 44100,
            #     },
            #     'description': 'MusicBench: ~50k music samples with prompts (~50GB)',
            #     'phase': 1,
            # },
            'fma_large': {
                'class': FMADownloader,
                'output_dir': self.base_output_dir / 'fma_large',
                'kwargs': {
                    'dataset_size': 'large',
                    'extract_clips': False,
                },
                'description': 'FMA Large: 106k tracks 30s clips (~93GB)',
                'phase': 1,
            },
            'fma_full': {
                'class': FMADownloader,
                'output_dir': self.base_output_dir / 'fma_full',
                'kwargs': {
                    'dataset_size': 'large',
                    'extract_clips': True,
                    'clips_per_track': 3,
                },
                'description': 'FMA Full: 106k full tracks with extracted clips (~900GB)',
                'phase': 2,  # Phase 2: Full songs
            },
            'jamendo': {
                'class': JamendoDownloader,
                'output_dir': self.base_output_dir / 'jamendo',
                'kwargs': {
                    'max_samples': None,
                    'num_tar_files': 10,  # Download 10 tar files (~50-60GB, ~5.5k tracks)
                },
                'description': 'MTG-Jamendo: ~5.5k tracks with tags (~60GB)',
                'phase': 1,
            },
        }

    def download_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """
        Download a specific dataset.

        Args:
            dataset_name: Name of the dataset to download
            force: Force re-download even if already exists

        Returns:
            True if successful, False otherwise
        """
        if dataset_name not in self.datasets:
            self.logger.error(f"Unknown dataset: {dataset_name}")
            self.logger.info(f"Available datasets: {list(self.datasets.keys())}")
            return False

        config = self.datasets[dataset_name]
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Downloading: {dataset_name}")
        self.logger.info(f"Description: {config['description']}")
        self.logger.info(f"Output: {config['output_dir']}")
        self.logger.info(f"{'='*80}\n")

        try:
            # Create downloader
            downloader = config['class'](
                output_dir=config['output_dir'],
                logger=self.logger,
                **config['kwargs']
            )

            # Check if already downloaded
            if not force and downloader.is_downloaded():
                self.logger.info(f"{dataset_name} already downloaded. Use --force to re-download.")
                return True

            # Download
            metadata = downloader.download()
            self.logger.info(f"\nDownload complete!")
            self.logger.info(f"  Samples: {metadata.num_samples}")
            self.logger.info(f"  Size: {metadata.total_size_bytes / (1024**3):.2f} GB")
            self.logger.info(f"  Duration: {metadata.audio_duration_total_seconds / 3600:.2f} hours")

            # Verify
            if downloader.verify():
                self.logger.info(f"✓ {dataset_name} verified successfully\n")
                return True
            else:
                self.logger.error(f"✗ {dataset_name} verification failed\n")
                return False

        except Exception as e:
            self.logger.error(f"Error downloading {dataset_name}: {e}", exc_info=True)
            return False

    def download_phase(self, phase: int, force: bool = False) -> Dict[str, bool]:
        """
        Download all datasets for a specific phase.

        Args:
            phase: Phase number (1 or 2)
            force: Force re-download even if already exists

        Returns:
            Dictionary mapping dataset names to success status
        """
        phase_datasets = [
            name for name, config in self.datasets.items()
            if config['phase'] == phase
        ]

        self.logger.info(f"\n{'#'*80}")
        self.logger.info(f"PHASE {phase} DOWNLOAD")
        self.logger.info(f"Datasets: {', '.join(phase_datasets)}")
        self.logger.info(f"{'#'*80}\n")

        results = {}
        for dataset_name in phase_datasets:
            results[dataset_name] = self.download_dataset(dataset_name, force=force)

        # Summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"PHASE {phase} SUMMARY")
        self.logger.info(f"{'='*80}")
        for dataset_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            self.logger.info(f"  {dataset_name}: {status}")
        self.logger.info(f"{'='*80}\n")

        return results

    def download_all(self, force: bool = False) -> Dict[str, bool]:
        """
        Download all datasets.

        Args:
            force: Force re-download even if already exists

        Returns:
            Dictionary mapping dataset names to success status
        """
        self.logger.info("\n" + "#"*80)
        self.logger.info("DOWNLOADING ALL DATASETS")
        self.logger.info("#"*80 + "\n")

        # Download phase 1 first (short clips)
        phase1_results = self.download_phase(1, force=force)

        # Then phase 2 (full songs)
        phase2_results = self.download_phase(2, force=force)

        # Combine results
        all_results = {**phase1_results, **phase2_results}

        # Final summary
        total = len(all_results)
        successful = sum(1 for success in all_results.values() if success)

        self.logger.info(f"\n{'#'*80}")
        self.logger.info(f"FINAL SUMMARY")
        self.logger.info(f"{'#'*80}")
        self.logger.info(f"Total datasets: {total}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {total - successful}")
        self.logger.info(f"Success rate: {successful/total*100:.1f}%")
        self.logger.info(f"{'#'*80}\n")

        return all_results

    def get_statistics(self) -> Dict:
        """Get statistics for all downloaded datasets."""
        stats = {
            'datasets': {},
            'total_samples': 0,
            'total_size_gb': 0.0,
            'total_duration_hours': 0.0,
        }

        for dataset_name, config in self.datasets.items():
            downloader = config['class'](
                output_dir=config['output_dir'],
                logger=self.logger,
                **config['kwargs']
            )

            metadata = downloader.load_metadata()
            if metadata:
                stats['datasets'][dataset_name] = {
                    'num_samples': metadata.num_samples,
                    'size_gb': metadata.total_size_bytes / (1024**3),
                    'duration_hours': metadata.audio_duration_total_seconds / 3600,
                    'downloaded': True,
                }
                stats['total_samples'] += metadata.num_samples
                stats['total_size_gb'] += metadata.total_size_bytes / (1024**3)
                stats['total_duration_hours'] += metadata.audio_duration_total_seconds / 3600
            else:
                stats['datasets'][dataset_name] = {
                    'downloaded': False,
                }

        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download music-text datasets for embedding training'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/datasets',
        help='Base output directory for all datasets'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Specific dataset to download (e.g., musiccaps, fma_large)'
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2],
        help='Download all datasets for a specific phase (1: short clips, 2: full songs)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all datasets'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if already exists'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics for downloaded datasets'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create manager
    manager = DatasetManager(Path(args.output_dir))

    # Execute requested operation
    if args.stats:
        stats = manager.get_statistics()
        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        for dataset_name, dataset_stats in stats['datasets'].items():
            if dataset_stats.get('downloaded', False):
                print(f"\n{dataset_name}:")
                print(f"  Samples: {dataset_stats['num_samples']:,}")
                print(f"  Size: {dataset_stats['size_gb']:.2f} GB")
                print(f"  Duration: {dataset_stats['duration_hours']:.2f} hours")
            else:
                print(f"\n{dataset_name}: Not downloaded")

        print(f"\n{'='*80}")
        print("TOTAL:")
        print(f"  Samples: {stats['total_samples']:,}")
        print(f"  Size: {stats['total_size_gb']:.2f} GB")
        print(f"  Duration: {stats['total_duration_hours']:.2f} hours")
        print("="*80 + "\n")

    elif args.all:
        manager.download_all(force=args.force)

    elif args.phase:
        manager.download_phase(args.phase, force=args.force)

    elif args.dataset:
        manager.download_dataset(args.dataset, force=args.force)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Download all datasets")
        print("  python download_all.py --all")
        print("\n  # Download Phase 1 datasets only (short clips)")
        print("  python download_all.py --phase 1")
        print("\n  # Download specific dataset")
        print("  python download_all.py --dataset musiccaps")
        print("\n  # Show statistics")
        print("  python download_all.py --stats")


if __name__ == "__main__":
    main()
