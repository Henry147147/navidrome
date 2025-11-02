#!/usr/bin/env python3
"""
download_datasets.py - Unified Music Dataset Downloader

Downloads and organizes multiple music datasets for training.

Supported datasets:
- Song Describer: Human captions for ~2 min clips
- JamendoMaxCaps: Synthetic captions for full tracks
- MTG-Jamendo: Full tracks with tags
- FMA: Free Music Archive with rich metadata
- DALI: Time-aligned lyrics
- Clotho: General audio captions (auxiliary)

Usage:
    # Download all datasets
    python download_datasets.py --all

    # Download specific datasets
    python download_datasets.py --datasets song_describer jamendo_max_caps

    # Download with FMA size specification
    python download_datasets.py --datasets fma --fma-size small

    # Resume interrupted download
    python download_datasets.py --continue

    # Dry run (see what would be downloaded)
    python download_datasets.py --all --dry-run
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict

from dataset_downloaders import (
    SongDescriberDownloader,
    JamendoMaxCapsDownloader,
    MTGJamendoDownloader,
    FMADownloader,
    DALIDownloader,
    ClothoDownloader
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_datasets.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Dataset registry
DATASETS = {
    'song_describer': {
        'class': SongDescriberDownloader,
        'description': 'Human captions for ~2min clips (~1,100 songs, ~3GB)',
        'priority': 'HIGH',
        'ready': True
    },
    'jamendo_max_caps': {
        'class': JamendoMaxCapsDownloader,
        'description': 'Synthetic captions for full tracks (~28K songs, ~600GB)',
        'priority': 'HIGH',
        'ready': True
    },
    'mtg_jamendo': {
        'class': MTGJamendoDownloader,
        'description': 'Full tracks with tags (~55K songs, ~1.2TB)',
        'priority': 'MEDIUM',
        'ready': False  # Needs implementation
    },
    'fma': {
        'class': FMADownloader,
        'description': 'Free Music Archive (8K-106K songs, 7GB-900GB)',
        'priority': 'MEDIUM',
        'ready': True
    },
    'dali': {
        'class': DALIDownloader,
        'description': 'Time-aligned lyrics (~5K songs, ~50GB)',
        'priority': 'LOW',
        'ready': False  # Needs implementation
    },
    'clotho': {
        'class': ClothoDownloader,
        'description': 'General audio captions (~6K clips, ~1GB)',
        'priority': 'LOW',
        'ready': False  # Needs implementation
    }
}


def list_datasets():
    """Print available datasets"""
    logger.info("\n" + "="*80)
    logger.info("AVAILABLE DATASETS")
    logger.info("="*80)

    for name, info in DATASETS.items():
        status = "✓ READY" if info['ready'] else "⚠ PENDING"
        logger.info(f"\n{name.upper()} ({status})")
        logger.info(f"  Priority: {info['priority']}")
        logger.info(f"  {info['description']}")

    logger.info("\n" + "="*80)


def download_dataset(
    dataset_name: str,
    output_base_dir: str,
    resume: bool = True,
    dry_run: bool = False,
    **kwargs
) -> bool:
    """
    Download a single dataset.

    Args:
        dataset_name: Name of dataset to download
        output_base_dir: Base output directory
        resume: Whether to resume interrupted downloads
        dry_run: If True, don't actually download
        **kwargs: Additional arguments for specific downloaders

    Returns:
        True if successful
    """
    if dataset_name not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.error(f"Available: {list(DATASETS.keys())}")
        return False

    dataset_info = DATASETS[dataset_name]

    if not dataset_info['ready']:
        logger.warning(f"{dataset_name} downloader not yet implemented")
        logger.warning("Please download manually - check dataset README")
        return False

    # Create dataset-specific output directory
    output_dir = Path(output_base_dir) / "raw" / dataset_name

    logger.info(f"\n{'='*80}")
    logger.info(f"DOWNLOADING: {dataset_name.upper()}")
    logger.info(f"{'='*80}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Resume: {resume}")
    logger.info(f"Dry run: {dry_run}")

    try:
        # Initialize downloader
        downloader_class = dataset_info['class']

        # Handle special cases (e.g., FMA with size parameter)
        if dataset_name == 'fma':
            fma_size = kwargs.get('fma_size', 'small')
            downloader = downloader_class(
                output_dir=str(output_dir),
                size=fma_size,
                resume=resume,
                dry_run=dry_run
            )
        else:
            downloader = downloader_class(
                output_dir=str(output_dir),
                resume=resume,
                dry_run=dry_run
            )

        # Download
        success = downloader.download()

        if success:
            logger.info(f"✓ {dataset_name} download completed successfully")
        else:
            logger.error(f"✗ {dataset_name} download failed")

        return success

    except Exception as e:
        logger.error(f"Error downloading {dataset_name}: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download music datasets for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=list(DATASETS.keys()),
        help='Datasets to download'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all ready datasets'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets and exit'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Base output directory (default: data/)'
    )

    parser.add_argument(
        '--fma-size',
        type=str,
        choices=['small', 'medium', 'large', 'full'],
        default='small',
        help='FMA dataset size (default: small)'
    )

    parser.add_argument(
        '--continue',
        dest='resume',
        action='store_true',
        default=True,
        help='Resume interrupted downloads (default: True)'
    )

    parser.add_argument(
        '--no-resume',
        dest='resume',
        action='store_false',
        help='Start downloads from scratch'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without actually downloading'
    )

    args = parser.parse_args()

    # List datasets if requested
    if args.list:
        list_datasets()
        return 0

    # Determine which datasets to download
    datasets_to_download = []

    if args.all:
        # Download all ready datasets
        datasets_to_download = [
            name for name, info in DATASETS.items()
            if info['ready']
        ]
        logger.info(f"Downloading all ready datasets: {datasets_to_download}")
    elif args.datasets:
        datasets_to_download = args.datasets
    else:
        logger.error("No datasets specified. Use --datasets or --all")
        list_datasets()
        return 1

    # Download each dataset
    results = {}
    for dataset_name in datasets_to_download:
        success = download_dataset(
            dataset_name=dataset_name,
            output_base_dir=args.output_dir,
            resume=args.resume,
            dry_run=args.dry_run,
            fma_size=args.fma_size
        )
        results[dataset_name] = success

    # Print final summary
    logger.info(f"\n{'='*80}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*80}")

    for dataset_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {dataset_name}: {status}")

    logger.info(f"{'='*80}\n")

    # Return appropriate exit code
    all_success = all(results.values())
    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())
