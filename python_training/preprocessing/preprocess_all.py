"""Preprocess all datasets for all models."""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.audio_embedding_generator import AudioEmbeddingGenerator


class PreprocessingManager:
    """Manager for preprocessing all datasets with all models."""

    # Available models
    MODELS = ['muq', 'mert', 'music2latent']

    # Dataset names (must match directory names in data/datasets)
    DATASETS = ['musiccaps', 'fma_large', 'fma_full', 'jamendo']

    def __init__(
        self,
        datasets_dir: Path,
        output_dir: Path,
        device: str = "cuda",
        batch_size: int = 8,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize preprocessing manager.

        Args:
            datasets_dir: Directory containing all datasets
            output_dir: Directory to save preprocessed embeddings
            device: Device to use for inference
            batch_size: Batch size for processing
            logger: Logger instance
        """
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger("PreprocessingManager")

    def discover_datasets(self) -> List[str]:
        """
        Discover available datasets in the datasets directory.

        Returns:
            List of dataset names that exist
        """
        available_datasets = []

        for dataset_name in self.DATASETS:
            dataset_path = self.datasets_dir / dataset_name
            if dataset_path.exists() and dataset_path.is_dir():
                # Check if it has an audio subdirectory
                audio_dir = dataset_path / "audio"
                if audio_dir.exists():
                    available_datasets.append(dataset_name)
                    self.logger.info(f"Found dataset: {dataset_name} at {dataset_path}")
                else:
                    self.logger.warning(f"Dataset {dataset_name} exists but has no audio/ directory")
            else:
                self.logger.debug(f"Dataset {dataset_name} not found at {dataset_path}")

        return available_datasets

    def preprocess_dataset_model(
        self,
        dataset_name: str,
        model_name: str,
        max_samples: Optional[int] = None,
        force: bool = False,
    ) -> Tuple[bool, Dict]:
        """
        Preprocess a single dataset with a single model.

        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            max_samples: Maximum number of samples to process
            force: Force re-processing even if already exists

        Returns:
            Tuple of (success, stats)
        """
        dataset_dir = self.datasets_dir / dataset_name
        output_file = self.output_dir / dataset_name / f"{model_name}_embeddings.h5"

        # Check if already processed
        if output_file.exists() and not force:
            self.logger.info(f"Embeddings already exist for {dataset_name}/{model_name}: {output_file}")
            self.logger.info(f"Use --force to re-process")
            return True, {'skipped': True}

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Processing: {dataset_name} with {model_name}")
        self.logger.info(f"Output: {output_file}")
        self.logger.info(f"{'='*80}\n")

        try:
            # Create embedding generator
            generator = AudioEmbeddingGenerator(
                model_name=model_name,
                device=self.device,
                batch_size=self.batch_size,
                logger=self.logger,
            )

            # Generate embeddings
            stats = generator.generate_embeddings_for_dataset(
                dataset_dir=dataset_dir,
                output_file=output_file,
                max_samples=max_samples,
            )

            self.logger.info(f"✓ Successfully processed {dataset_name}/{model_name}")
            self.logger.info(f"  Successful: {stats['successful']}/{stats['total_files']}")

            return True, stats

        except Exception as e:
            self.logger.error(f"✗ Failed to process {dataset_name}/{model_name}: {e}", exc_info=True)
            return False, {'error': str(e)}

    def preprocess_dataset_all_models(
        self,
        dataset_name: str,
        models: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Tuple[bool, Dict]]:
        """
        Preprocess a single dataset with all models.

        Args:
            dataset_name: Name of the dataset
            models: List of models to use (default: all)
            max_samples: Maximum number of samples to process
            force: Force re-processing even if already exists

        Returns:
            Dictionary mapping model names to (success, stats) tuples
        """
        models = models or self.MODELS
        results = {}

        self.logger.info(f"\n{'#'*80}")
        self.logger.info(f"PREPROCESSING DATASET: {dataset_name}")
        self.logger.info(f"Models: {', '.join(models)}")
        self.logger.info(f"{'#'*80}\n")

        for model_name in models:
            results[model_name] = self.preprocess_dataset_model(
                dataset_name=dataset_name,
                model_name=model_name,
                max_samples=max_samples,
                force=force,
            )

        # Summary for this dataset
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"DATASET SUMMARY: {dataset_name}")
        self.logger.info(f"{'='*80}")
        for model_name, (success, stats) in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            if stats.get('skipped'):
                status = "⊘ SKIPPED"
            self.logger.info(f"  {model_name}: {status}")
        self.logger.info(f"{'='*80}\n")

        return results

    def preprocess_all(
        self,
        datasets: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Dict[str, Tuple[bool, Dict]]]:
        """
        Preprocess all datasets with all models.

        Args:
            datasets: List of datasets to process (default: all discovered)
            models: List of models to use (default: all)
            max_samples: Maximum number of samples per dataset
            force: Force re-processing even if already exists

        Returns:
            Nested dictionary: dataset_name -> model_name -> (success, stats)
        """
        # Discover available datasets if not specified
        if datasets is None:
            datasets = self.discover_datasets()
            if not datasets:
                self.logger.error(f"No datasets found in {self.datasets_dir}")
                return {}

        models = models or self.MODELS

        self.logger.info("\n" + "#"*80)
        self.logger.info("PREPROCESSING ALL DATASETS")
        self.logger.info(f"Datasets: {', '.join(datasets)}")
        self.logger.info(f"Models: {', '.join(models)}")
        self.logger.info("#"*80 + "\n")

        all_results = {}
        start_time = datetime.now()

        for dataset_name in datasets:
            all_results[dataset_name] = self.preprocess_dataset_all_models(
                dataset_name=dataset_name,
                models=models,
                max_samples=max_samples,
                force=force,
            )

        # Final summary
        self.logger.info(f"\n{'#'*80}")
        self.logger.info(f"FINAL PREPROCESSING SUMMARY")
        self.logger.info(f"{'#'*80}")

        total_jobs = 0
        successful_jobs = 0
        skipped_jobs = 0
        failed_jobs = 0

        for dataset_name, model_results in all_results.items():
            self.logger.info(f"\n{dataset_name}:")
            for model_name, (success, stats) in model_results.items():
                total_jobs += 1
                if stats.get('skipped'):
                    status = "⊘ SKIPPED"
                    skipped_jobs += 1
                elif success:
                    status = "✓ SUCCESS"
                    successful_jobs += 1
                else:
                    status = "✗ FAILED"
                    failed_jobs += 1
                self.logger.info(f"  {model_name}: {status}")

        elapsed_time = datetime.now() - start_time

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Total jobs: {total_jobs}")
        self.logger.info(f"Successful: {successful_jobs}")
        self.logger.info(f"Skipped: {skipped_jobs}")
        self.logger.info(f"Failed: {failed_jobs}")
        self.logger.info(f"Time elapsed: {elapsed_time}")
        if total_jobs > 0 and (total_jobs - skipped_jobs) > 0:
            self.logger.info(f"Success rate: {successful_jobs/(total_jobs-skipped_jobs)*100:.1f}%")
        self.logger.info(f"{'#'*80}\n")

        # Save summary to file
        self._save_summary(all_results, elapsed_time)

        return all_results

    def _save_summary(self, results: Dict, elapsed_time) -> None:
        """Save preprocessing summary to JSON file."""
        summary_file = self.output_dir / "preprocessing_summary.json"

        summary = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_time_seconds': elapsed_time.total_seconds(),
            'datasets_dir': str(self.datasets_dir),
            'output_dir': str(self.output_dir),
            'device': self.device,
            'batch_size': self.batch_size,
            'results': {}
        }

        for dataset_name, model_results in results.items():
            summary['results'][dataset_name] = {}
            for model_name, (success, stats) in model_results.items():
                summary['results'][dataset_name][model_name] = {
                    'success': success,
                    'stats': stats
                }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Summary saved to: {summary_file}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Preprocess all datasets for all models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess all datasets with all models
  python preprocess_all.py

  # Preprocess specific dataset with all models
  python preprocess_all.py --dataset musiccaps

  # Preprocess all datasets with specific model
  python preprocess_all.py --model muq

  # Preprocess specific dataset with specific model
  python preprocess_all.py --dataset musiccaps --model muq

  # Limit samples for testing
  python preprocess_all.py --max-samples 100

  # Force re-processing
  python preprocess_all.py --force
"""
    )

    parser.add_argument(
        '--datasets-dir',
        type=str,
        default='../data/datasets',
        help='Directory containing all datasets (default: ../data/datasets)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/preprocessed',
        help='Output directory for preprocessed embeddings (default: ../data/preprocessed)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        action='append',
        dest='datasets',
        help='Specific dataset(s) to preprocess (can be specified multiple times)'
    )
    parser.add_argument(
        '--model',
        type=str,
        action='append',
        dest='models',
        choices=['muq', 'mert', 'music2latent'],
        help='Specific model(s) to use (can be specified multiple times)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use for inference (default: cuda)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for processing (default: 8)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples per dataset (for testing)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-processing even if embeddings already exist'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Convert relative paths to absolute
    datasets_dir = Path(args.datasets_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Create preprocessing manager
    manager = PreprocessingManager(
        datasets_dir=datasets_dir,
        output_dir=output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Run preprocessing
    manager.preprocess_all(
        datasets=args.datasets,
        models=args.models,
        max_samples=args.max_samples,
        force=args.force,
    )


if __name__ == "__main__":
    main()
