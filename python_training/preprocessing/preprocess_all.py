"""Preprocess all datasets for all models."""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import sys
from datetime import datetime
import threading
import signal
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.audio_embedding_generator import AudioEmbeddingGenerator


class PauseHandler:
    """Handles pause requests and keyboard input."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize pause handler."""
        self.pause_requested = False
        self.logger = logger or logging.getLogger("PauseHandler")
        self._listener_thread = None
        self._running = False

    def start_listener(self):
        """Start listening for keyboard input in a separate thread."""
        if self._listener_thread is not None and self._listener_thread.is_alive():
            return

        self._running = True
        self._listener_thread = threading.Thread(target=self._listen_for_pause, daemon=True)
        self._listener_thread.start()

        # Also set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)

        self.logger.info("Pause listener started. Press 'p' + Enter to pause, or Ctrl+C to quit.")

    def _listen_for_pause(self):
        """Listen for pause command in a separate thread."""
        import select

        while self._running:
            # Use select to check if input is available (with timeout)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                try:
                    line = sys.stdin.readline().strip().lower()
                    if line == 'p':
                        self.pause_requested = True
                        self.logger.warning("\n" + "="*80)
                        self.logger.warning("PAUSE REQUESTED - Will stop after current task completes")
                        self.logger.warning("="*80 + "\n")
                        break
                except:
                    pass
            time.sleep(0.1)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C signal."""
        self.logger.warning("\n" + "="*80)
        self.logger.warning("INTERRUPT RECEIVED - Will stop after current task completes")
        self.logger.warning("="*80 + "\n")
        self.pause_requested = True

    def stop_listener(self):
        """Stop the listener thread."""
        self._running = False
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=1.0)

    def is_paused(self) -> bool:
        """Check if pause has been requested."""
        return self.pause_requested


class StateManager:
    """Manages preprocessing state for pause/resume functionality."""

    def __init__(self, state_file: Path, logger: Optional[logging.Logger] = None):
        """Initialize state manager."""
        self.state_file = state_file
        self.logger = logger or logging.getLogger("StateManager")
        self.completed_jobs: Set[Tuple[str, str]] = set()
        # Track current job progress at track level
        self.current_dataset: Optional[str] = None
        self.current_model: Optional[str] = None
        self.last_processed_index: int = -1  # Index of last successfully processed track

    def load_state(self) -> bool:
        """
        Load state from file.

        Returns:
            True if state was loaded, False if no state file exists
        """
        if not self.state_file.exists():
            return False

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self.completed_jobs = set(
                (job['dataset'], job['model'])
                for job in state.get('completed_jobs', [])
            )

            # Load current job progress
            self.current_dataset = state.get('current_dataset')
            self.current_model = state.get('current_model')
            self.last_processed_index = state.get('last_processed_index', -1)

            self.logger.info(f"Loaded state from {self.state_file}")
            self.logger.info(f"Found {len(self.completed_jobs)} completed jobs")
            if self.current_dataset and self.current_model:
                self.logger.info(f"Resuming {self.current_dataset}/{self.current_model} from track {self.last_processed_index + 1}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False

    def save_state(self):
        """Save current state to file."""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'completed_jobs': [
                    {'dataset': dataset, 'model': model}
                    for dataset, model in sorted(self.completed_jobs)
                ],
                'current_dataset': self.current_dataset,
                'current_model': self.current_model,
                'last_processed_index': self.last_processed_index,
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            self.logger.debug(f"State saved to {self.state_file}")

        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def start_job(self, dataset: str, model: str):
        """Mark the start of a new dataset/model job."""
        self.current_dataset = dataset
        self.current_model = model
        self.last_processed_index = -1
        self.save_state()

    def update_progress(self, track_index: int):
        """Update progress for the current job."""
        self.last_processed_index = track_index
        self.save_state()

    def mark_completed(self, dataset: str, model: str):
        """Mark a dataset/model combination as completed."""
        self.completed_jobs.add((dataset, model))
        # Clear current job tracking
        self.current_dataset = None
        self.current_model = None
        self.last_processed_index = -1
        self.save_state()

    def is_completed(self, dataset: str, model: str) -> bool:
        """Check if a dataset/model combination is already completed."""
        return (dataset, model) in self.completed_jobs

    def should_resume(self, dataset: str, model: str) -> Tuple[bool, int]:
        """
        Check if we should resume processing for this dataset/model.

        Returns:
            Tuple of (should_resume, start_index)
        """
        if self.current_dataset == dataset and self.current_model == model:
            return True, self.last_processed_index + 1
        return False, 0

    def clear_state(self):
        """Clear the state file."""
        if self.state_file.exists():
            self.state_file.unlink()
            self.logger.info(f"Cleared state file: {self.state_file}")
        self.completed_jobs.clear()
        self.current_dataset = None
        self.current_model = None
        self.last_processed_index = -1


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
        state_manager: Optional[StateManager] = None,
        pause_handler: Optional[PauseHandler] = None,
    ):
        """
        Initialize preprocessing manager.

        Args:
            datasets_dir: Directory containing all datasets
            output_dir: Directory to save preprocessed embeddings
            device: Device to use for inference
            batch_size: Batch size for processing
            logger: Logger instance
            state_manager: State manager for pause/resume
            pause_handler: Pause handler for interrupt handling
        """
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger("PreprocessingManager")
        self.state_manager = state_manager
        self.pause_handler = pause_handler

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

        # Check if already processed (and not resuming this specific job)
        if self.state_manager:
            should_resume, _ = self.state_manager.should_resume(dataset_name, model_name)
            if not should_resume and output_file.exists() and not force:
                self.logger.info(f"Embeddings already exist for {dataset_name}/{model_name}: {output_file}")
                self.logger.info(f"Use --force to re-process")
                return True, {'skipped': True}
        elif output_file.exists() and not force:
            self.logger.info(f"Embeddings already exist for {dataset_name}/{model_name}: {output_file}")
            self.logger.info(f"Use --force to re-process")
            return True, {'skipped': True}

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Processing: {dataset_name} with {model_name}")
        self.logger.info(f"Output: {output_file}")
        self.logger.info(f"{'='*80}\n")

        try:
            # Mark job as started in state manager (only if not resuming this job)
            if self.state_manager:
                should_resume, resume_idx = self.state_manager.should_resume(dataset_name, model_name)
                if should_resume:
                    self.logger.info(f"Resuming {dataset_name}/{model_name} from track {resume_idx}")
                else:
                    self.logger.info(f"Starting new job: {dataset_name}/{model_name}")
                    self.state_manager.start_job(dataset_name, model_name)

            # Create embedding generator
            generator = AudioEmbeddingGenerator(
                model_name=model_name,
                device=self.device,
                batch_size=self.batch_size,
                logger=self.logger,
                pause_handler=self.pause_handler,
                state_manager=self.state_manager,
            )

            # Generate embeddings
            stats = generator.generate_embeddings_for_dataset(
                dataset_dir=dataset_dir,
                output_file=output_file,
                dataset_name=dataset_name,
                max_samples=max_samples,
            )

            # Check if paused
            if stats.get('paused'):
                return False, stats

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
            # Check if pause was requested
            if self.pause_handler and self.pause_handler.is_paused():
                self.logger.warning(f"Pausing before {dataset_name}/{model_name}")
                break

            # Check if already completed (from previous run)
            if self.state_manager and self.state_manager.is_completed(dataset_name, model_name):
                self.logger.info(f"Skipping {dataset_name}/{model_name} - already completed in previous run")
                results[model_name] = (True, {'resumed_skip': True})
                continue

            results[model_name] = self.preprocess_dataset_model(
                dataset_name=dataset_name,
                model_name=model_name,
                max_samples=max_samples,
                force=force,
            )

            # Mark as completed if successful (and not paused)
            success, stats = results[model_name]
            if success and self.state_manager and not stats.get('skipped'):
                self.state_manager.mark_completed(dataset_name, model_name)
            elif stats.get('paused'):
                # Job was paused, state is already saved with progress
                self.logger.info(f"Progress saved for {dataset_name}/{model_name}")
                break

        # Summary for this dataset
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"DATASET SUMMARY: {dataset_name}")
        self.logger.info(f"{'='*80}")
        for model_name, (success, stats) in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            if stats.get('skipped') or stats.get('resumed_skip'):
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
            # Check if pause was requested
            if self.pause_handler and self.pause_handler.is_paused():
                self.logger.warning(f"Pausing before dataset {dataset_name}")
                break

            all_results[dataset_name] = self.preprocess_dataset_all_models(
                dataset_name=dataset_name,
                models=models,
                max_samples=max_samples,
                force=force,
            )

            # Check again after processing dataset (in case pause happened during processing)
            if self.pause_handler and self.pause_handler.is_paused():
                self.logger.warning(f"Pause detected after processing {dataset_name}")
                break

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
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous paused session'
    )
    parser.add_argument(
        '--clear-state',
        action='store_true',
        help='Clear saved state and start fresh'
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

    # Create state file path
    state_file = output_dir / ".preprocessing_state.json"
    logging.info(f"State file: {state_file}")

    # Create state manager
    state_manager = StateManager(state_file=state_file)

    # Handle clear state
    if args.clear_state:
        state_manager.clear_state()
        logging.info("State cleared. Starting fresh.")

    # Load state if resuming
    if args.resume:
        if state_manager.load_state():
            logging.info("Resuming from previous session...")
        else:
            logging.warning("No previous state found. Starting fresh.")

    # Create pause handler
    pause_handler = PauseHandler()
    pause_handler.start_listener()

    # Create preprocessing manager
    manager = PreprocessingManager(
        datasets_dir=datasets_dir,
        output_dir=output_dir,
        device=args.device,
        batch_size=args.batch_size,
        state_manager=state_manager,
        pause_handler=pause_handler,
    )

    try:
        # Run preprocessing
        manager.preprocess_all(
            datasets=args.datasets,
            models=args.models,
            max_samples=args.max_samples,
            force=args.force,
        )

        # If completed successfully, clear the state
        if not pause_handler.is_paused():
            state_manager.clear_state()
            logging.info("All preprocessing completed successfully!")
        else:
            logging.warning("\n" + "="*80)
            logging.warning("PREPROCESSING PAUSED")
            logging.warning("="*80)
            logging.warning(f"State saved to: {state_file}")
            logging.warning("To resume, run with --resume flag")
            logging.warning("="*80 + "\n")

    finally:
        # Stop the pause handler
        pause_handler.stop_listener()


if __name__ == "__main__":
    main()
