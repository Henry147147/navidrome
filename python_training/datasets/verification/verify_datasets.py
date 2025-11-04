"""Tools for verifying downloaded datasets."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import soundfile as sf
import numpy as np
from tqdm import tqdm


class DatasetVerifier:
    """Verify integrity and quality of downloaded datasets."""

    def __init__(self, dataset_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize verifier.

        Args:
            dataset_dir: Directory containing the dataset
            logger: Logger instance
        """
        self.dataset_dir = Path(dataset_dir)
        self.logger = logger or logging.getLogger("DatasetVerifier")
        self.audio_dir = self.dataset_dir / "audio"
        self.metadata_dir = self.dataset_dir / "metadata"

    def verify_audio_files(self) -> Dict[str, any]:
        """
        Verify all audio files in the dataset.

        Returns:
            Dictionary with verification results
        """
        self.logger.info("Verifying audio files...")

        if not self.audio_dir.exists():
            return {
                'error': 'Audio directory not found',
                'valid': False,
            }

        # Get all audio files
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        audio_files = [
            f for f in self.audio_dir.rglob('*')
            if f.is_file() and f.suffix.lower() in audio_extensions
        ]

        if not audio_files:
            return {
                'error': 'No audio files found',
                'valid': False,
            }

        # Verify each file
        results = {
            'total_files': len(audio_files),
            'valid_files': 0,
            'corrupt_files': [],
            'empty_files': [],
            'short_files': [],
            'long_files': [],
            'sample_rate_issues': [],
            'channel_issues': [],
        }

        for audio_file in tqdm(audio_files, desc="Verifying audio"):
            try:
                # Try to load the file
                data, sr = sf.read(str(audio_file))

                # Check if empty
                if len(data) == 0:
                    results['empty_files'].append(str(audio_file))
                    continue

                # Check duration (warn if < 1s or > 600s)
                duration = len(data) / sr
                if duration < 1.0:
                    results['short_files'].append({
                        'file': str(audio_file),
                        'duration': duration,
                    })
                elif duration > 600.0:
                    results['long_files'].append({
                        'file': str(audio_file),
                        'duration': duration,
                    })

                # Check sample rate (should be 44100 or 48000 typically)
                if sr not in [16000, 22050, 24000, 44100, 48000]:
                    results['sample_rate_issues'].append({
                        'file': str(audio_file),
                        'sample_rate': sr,
                    })

                # Check channels
                if data.ndim > 1 and data.shape[1] > 2:
                    results['channel_issues'].append({
                        'file': str(audio_file),
                        'channels': data.shape[1],
                    })

                results['valid_files'] += 1

            except Exception as e:
                results['corrupt_files'].append({
                    'file': str(audio_file),
                    'error': str(e),
                })

        # Calculate success rate
        results['success_rate'] = results['valid_files'] / results['total_files']
        results['valid'] = results['success_rate'] >= 0.95  # 95% threshold

        return results

    def verify_metadata(self) -> Dict[str, any]:
        """
        Verify metadata files.

        Returns:
            Dictionary with verification results
        """
        self.logger.info("Verifying metadata...")

        if not self.metadata_dir.exists():
            return {
                'error': 'Metadata directory not found',
                'valid': False,
            }

        results = {
            'files_found': [],
            'missing_files': [],
            'valid': False,
        }

        # Check for required metadata files
        required_files = ['dataset_metadata.json']
        for filename in required_files:
            filepath = self.metadata_dir / filename
            if filepath.exists():
                results['files_found'].append(filename)
            else:
                results['missing_files'].append(filename)

        # Check if we can load the main metadata
        dataset_meta_path = self.metadata_dir / 'dataset_metadata.json'
        if dataset_meta_path.exists():
            try:
                with open(dataset_meta_path, 'r') as f:
                    meta = json.load(f)
                results['dataset_info'] = meta
                results['valid'] = True
            except Exception as e:
                results['error'] = f"Failed to load dataset metadata: {e}"
                results['valid'] = False

        return results

    def verify_text_data(self) -> Dict[str, any]:
        """
        Verify text data quality.

        Returns:
            Dictionary with verification results
        """
        self.logger.info("Verifying text data...")

        # Find metadata files with text
        text_files = list(self.metadata_dir.glob('*.json'))
        samples_files = [f for f in text_files if 'samples' in f.name or 'processed' in f.name]

        if not samples_files:
            return {
                'error': 'No sample metadata files found',
                'valid': False,
            }

        results = {
            'total_samples': 0,
            'samples_with_text': 0,
            'empty_text': [],
            'short_text': [],
            'avg_text_length': 0,
            'text_length_distribution': {},
        }

        text_lengths = []

        for samples_file in samples_files:
            try:
                with open(samples_file, 'r') as f:
                    samples = json.load(f)

                for sample in samples:
                    results['total_samples'] += 1

                    # Look for text fields
                    text = None
                    for key in ['caption', 'text', 'description', 'genres', 'tags']:
                        if key in sample:
                            text = sample[key]
                            break

                    if text:
                        # Handle lists (like genres/tags)
                        if isinstance(text, list):
                            text = ', '.join(str(t) for t in text)
                        else:
                            text = str(text)

                        text_len = len(text)
                        text_lengths.append(text_len)

                        if text_len > 0:
                            results['samples_with_text'] += 1
                        else:
                            results['empty_text'].append(sample.get('id', sample.get('track_id', 'unknown')))

                        if text_len < 10:
                            results['short_text'].append({
                                'id': sample.get('id', sample.get('track_id', 'unknown')),
                                'text': text,
                                'length': text_len,
                            })

            except Exception as e:
                self.logger.warning(f"Error processing {samples_file}: {e}")

        if text_lengths:
            results['avg_text_length'] = np.mean(text_lengths)
            results['median_text_length'] = np.median(text_lengths)
            results['min_text_length'] = np.min(text_lengths)
            results['max_text_length'] = np.max(text_lengths)

            # Distribution
            results['text_length_distribution'] = {
                '0-10': sum(1 for l in text_lengths if 0 <= l < 10),
                '10-50': sum(1 for l in text_lengths if 10 <= l < 50),
                '50-100': sum(1 for l in text_lengths if 50 <= l < 100),
                '100-200': sum(1 for l in text_lengths if 100 <= l < 200),
                '200+': sum(1 for l in text_lengths if l >= 200),
            }

        results['text_coverage'] = results['samples_with_text'] / results['total_samples'] if results['total_samples'] > 0 else 0
        results['valid'] = results['text_coverage'] >= 0.8  # 80% threshold

        return results

    def verify_all(self) -> Dict[str, any]:
        """
        Run all verification checks.

        Returns:
            Dictionary with all verification results
        """
        self.logger.info(f"Verifying dataset: {self.dataset_dir}")

        results = {
            'dataset_dir': str(self.dataset_dir),
            'audio': self.verify_audio_files(),
            'metadata': self.verify_metadata(),
            'text': self.verify_text_data(),
        }

        # Overall validity
        results['valid'] = all([
            results['audio'].get('valid', False),
            results['metadata'].get('valid', False),
            results['text'].get('valid', False),
        ])

        return results

    def print_report(self, results: Dict[str, any]) -> None:
        """Print a human-readable verification report."""
        print("\n" + "="*80)
        print(f"VERIFICATION REPORT: {self.dataset_dir.name}")
        print("="*80)

        # Audio verification
        print("\nAUDIO FILES:")
        audio = results['audio']
        if 'error' in audio:
            print(f"  ✗ Error: {audio['error']}")
        else:
            print(f"  Total files: {audio['total_files']}")
            print(f"  Valid files: {audio['valid_files']}")
            print(f"  Success rate: {audio['success_rate']*100:.1f}%")
            if audio['corrupt_files']:
                print(f"  ✗ Corrupt files: {len(audio['corrupt_files'])}")
            if audio['empty_files']:
                print(f"  ✗ Empty files: {len(audio['empty_files'])}")
            if audio['short_files']:
                print(f"  ⚠ Short files (<1s): {len(audio['short_files'])}")
            if audio['sample_rate_issues']:
                print(f"  ⚠ Sample rate issues: {len(audio['sample_rate_issues'])}")

        # Metadata verification
        print("\nMETADATA:")
        metadata = results['metadata']
        if 'error' in metadata:
            print(f"  ✗ Error: {metadata['error']}")
        else:
            print(f"  Files found: {', '.join(metadata['files_found'])}")
            if metadata['missing_files']:
                print(f"  ✗ Missing files: {', '.join(metadata['missing_files'])}")

        # Text verification
        print("\nTEXT DATA:")
        text = results['text']
        if 'error' in text:
            print(f"  ✗ Error: {text['error']}")
        else:
            print(f"  Total samples: {text['total_samples']}")
            print(f"  Samples with text: {text['samples_with_text']}")
            print(f"  Text coverage: {text['text_coverage']*100:.1f}%")
            if 'avg_text_length' in text:
                print(f"  Avg text length: {text['avg_text_length']:.1f} chars")
                print(f"  Median text length: {text['median_text_length']:.1f} chars")
                print(f"  Text length distribution:")
                for range_str, count in text['text_length_distribution'].items():
                    print(f"    {range_str} chars: {count}")

        # Overall status
        print("\n" + "="*80)
        if results['valid']:
            print("✓ DATASET VERIFICATION PASSED")
        else:
            print("✗ DATASET VERIFICATION FAILED")
        print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Verify downloaded datasets')
    parser.add_argument(
        'dataset_dir',
        type=str,
        help='Directory containing the dataset to verify'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for verification results'
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
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Verify dataset
    verifier = DatasetVerifier(Path(args.dataset_dir))
    results = verifier.verify_all()
    verifier.print_report(results)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
