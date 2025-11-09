"""Verify and inspect preprocessed embeddings."""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import h5py
import numpy as np


def verify_embedding_file(embedding_file: Path) -> Dict:
    """
    Verify a single embedding file.

    Args:
        embedding_file: Path to HDF5 embedding file

    Returns:
        Dictionary with verification results
    """
    result = {
        'file': str(embedding_file),
        'exists': embedding_file.exists(),
        'valid': False,
        'error': None,
    }

    if not embedding_file.exists():
        result['error'] = "File does not exist"
        return result

    try:
        with h5py.File(embedding_file, 'r') as f:
            # Check required datasets
            if 'embeddings' not in f:
                result['error'] = "Missing 'embeddings' dataset"
                return result

            if 'file_paths' not in f:
                result['error'] = "Missing 'file_paths' dataset"
                return result

            # Get metadata
            embeddings = f['embeddings']
            file_paths = f['file_paths']

            result['num_samples'] = embeddings.shape[0]
            result['embedding_dim'] = embeddings.shape[1]
            result['model_name'] = f.attrs.get('model_name', 'unknown')
            result['dataset_dir'] = f.attrs.get('dataset_dir', 'unknown')

            # Check shape consistency
            if embeddings.shape[0] != file_paths.shape[0]:
                result['error'] = f"Shape mismatch: embeddings={embeddings.shape[0]}, file_paths={file_paths.shape[0]}"
                return result

            # Check for NaN or Inf values
            chunk_size = 1000
            has_nan = False
            has_inf = False

            for i in range(0, embeddings.shape[0], chunk_size):
                end_idx = min(i + chunk_size, embeddings.shape[0])
                chunk = embeddings[i:end_idx]

                if np.any(np.isnan(chunk)):
                    has_nan = True
                if np.any(np.isinf(chunk)):
                    has_inf = True

            result['has_nan'] = has_nan
            result['has_inf'] = has_inf

            # Compute basic statistics
            # Sample a subset for efficiency
            sample_size = min(10000, embeddings.shape[0])
            sample_indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
            sample_embeddings = embeddings[sample_indices]

            result['stats'] = {
                'mean': float(np.mean(sample_embeddings)),
                'std': float(np.std(sample_embeddings)),
                'min': float(np.min(sample_embeddings)),
                'max': float(np.max(sample_embeddings)),
            }

            # Check for zero embeddings (failures)
            zero_count = 0
            for i in range(0, embeddings.shape[0], chunk_size):
                end_idx = min(i + chunk_size, embeddings.shape[0])
                chunk = embeddings[i:end_idx]
                zero_count += np.sum(np.all(chunk == 0, axis=1))

            result['zero_embeddings'] = int(zero_count)
            result['zero_embeddings_pct'] = (zero_count / embeddings.shape[0]) * 100

            # Mark as valid if no critical errors
            if not has_nan and not has_inf:
                result['valid'] = True
            else:
                result['error'] = f"Data quality issues: has_nan={has_nan}, has_inf={has_inf}"

    except Exception as e:
        result['error'] = str(e)

    return result


def verify_all_embeddings(preprocessed_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """
    Verify all embedding files in the preprocessed directory.

    Args:
        preprocessed_dir: Directory containing preprocessed embeddings

    Returns:
        Nested dictionary: dataset_name -> model_name -> verification_result
    """
    results = {}

    # Find all .h5 files
    for dataset_dir in preprocessed_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        results[dataset_name] = {}

        for embedding_file in dataset_dir.glob("*.h5"):
            # Extract model name from filename (e.g., "muq_embeddings.h5" -> "muq")
            model_name = embedding_file.stem.replace('_embeddings', '')

            print(f"Verifying {dataset_name}/{model_name}...")
            result = verify_embedding_file(embedding_file)
            results[dataset_name][model_name] = result

    return results


def print_summary(results: Dict[str, Dict[str, Dict]]) -> None:
    """Print a summary of verification results."""
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80 + "\n")

    total_files = 0
    valid_files = 0
    invalid_files = 0
    missing_files = 0

    for dataset_name, model_results in results.items():
        print(f"{dataset_name}:")
        for model_name, result in model_results.items():
            total_files += 1

            if not result['exists']:
                status = "✗ MISSING"
                missing_files += 1
            elif result['valid']:
                status = "✓ VALID"
                valid_files += 1
                # Add additional info
                status += f" ({result['num_samples']:,} samples, dim={result['embedding_dim']})"
                if result['zero_embeddings'] > 0:
                    status += f" [{result['zero_embeddings']} zeros ({result['zero_embeddings_pct']:.1f}%)]"
            else:
                status = "✗ INVALID"
                invalid_files += 1
                if result['error']:
                    status += f" - {result['error']}"

            print(f"  {model_name}: {status}")

        print()

    print("="*80)
    print(f"Total files: {total_files}")
    print(f"Valid: {valid_files}")
    print(f"Invalid: {invalid_files}")
    print(f"Missing: {missing_files}")

    if total_files > 0:
        print(f"Success rate: {valid_files/total_files*100:.1f}%")

    print("="*80 + "\n")


def print_detailed_report(results: Dict[str, Dict[str, Dict]]) -> None:
    """Print a detailed report of verification results."""
    print("\n" + "="*80)
    print("DETAILED REPORT")
    print("="*80 + "\n")

    for dataset_name, model_results in results.items():
        print(f"\n{'#'*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'#'*80}\n")

        for model_name, result in model_results.items():
            print(f"Model: {model_name}")
            print(f"  File: {result['file']}")
            print(f"  Exists: {result['exists']}")
            print(f"  Valid: {result['valid']}")

            if result['exists']:
                print(f"  Samples: {result.get('num_samples', 'N/A'):,}")
                print(f"  Embedding dim: {result.get('embedding_dim', 'N/A')}")
                print(f"  Model name: {result.get('model_name', 'N/A')}")
                print(f"  Dataset dir: {result.get('dataset_dir', 'N/A')}")

                if 'stats' in result:
                    print(f"  Statistics:")
                    print(f"    Mean: {result['stats']['mean']:.6f}")
                    print(f"    Std: {result['stats']['std']:.6f}")
                    print(f"    Min: {result['stats']['min']:.6f}")
                    print(f"    Max: {result['stats']['max']:.6f}")

                print(f"  Has NaN: {result.get('has_nan', 'N/A')}")
                print(f"  Has Inf: {result.get('has_inf', 'N/A')}")
                print(f"  Zero embeddings: {result.get('zero_embeddings', 'N/A')} ({result.get('zero_embeddings_pct', 'N/A'):.1f}%)")

            if result['error']:
                print(f"  Error: {result['error']}")

            print()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description='Verify preprocessed embeddings')

    parser.add_argument(
        '--preprocessed-dir',
        type=str,
        default='../data/preprocessed',
        help='Directory containing preprocessed embeddings (default: ../data/preprocessed)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Print detailed report'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )

    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir).resolve()

    if not preprocessed_dir.exists():
        print(f"Error: Directory does not exist: {preprocessed_dir}")
        return

    # Verify all embeddings
    results = verify_all_embeddings(preprocessed_dir)

    # Print summary
    print_summary(results)

    # Print detailed report if requested
    if args.detailed:
        print_detailed_report(results)

    # Save to JSON if requested
    if args.output:
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
