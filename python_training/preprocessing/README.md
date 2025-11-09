# Dataset Preprocessing

This directory contains tools for preprocessing music datasets and generating audio embeddings for training music-text alignment models.

## Overview

The preprocessing pipeline generates audio embeddings from raw audio files using multiple pre-trained models. These embeddings are then used to train music-text alignment models.

## Available Models

- **muq**: MuQ embedding model (dimension: 1536)
- **mert**: MERT model (dimension: 76800)
- **music2latent**: Music Latent Space model (dimension: 576)

## Available Datasets

- **musiccaps**: ~5.5k clips with detailed captions
- **fma_large**: 106k tracks 30s clips
- **fma_full**: 106k full tracks with extracted clips
- **jamendo**: ~5.5k tracks with tags

## Scripts

### `run_preprocessing.sh` (Recommended)

Convenient shell script for running preprocessing with sensible defaults.

**Usage:**

```bash
# Make executable (first time only)
chmod +x run_preprocessing.sh

# Process all datasets with all models
./run_preprocessing.sh

# Test run with 100 samples
./run_preprocessing.sh --test

# Process specific dataset
./run_preprocessing.sh --dataset musiccaps

# Use specific model
./run_preprocessing.sh --model muq

# Use CPU instead of GPU
./run_preprocessing.sh --cpu

# Force re-processing
./run_preprocessing.sh --force

# Combine options
./run_preprocessing.sh --dataset musiccaps --model muq --test

# Get help
./run_preprocessing.sh --help
```

This script automatically:
- Validates the datasets directory exists
- Runs preprocessing with your specified options
- Verifies the generated embeddings
- Shows a summary of results

### `preprocess_all.py`

Main script for preprocessing all datasets with all models.

**Usage:**

```bash
# Preprocess all datasets with all models
python preprocess_all.py

# Preprocess specific dataset with all models
python preprocess_all.py --dataset musiccaps

# Preprocess all datasets with specific model
python preprocess_all.py --model muq

# Preprocess specific dataset with specific model
python preprocess_all.py --dataset musiccaps --model muq

# Process multiple specific datasets
python preprocess_all.py --dataset musiccaps --dataset jamendo

# Process multiple specific models
python preprocess_all.py --model muq --model mert

# Limit samples for testing
python preprocess_all.py --max-samples 100

# Force re-processing (overwrite existing embeddings)
python preprocess_all.py --force

# Custom directories
python preprocess_all.py \
  --datasets-dir /path/to/datasets \
  --output-dir /path/to/output

# Use CPU instead of GPU
python preprocess_all.py --device cpu

# Adjust batch size
python preprocess_all.py --batch-size 16
```

**Arguments:**

- `--datasets-dir`: Directory containing all datasets (default: `../data/datasets`)
- `--output-dir`: Output directory for preprocessed embeddings (default: `../data/preprocessed`)
- `--dataset`: Specific dataset(s) to preprocess (can be specified multiple times)
- `--model`: Specific model(s) to use (can be specified multiple times)
- `--device`: Device to use for inference (default: `cuda`)
- `--batch-size`: Batch size for processing (default: 8)
- `--max-samples`: Maximum number of samples per dataset (for testing)
- `--force`: Force re-processing even if embeddings already exist
- `--log-level`: Logging level (default: `INFO`)

### `audio_embedding_generator.py`

Lower-level script for generating embeddings for a single dataset with a single model.

**Usage:**

```bash
python audio_embedding_generator.py \
  /path/to/dataset \
  /path/to/output.h5 \
  --model muq \
  --device cuda \
  --batch-size 8
```

### `verify_embeddings.py`

Utility script for verifying and inspecting preprocessed embeddings.

**Usage:**

```bash
# Verify all embeddings
python verify_embeddings.py

# Show detailed report
python verify_embeddings.py --detailed

# Verify embeddings in custom directory
python verify_embeddings.py --preprocessed-dir /path/to/preprocessed

# Save results to JSON
python verify_embeddings.py --output verification_results.json
```

**Checks performed:**

- File existence
- HDF5 structure validity
- Shape consistency
- NaN/Inf detection
- Zero embeddings (failed samples)
- Basic statistics (mean, std, min, max)

## Output Format

Embeddings are saved in HDF5 format (`.h5` files) with the following structure:

- `embeddings`: Array of shape `(num_samples, embedding_dim)` containing the embeddings
- `file_paths`: Array of relative file paths for each audio file
- Metadata attributes:
  - `model_name`: Name of the model used
  - `embedding_dim`: Dimension of embeddings
  - `num_samples`: Number of samples processed
  - `dataset_dir`: Source dataset directory

## Directory Structure

After preprocessing, the output directory will have this structure:

```
data/preprocessed/
├── preprocessing_summary.json      # Summary of all preprocessing jobs
├── musiccaps/
│   ├── muq_embeddings.h5
│   ├── mert_embeddings.h5
│   └── music2latent_embeddings.h5
├── fma_large/
│   ├── muq_embeddings.h5
│   ├── mert_embeddings.h5
│   └── music2latent_embeddings.h5
├── fma_full/
│   └── ...
└── jamendo/
    └── ...
```

## Quick Start

```bash
# 1. Download datasets (if not already done)
cd ../datasets
python download_all.py --all
cd ../preprocessing

# 2. Test preprocessing on a small sample
./run_preprocessing.sh --test --dataset musiccaps --model muq

# 3. Run full preprocessing
./run_preprocessing.sh

# 4. View detailed results
python verify_embeddings.py --detailed
```

## Workflow

1. **Download datasets** (if not already done):
   ```bash
   cd ../datasets
   python download_all.py --all
   ```

2. **Test preprocessing on a small sample** (recommended first step):
   ```bash
   # Using shell script (recommended)
   ./run_preprocessing.sh --test --dataset musiccaps --model muq

   # Or using Python directly
   python preprocess_all.py --dataset musiccaps --model muq --max-samples 10
   ```

3. **Run full preprocessing**:
   ```bash
   # Using shell script (recommended - includes verification)
   ./run_preprocessing.sh

   # Or using Python directly
   python preprocess_all.py
   ```

4. **Verify the embeddings**:
   ```bash
   python verify_embeddings.py --detailed
   ```

5. **Check the summary**:
   ```bash
   cat ../data/preprocessed/preprocessing_summary.json
   ```

## Performance Tips

- **GPU Memory**: If you run out of GPU memory, reduce `--batch-size`
- **Parallel Processing**: Process different datasets in parallel by running multiple instances with different `--dataset` arguments
- **Testing**: Use `--max-samples` to test on a small subset before processing everything
- **Resume**: The script automatically skips already-processed datasets unless `--force` is specified

## Requirements

- PyTorch
- h5py
- numpy
- tqdm
- soundfile
- python_services (for embedding models)

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python preprocess_all.py --batch-size 1
```

### CUDA Not Available
```bash
# Use CPU
python preprocess_all.py --device cpu
```

### Dataset Not Found
```bash
# Check available datasets
ls -la ../data/datasets/

# Specify custom datasets directory
python preprocess_all.py --datasets-dir /path/to/datasets
```

### Verify Embeddings
```python
import h5py

# Load and inspect embeddings
with h5py.File('../data/preprocessed/musiccaps/muq_embeddings.h5', 'r') as f:
    print("Embeddings shape:", f['embeddings'].shape)
    print("Embedding dim:", f.attrs['embedding_dim'])
    print("Model:", f.attrs['model_name'])
    print("Num samples:", f.attrs['num_samples'])
    print("First file:", f['file_paths'][0])
```
