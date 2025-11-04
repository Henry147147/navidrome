# Music-Text Embedding Training

This project trains text embedding models to align with audio embedding models in the Navidrome music recommendation system. The goal is to enable text-based music search and recommendation by learning embeddings that capture the semantic relationship between text descriptions and music audio.

## Overview

We train 3 text embedding models, each matching a corresponding audio embedding model:

1. **MuQ Text Embedder** → Aligns with MuQ MuLan (1536D)
2. **MERT Text Embedder** → Aligns with MERT-v1-330M (76,800D)
3. **Music2Latent Text Embedder** → Aligns with Music2Latent (576D)

## Project Structure

```
python_training/
├── datasets/                    # Dataset downloaders and management
│   ├── downloaders/            # Individual dataset downloaders
│   │   ├── musiccaps_downloader.py
│   │   ├── fma_downloader.py
│   │   ├── jamendo_downloader.py
│   │   └── musicbench_downloader.py
│   ├── verification/           # Dataset verification tools
│   ├── download_all.py         # Master download script
│   └── base_downloader.py
├── models/                      # Text embedding models
│   ├── text_embedders/
│   │   ├── muq_text_embedder.py
│   │   ├── mert_text_embedder.py
│   │   └── music2latent_text_embedder.py
│   └── base_text_embedder.py
├── preprocessing/               # Audio embedding generation
│   └── audio_embedding_generator.py
├── training/                    # Training infrastructure
│   ├── train.py                # Main training script
│   └── losses.py               # Loss functions
├── utils/                       # Utilities
│   └── data_loader.py          # PyTorch data loaders
├── configs/                     # Training configurations
│   ├── muq_config.json
│   ├── mert_config.json
│   └── music2latent_config.json
├── TRAINING_PLAN.md            # Detailed training plan
└── README.md                   # This file
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install torch torchaudio transformers datasets
pip install h5py soundfile librosa yt-dlp
pip install wandb tqdm pandas

# Install existing python_services dependencies
cd ../python_services
pip install -r requirements.txt
```

### 2. Download Datasets

Download all Phase 1 datasets (large-scale short clips):

```bash
cd python_training/datasets
python download_all.py --phase 1 --output-dir ../../data/datasets
```

Or download a specific dataset:

```bash
python download_all.py --dataset musiccaps --output-dir ../../data/datasets
```

Check download statistics:

```bash
python download_all.py --stats --output-dir ../../data/datasets
```

### 3. Verify Downloaded Datasets

```bash
cd verification
python verify_datasets.py ../../data/datasets/musiccaps
```

### 4. Generate Audio Embeddings

Generate audio embeddings for training (this step uses the existing audio models from `python_services`):

```bash
# MuQ embeddings
python preprocessing/audio_embedding_generator.py \
    ../data/datasets/musiccaps \
    ../data/embeddings/muq/musiccaps_embeddings.h5 \
    --model muq \
    --device cuda

# MERT embeddings
python preprocessing/audio_embedding_generator.py \
    ../data/datasets/musiccaps \
    ../data/embeddings/mert/musiccaps_embeddings.h5 \
    --model mert \
    --device cuda

# Music2Latent embeddings
python preprocessing/audio_embedding_generator.py \
    ../data/datasets/musiccaps \
    ../data/embeddings/music2latent/musiccaps_embeddings.h5 \
    --model music2latent \
    --device cuda
```

### 5. Train Text Embedding Models

Train a text embedding model:

```bash
# Train MuQ text embedder
python training/train.py --config configs/muq_config.json

# Train MERT text embedder
python training/train.py --config configs/mert_config.json

# Train Music2Latent text embedder
python training/train.py --config configs/music2latent_config.json
```

Resume training from a checkpoint:

```bash
python training/train.py \
    --config configs/muq_config.json \
    --resume checkpoints/muq/latest.pt
```

Disable Weights & Biases logging:

```bash
python training/train.py \
    --config configs/muq_config.json \
    --no-wandb
```

## Training Strategy

### Two-Stage Training

#### Stage 1: Large Corpus Pre-training
- **Data**: Phase 1 datasets (short clips: 5-30 seconds)
- **Duration**: 100 epochs
- **Batch Size**: 256 (MuQ/Music2Latent), 64 (MERT)
- **Goal**: Learn general music-text alignment from large, diverse corpus

Datasets:
- MusicCaps: 5.5k clips with detailed captions
- MusicBench: 50k samples with prompts
- FMA Large: 106k 30s clips
- MTG-Jamendo: 55k tracks with tags

#### Stage 2: Full Song Fine-tuning
- **Data**: Phase 2 datasets (full songs: 3-5 minutes)
- **Duration**: 50 epochs
- **Batch Size**: 64-128
- **Goal**: Refine with complete musical context

Datasets:
- FMA Full: Full-length tracks with extracted clips
- Custom high-quality song-description pairs

### Loss Functions

**InfoNCE Loss (Contrastive Learning)**:
- Encourages matching text-audio pairs to have high similarity
- Non-matching pairs have low similarity
- Bidirectional: text→audio and audio→text

**Combined Loss**:
- InfoNCE: Contrastive alignment (weight: 1.0)
- MSE: Direct embedding matching (weight: 0.1)

## Model Architectures

### MuQ Text Embedder (1536D)
```
Base: sentence-transformers/all-MiniLM-L6-v2
Architecture:
  - Transformer encoder (6 layers, 384 hidden dim)
  - Projection: 384 → 512
  - Enrichment: 512 → 1536 (mean, robust_sigma, dmean)
Parameters: ~23M
Training: ~8 hours on A100
```

### MERT Text Embedder (76,800D)
```
Base: roberta-base
Architecture:
  - Transformer encoder (12 layers, 768 hidden dim)
  - Multi-layer extraction (all 12 layers)
  - Projection: 9216 → 25,600
  - Enrichment: 25,600 → 76,800
Parameters: ~125M
Training: ~20 hours on A100
```

### Music2Latent Text Embedder (576D)
```
Base: distilbert-base-uncased
Architecture:
  - Transformer encoder (6 layers, 768 hidden dim)
  - Projection: 768 → 384 → 192
  - Enrichment: 192 → 576
Parameters: ~67M
Training: ~6 hours on A100
```

## Dataset Information

### Phase 1: Large-Scale Short Clips

| Dataset | Samples | Duration | Size | Text Quality |
|---------|---------|----------|------|--------------|
| MusicCaps | 5.5k | 10s clips | 50GB | High |
| MusicBench | 50k | Varies | 50GB | High |
| FMA Large | 106k | 30s clips | 93GB | Medium |
| MTG-Jamendo | 55k | 30s clips | 200GB | Medium |
| **Total** | **~220k** | **~1800 hours** | **~400GB** | - |

### Phase 2: High-Quality Full Songs

| Dataset | Samples | Duration | Size | Text Quality |
|---------|---------|----------|------|--------------|
| FMA Full | 106k × 3 clips | Full tracks | 900GB | Medium |
| Custom datasets | 10-50k | Full songs | 100-500GB | High |

### Total Dataset: ~2-2.5TB

## Hardware Requirements

### Minimum
- GPU: 1x NVIDIA RTX 3090 (24GB)
- RAM: 64GB
- Storage: 3TB (datasets + embeddings + checkpoints)

### Recommended
- GPU: 2-4x NVIDIA A100 (40GB or 80GB)
- RAM: 256GB
- Storage: 5TB
- Network: High-bandwidth for dataset downloads

## Configuration

Training configurations are in `configs/*.json`. Key parameters:

```json
{
  "model": {
    "type": "muq",  // or "mert", "music2latent"
    "params": {
      "base_model": "...",
      "target_dim": 512,
      "enrichment_enabled": true
    }
  },
  "training": {
    "num_epochs": 100,
    "loss": {
      "type": "combined",
      "params": {
        "temperature": 0.07,
        "infonce_weight": 1.0,
        "mse_weight": 0.1
      }
    },
    "optimizer": {
      "lr": 1e-4,
      "weight_decay": 0.01
    }
  }
}
```

## Evaluation

Models are evaluated on:

1. **Retrieval Metrics**:
   - Text→Music Recall@K (K=1,5,10)
   - Music→Text Recall@K
   - Mean Reciprocal Rank (MRR)

2. **Embedding Quality**:
   - Cosine similarity distribution
   - Embedding space visualization (t-SNE/UMAP)

3. **Downstream Tasks**:
   - Music search accuracy
   - Genre classification from text
   - Mood prediction from text

## Integration with Navidrome

After training, the text embedders can be integrated into the Navidrome recommendation system:

1. Export trained models to HuggingFace format (done automatically)
2. Load models in `python_services/embedding_models.py`
3. Use for text-based music search and recommendation

Example usage:

```python
from models.text_embedders.muq_text_embedder import MuQTextEmbedder

# Load trained model
model = MuQTextEmbedder.from_pretrained("checkpoints/muq/best_model")

# Encode text query
texts = ["Upbeat electronic dance music with strong bass"]
embeddings = model.encode_texts(texts, device="cuda")

# Use embeddings for search in vector database
results = milvus_client.search(
    collection_name="embedding",
    data=embeddings.tolist(),
    limit=10
)
```

## Monitoring

Training progress is logged to Weights & Biases. Key metrics:

- Training/validation loss (total, InfoNCE, MSE)
- Learning rate schedule
- Gradient norms
- Embedding statistics

Access dashboard:
```
wandb login
# Training will log to: wandb.ai/your-project/music-text-embeddings
```

## Troubleshooting

### Out of Memory

Reduce batch size in config:
```json
"loader_params": {
  "batch_size": 128  // Reduce from 256
}
```

### YouTube Downloads Failing

Some videos may be unavailable. This is expected. The verification accepts 90%+ success rate.

### Slow Training

- Enable mixed precision training (add to train.py)
- Use gradient accumulation
- Increase num_workers in data loader
- Use faster storage (NVMe SSD)

## Citation

If you use this code or models, please cite:

```bibtex
@software{navidrome_music_text_embeddings,
  title = {Music-Text Embedding Training for Navidrome},
  year = {2025},
  url = {https://github.com/navidrome/navidrome}
}
```

## License

See main Navidrome repository for license information.

## Contributing

Contributions are welcome! Areas for improvement:

- Additional dataset downloaders
- More loss functions (e.g., triplet, NT-Xent)
- Model architectures
- Evaluation scripts
- Documentation

## Contact

For questions or issues, please open an issue in the main Navidrome repository.
