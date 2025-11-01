# Text-to-Audio Embedding Training

This directory contains scripts to train text-to-audio projection models that map text embeddings (from Qwen3-Embedding-4B) into the audio embedding spaces of your 3 audio models (MuQ, MERT, Music2Latent).

## Overview

**Goal**: Enable text-based similarity search for music by projecting text queries into the same embedding space as your audio embeddings.

**Approach**: Contrastive learning (InfoNCE loss) to align text and audio embeddings using the MusicBench dataset.

**Models Trained**:
1. **MuQ Projection**: Qwen3 (2560D) → MuQ audio space (1536D)
2. **MERT Projection**: Qwen3 (2560D) → MERT audio space (76,800D)
3. **Music2Latent Projection**: Qwen3 (2560D) → Music2Latent audio space (576D)

## Files

- **`prepare_dataset.py`**: Pre-processes MusicBench dataset through all 3 audio embedders
- **`train.py`**: Trains a single projection model
- **`train_all.sh`**: Helper script to train all 3 models sequentially
- **`requirements_training.txt`**: Python dependencies
- **`TRAINING_README.md`**: This file

## Requirements

### Hardware
- **GPU**: 32GB VRAM recommended (tested configuration)
- **Disk**: ~50GB for pre-computed embeddings
- **RAM**: 32GB+ recommended

### Software
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- Existing audio embedding models (MuQ, MERT, Music2Latent)

## Installation

```bash
# Install dependencies
pip install -r requirements_training.txt

# Verify existing models are available
python -c "from embedding_models import MuQEmbeddingModel, MertModel, MusicLatentSpaceModel; print('Models loaded successfully!')"
```

## Usage

### Step 1: Prepare Dataset

Pre-compute embeddings for all MusicBench audio samples through the 3 audio models:

```bash
python prepare_dataset.py \
    --output-dir data/musicbench_embeddings \
    --device cuda \
    --dtype float32
```

**What this does**:
- Downloads MusicBench dataset (52,800 train + 800 test samples)
- Splits train into 90% train (47,520) / 10% validation (5,280)
- Processes each audio file through MuQ, MERT, and Music2Latent
- Applies enrichment (mean, IQR, temporal derivative)
- Saves embeddings to HDF5 file with both main and alt captions (2x augmentation)

**Expected output**: `data/musicbench_embeddings/embeddings.h5` (~40-50GB)

**Time**: 6-12 hours depending on GPU

**Monitor progress**:
```bash
tail -f prepare_dataset.log
```

### Step 2: Train Models

#### Option A: Train All Models Sequentially (Recommended)

```bash
# Make script executable
chmod +x train_all.sh

# Train all 3 models
./train_all.sh
```

This trains in order:
1. Music2Latent (fastest, validates setup)
2. MuQ (medium complexity)
3. MERT (largest dimension)

**Total time**: ~24-36 hours (8-12 hours per model)

#### Option B: Train Individual Models

Train a specific model:

```bash
# Music2Latent projection (576D)
python train.py --audio-encoder latent

# MuQ projection (1536D)
python train.py --audio-encoder muq

# MERT projection (76,800D)
python train.py --audio-encoder mert
```

**Advanced options**:
```bash
python train.py \
    --audio-encoder muq \
    --epochs 20 \
    --batch-size 32 \
    --lr 5e-5 \
    --text-lr 2e-5 \
    --hidden-dim 1024 \
    --checkpoint-dir checkpoints \
    --log-dir logs/tensorboard
```

### Step 3: Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

Then open http://localhost:6006 in your browser.

**Metrics tracked**:
- Training/validation loss
- Text→Audio retrieval: R@1, R@5, R@10
- Audio→Text retrieval: R@1, R@5, R@10
- Mean Reciprocal Rank (MRR)
- Median Rank
- Learning rate schedule

### Step 4: Evaluate Results

After training completes, check the summary:

```bash
cat checkpoints/training_summary.json
```

Example output:
```json
{
  "latent": {
    "t2a_R@1": 45.2,
    "t2a_R@5": 72.8,
    "t2a_R@10": 83.1,
    "t2a_MRR": 0.5823
  },
  "muq": {
    "t2a_R@1": 52.3,
    "t2a_R@5": 78.9,
    "t2a_R@10": 87.4,
    "t2a_MRR": 0.6421
  },
  "mert": {
    "t2a_R@1": 48.7,
    "t2a_R@5": 75.2,
    "t2a_R@10": 85.3,
    "t2a_MRR": 0.6112
  }
}
```

## Checkpoints

After training, you'll have these checkpoints:

```
checkpoints/
├── latent_best_r1.pt          # Best Music2Latent model (by R@1)
├── latent_best_loss.pt         # Best Music2Latent model (by loss)
├── latent_last.pt              # Latest Music2Latent checkpoint
├── muq_best_r1.pt              # Best MuQ model
├── muq_best_loss.pt
├── muq_last.pt
├── mert_best_r1.pt             # Best MERT model
├── mert_best_loss.pt
├── mert_last.pt
└── training_summary.json       # All test metrics
```

**Recommended**: Use `*_best_r1.pt` checkpoints for deployment (best retrieval performance).

## Using Trained Models

### Loading a Checkpoint

```python
import torch
from train import TextEncoder, ProjectionHead, TrainingConfig

# Load checkpoint
checkpoint = torch.load('checkpoints/muq_best_r1.pt')
config_dict = checkpoint['config']

# Recreate models
config = TrainingConfig(**config_dict)
text_encoder = TextEncoder(config)
projection = ProjectionHead(config.text_dim, config.hidden_dim, config.audio_dim)

# Load weights
text_encoder.model.load_state_dict(checkpoint['text_encoder_state_dict'])
projection.load_state_dict(checkpoint['projection_state_dict'])

# Set to eval mode
text_encoder.eval()
projection.eval()
```

### Text Query Embedding

```python
@torch.no_grad()
def embed_text_query(query: str, encoder, projection, device='cuda'):
    """Embed text query into audio space"""
    encoder.eval()
    projection.eval()

    # Encode text
    text_emb = encoder([query])

    # Project to audio space (returns normalized features)
    text_features, _ = projection(
        text_emb,
        torch.zeros(1, projection.audio_dim, device=device)  # Dummy audio
    )

    return text_features[0].cpu().numpy()

# Example usage
query = "upbeat electronic dance music with heavy bass"
query_embedding = embed_text_query(query, text_encoder, projection)

# Now search in Milvus with this embedding
# results = milvus_collection.search(
#     data=[query_embedding],
#     anns_field="embedding",
#     param={"metric_type": "L2", "params": {"nprobe": 10}},
#     limit=10
# )
```

### Batch Text Embedding

```python
@torch.no_grad()
def embed_texts_batch(texts: List[str], encoder, projection, device='cuda', batch_size=32):
    """Embed multiple texts efficiently"""
    encoder.eval()
    projection.eval()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Encode
        text_emb = encoder(batch_texts)

        # Project
        text_features, _ = projection(
            text_emb,
            torch.zeros(len(batch_texts), projection.audio_dim, device=device)
        )

        all_embeddings.append(text_features.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)

# Example: Index a music library
descriptions = [
    "calm piano solo with gentle melodies",
    "heavy metal guitar riffs with drums",
    "jazz saxophone improvisation",
    # ... thousands more
]

embeddings = embed_texts_batch(descriptions, text_encoder, projection)
# Insert into Milvus...
```

## Training Configuration

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 20 | Maximum training epochs |
| Batch Size | 32 | Per-GPU batch size |
| Gradient Accumulation | 8 | Effective batch = 256 |
| Learning Rate (Projection) | 5e-5 | Projection layer LR |
| Learning Rate (Text LoRA) | 2e-5 | Qwen3 LoRA LR |
| Weight Decay | 0.1 | AdamW weight decay |
| Warmup Steps | 2000 | LR warmup steps |
| Temperature | 0.07 | InfoNCE temperature |
| LoRA Rank | 16 | LoRA adapter rank |
| LoRA Alpha | 32 | LoRA scaling factor |
| Early Stopping Patience | 5 | Epochs without improvement |

### Memory Usage

Approximate GPU memory usage during training:

| Model | VRAM Usage | Notes |
|-------|------------|-------|
| Music2Latent | ~16-18GB | Smallest dimension |
| MuQ | ~18-20GB | Medium dimension |
| MERT | ~22-24GB | Largest dimension (76,800D) |

**Memory optimization enabled**:
- Mixed precision (BF16)
- Gradient checkpointing
- Gradient accumulation
- Frozen audio embeddings (pre-computed)

## Troubleshooting

### Out of Memory

**Reduce batch size**:
```bash
python train.py --audio-encoder mert --batch-size 16
```

**Increase gradient accumulation** (maintains effective batch size):
```bash
# Edit train.py: config.gradient_accumulation_steps = 16
```

### Slow Dataset Preparation

The embedding extraction is compute-intensive. To speed up:

1. Use multiple GPUs:
```bash
# Split dataset manually and process in parallel
CUDA_VISIBLE_DEVICES=0 python prepare_dataset.py --split-id 0 --num-splits 2 &
CUDA_VISIBLE_DEVICES=1 python prepare_dataset.py --split-id 1 --num-splits 2 &
```

2. Use faster precision:
```bash
python prepare_dataset.py --dtype float16
```

### Missing Audio Files

If MusicBench audio download fails, the script will skip those samples. Check `prepare_dataset.log` for errors.

### Poor Retrieval Performance

If R@1 < 20% after training:

1. **Check data**: Verify embeddings were computed correctly
2. **Increase epochs**: Try 30-40 epochs
3. **Adjust temperature**: Try 0.05 or 0.1
4. **Increase hidden dim**: For MERT, try `--hidden-dim 4096`
5. **Check LoRA**: Ensure text encoder is learning (check gradients in TensorBoard)

## Citation

If you use this code or the MusicBench dataset:

```bibtex
@inproceedings{musicbench2024,
  title={MusicBench: A Benchmark for Evaluating Music Understanding and Generation},
  author={...},
  booktitle={...},
  year={2024}
}
```

## License

Same as parent project (Navidrome).

## Support

For issues:
1. Check logs: `prepare_dataset.log`, `train.log`
2. Verify GPU: `nvidia-smi`
3. Check disk space: `df -h data/`
4. Review TensorBoard: `tensorboard --logdir logs/tensorboard`
