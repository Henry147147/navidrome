# Music-Text Embedding Training Plan

## Overview
This project aims to train 3 text embedding models that align with existing audio embedding models in the Navidrome system. Each text embedder will produce vectors matching the dimensionality of its corresponding audio embedder.

## Target Models

### 1. MuQ Text Embedder (1536D)
- **Target Dimension**: 1536 (matching MuQEmbeddingModel)
- **Audio Model**: MuQ MuLan (OpenMuQ/MuQ-MuLan-large)
- **Base Model**: Fine-tuned transformer (e.g., BERT, RoBERTa, or GPT-based)
- **Training Strategy**: Two-stage training (large corpus → full songs)

### 2. MERT Text Embedder (76,800D)
- **Target Dimension**: 76,800 (matching MertModel)
- **Audio Model**: MERT-v1-330M (m-a-p/MERT-v1-330M)
- **Base Model**: Large transformer with projection layers
- **Training Strategy**: Two-stage training with special handling for high dimensionality

### 3. Music2Latent Text Embedder (576D)
- **Target Dimension**: 576 (matching MusicLatentSpaceModel)
- **Audio Model**: Music2Latent EncoderDecoder
- **Base Model**: Lightweight transformer
- **Training Strategy**: Two-stage training (large corpus → full songs)

## Dataset Plan

### Phase 1: Large-Scale Short Clips (~1TB)
Target: ~1,000,000+ music-text pairs with short clips (5-30 seconds)

#### Primary Datasets:

1. **MusicCaps** (~5.5k clips, ~50GB)
   - Source: Google Research
   - Format: 10-second clips with detailed captions
   - URL: https://huggingface.co/datasets/google/MusicCaps
   - Quality: High - professional annotations
   - Download: Direct from HuggingFace

2. **AudioSet (Music subset)** (~500k clips, ~400GB)
   - Source: Google Research
   - Format: 10-second clips from YouTube with labels
   - URL: https://research.google.com/audioset/
   - Quality: Medium - category labels, needs filtering for music
   - Download: YouTube download via yt-dlp, filter by music categories
   - Categories: Musical instruments, Music genres, Music mood

3. **MTG-Jamendo** (~55k tracks, ~200GB as clips)
   - Source: Music Technology Group
   - Format: Full tracks with tags, instruments, mood
   - URL: https://github.com/MTG/mtg-jamendo-dataset
   - Quality: Medium-High - multiple annotation types
   - Download: Direct download + API
   - Processing: Extract 30s clips from each track

4. **Free Music Archive (FMA)** (~100k tracks, ~900GB as clips)
   - Source: Free Music Archive
   - Format: Full tracks with metadata and genres
   - URL: https://github.com/mdeff/fma
   - Quality: Medium - genre and metadata tags
   - Download: Direct download (large dataset)
   - Processing: Extract multiple 30s clips per track

5. **MusicBench** (~50k samples, ~50GB)
   - Source: Music generation research
   - Format: Music clips with text prompts
   - URL: https://huggingface.co/datasets/amaai-lab/MusicBench
   - Quality: High - detailed text descriptions
   - Download: HuggingFace datasets

6. **Magnatagatune** (~25k clips, ~20GB)
   - Source: Music annotation research
   - Format: 29-second clips with tags
   - URL: https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
   - Quality: Medium - crowd-sourced tags
   - Download: Direct download

#### Estimated Total Phase 1: ~1.6TB raw, ~1TB processed

### Phase 2: High-Quality Full Songs
Target: ~10,000-50,000 full-song music-text pairs

#### Primary Datasets:

1. **Song Describer Dataset** (Custom - TBD)
   - Source: To be collected/scraped
   - Format: Full songs with paragraph descriptions
   - Quality: High - detailed descriptions
   - Collection: Manual curation + API scraping
   - Estimated: 5,000-10,000 songs (~50-100GB)

2. **MusicQA Extended** (~1k songs, ~10GB)
   - Source: Research dataset
   - Format: Full songs with Q&A pairs
   - URL: https://github.com/ductai199x/music-qa
   - Quality: High - detailed annotations
   - Processing: Convert Q&A to descriptions

3. **Million Song Dataset + Last.fm tags** (~50k subset, ~500GB)
   - Source: Million Song Dataset + Last.fm
   - Format: Full songs with user tags and metadata
   - URL: http://millionsongdataset.com/
   - Quality: Medium - user-generated tags
   - Download: Subset of full dataset with audio
   - Processing: Aggregate tags into descriptions

4. **AllMusic Scraped Dataset** (Custom - ~10k songs, ~100GB)
   - Source: AllMusic.com (scraped with permission)
   - Format: Full songs with professional reviews
   - Quality: Very High - professional descriptions
   - Collection: Web scraping + matching with available audio
   - Estimated: 10,000 songs

5. **Spotify API + Descriptions** (Custom - ~20k songs, ~200GB)
   - Source: Spotify API + Generated descriptions
   - Format: Full songs with audio features + LLM-generated descriptions
   - Quality: Medium - synthetic descriptions based on audio features
   - Collection: API access + audio download
   - Processing: Use audio features to generate rich text descriptions

#### Estimated Total Phase 2: ~1TB

### Total Dataset Size: ~2-2.5TB

## Dataset Preprocessing Pipeline

### Audio Processing (using existing embedders):
1. Load audio file (mono, correct sample rate for each model)
2. Chunk if necessary (based on model's window/hop parameters)
3. Generate embeddings using existing audio models:
   - MuQ: `MuQEmbeddingModel.embed_audio_tensor()`
   - MERT: `MertModel.embed_audio_tensor()`
   - Music2Latent: `MusicLatentSpaceModel.embed_audio_tensor()`
4. Apply enrichment (mean, robust_sigma, dmean)
5. Store embeddings in efficient format (HDF5 or NPY)

### Text Processing:
1. Clean text (remove special chars, normalize)
2. Tokenize using model-specific tokenizer
3. Store tokenized text with audio embedding reference
4. Create training pairs: (text_tokens, audio_embedding)

## Training Strategy

### Stage 1: Large Corpus Pre-training
- **Duration**: 100-200 epochs
- **Data**: Phase 1 datasets (short clips)
- **Batch Size**: 256-512 per GPU
- **Loss**: Contrastive loss (InfoNCE) between text and audio embeddings
- **Goal**: Learn general music-text alignment

### Stage 2: Full Song Fine-tuning
- **Duration**: 50-100 epochs
- **Data**: Phase 2 datasets (full songs)
- **Batch Size**: 64-128 per GPU (larger audio)
- **Loss**: Contrastive loss + MSE for embedding matching
- **Goal**: Refine understanding with complete musical context

## Model Architectures

### MuQ Text Embedder (1536D)
```
Base: sentence-transformers/all-MiniLM-L6-v2 or BERT-base
Architecture:
- Transformer encoder (6-12 layers)
- Projection layer: hidden_dim → 512
- 3x enrichment applied during training → 1536D
```

### MERT Text Embedder (76,800D)
```
Base: GPT-2 or RoBERTa-large
Architecture:
- Large transformer (12+ layers)
- Multi-layer projection: hidden_dim → 25600
- 3x enrichment applied during training → 76,800D
- Note: May use dimension reduction techniques
```

### Music2Latent Text Embedder (576D)
```
Base: DistilBERT or small BERT variant
Architecture:
- Lightweight transformer (6 layers)
- Projection layer: hidden_dim → 192
- 3x enrichment applied during training → 576D
```

## Training Infrastructure

### Hardware Requirements:
- GPU: 2-4x NVIDIA A100 (40GB or 80GB) or equivalent
- RAM: 256GB+ system memory
- Storage: 5TB+ (raw data + processed data + checkpoints)
- Network: High-bandwidth for dataset downloads

### Software Stack:
- PyTorch 2.0+
- transformers (HuggingFace)
- datasets (HuggingFace)
- wandb (experiment tracking)
- accelerate (distributed training)

## Evaluation Metrics

1. **Retrieval Metrics**:
   - Text-to-Music Recall@K (K=1,5,10)
   - Music-to-Text Recall@K
   - Mean Reciprocal Rank (MRR)

2. **Embedding Quality**:
   - Cosine similarity distribution
   - Embedding space visualization (t-SNE/UMAP)

3. **Downstream Tasks**:
   - Music search accuracy
   - Genre classification from text
   - Mood prediction from text

## Implementation Timeline

1. **Week 1-2**: Dataset download and verification
2. **Week 3-4**: Preprocessing pipeline implementation
3. **Week 5-6**: Model architecture implementation
4. **Week 7-10**: Stage 1 training (large corpus)
5. **Week 11-12**: Stage 2 fine-tuning (full songs)
6. **Week 13-14**: Evaluation and iteration

## Directory Structure

```
python_training/
├── datasets/
│   ├── downloaders/
│   │   ├── musiccaps_downloader.py
│   │   ├── audioset_downloader.py
│   │   ├── fma_downloader.py
│   │   ├── jamendo_downloader.py
│   │   ├── musicbench_downloader.py
│   │   └── fullsong_downloaders.py
│   ├── processors/
│   │   ├── audio_processor.py
│   │   ├── text_processor.py
│   │   └── dataset_builder.py
│   └── verification/
│       ├── verify_downloads.py
│       └── dataset_statistics.py
├── models/
│   ├── text_embedders/
│   │   ├── muq_text_embedder.py
│   │   ├── mert_text_embedder.py
│   │   └── music2latent_text_embedder.py
│   └── base_text_embedder.py
├── training/
│   ├── train.py
│   ├── trainer.py
│   ├── losses.py
│   └── evaluation.py
├── preprocessing/
│   ├── audio_embedding_generator.py
│   └── text_tokenizer.py
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   └── visualization.py
└── configs/
    ├── muq_config.yaml
    ├── mert_config.yaml
    └── music2latent_config.yaml
```

## Notes

- All audio embedding generation should use the existing models from `python_services/embedding_models.py`
- Text embedders should be saved in formats compatible with the existing inference pipeline
- Consider memory-mapped datasets for large-scale training
- Implement checkpointing every N steps for long training runs
- Use mixed precision (FP16/BF16) to speed up training
- Consider curriculum learning: start with easier examples (shorter clips) before full songs
