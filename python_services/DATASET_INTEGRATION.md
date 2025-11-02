# Dataset Integration - Implementation Summary

## What Has Been Implemented ✅

### 1. Download Infrastructure (`dataset_downloaders/`)

**Base Classes:**
- `base.py` - Provides:
  - Checkpoint/resume functionality
  - Progress tracking with tqdm
  - Atomic file writes
  - Disk space checking
  - Download state management
  - File verification

**Implemented Downloaders:**
- ✅ `song_describer.py` - Full implementation with Zenodo + YouTube audio
- ✅ `jamendo_max_caps.py` - Full implementation with HuggingFace datasets
- ✅ `fma.py` - Full implementation with direct download + unzip
- ⚠️  `mtg_jamendo.py` - Stub (manual download instructions)
- ⚠️  `dali.py` - Stub (manual download instructions)
- ⚠️  `clotho.py` - Stub (manual download instructions)

### 2. Main Download Script (`download_datasets.py`)

**Features:**
- CLI interface for downloading any combination of datasets
- `--all` flag to download all ready datasets
- `--list` to show available datasets
- `--resume` for continuing interrupted downloads
- `--dry-run` for testing without downloading
- `--fma-size` for choosing FMA dataset size
- Comprehensive logging and summary reports

**Usage Examples:**
```bash
# List available datasets
python download_datasets.py --list

# Download Song Describer + JamendoMaxCaps
python download_datasets.py --datasets song_describer jamendo_max_caps

# Download all ready datasets
python download_datasets.py --all

# Download FMA (small version)
python download_datasets.py --datasets fma --fma-size small

# Dry run to see what would be downloaded
python download_datasets.py --all --dry-run
```

## What Needs To Be Implemented 🚧

### 3. Dataset Unification (`unify_datasets.py`) - TO BE CREATED

**Purpose:** Convert all datasets to a common JSON format for prepare_dataset.py

**Unified Format Structure:**
```json
{
  "id": "unique_id",
  "dataset": "song_describer|jamendo|fma",
  "audio_path": "/absolute/path/to/audio.mp3",
  "caption": "Main description",
  "alt_captions": ["alt1", "alt2"],
  "duration": 120.5,
  "tags": ["rock", "guitar"],
  "artist": "Artist Name",
  "title": "Song Title",
  "license": "CC-BY-4.0"
}
```

**Implementation Needed:**

**unify_datasets.py**:
```python
#!/usr/bin/env python3
"""
Unify downloaded datasets into common format.

Usage:
    # Unify all downloaded datasets
    python unify_datasets.py --output data/unified_dataset.json

    # Unify specific datasets
    python unify_datasets.py --datasets song_describer fma

    # Sample 10K tracks total
    python unify_datasets.py --max-samples 10000

    # Stratified sampling (balanced across datasets)
    python unify_datasets.py --max-samples 50000 --stratify
```

**Core Classes:**

```python
class DatasetUnifier:
    def unify_song_describer(self, raw_dir: Path) -> List[Dict]:
        """Convert Song Describer to unified format"""
        # Read metadata.json
        # For each track:
        #   - Create entry with 5 human captions
        #   - Use first caption as main, others as alt_captions
        pass

    def unify_jamendo_max_caps(self, raw_dir: Path) -> List[Dict]:
        """Convert JamendoMaxCaps to unified format"""
        # Read metadata JSON files
        # For each track:
        #   - Use synthetic caption as main
        #   - No alt captions
        pass

    def unify_fma(self, raw_dir: Path) -> List[Dict]:
        """Convert FMA to unified format"""
        # Read tracks.csv
        # For each track:
        #   - Synthesize caption from genre + tags
        #   - E.g., "A {genre} song with {tags}"
        pass

    def apply_sampling(
        self,
        unified_data: List[Dict],
        max_samples: int,
        stratify: bool = False
    ) -> List[Dict]:
        """Apply sampling strategy"""
        if stratify:
            # Balance samples across datasets
            pass
        else:
            # Random sampling
            pass
```

### 4. Modify `prepare_dataset.py` - NEEDS UPDATES

**Changes Required:**

1. **Replace hardcoded MusicBench** with unified dataset loader:

```python
class UnifiedDatasetLoader:
    """Load unified JSON dataset"""

    def __init__(self, dataset_file: str, sample_size: Optional[int] = None):
        with open(dataset_file, 'r') as f:
            self.data = json.load(f)

        if sample_size:
            self.data = random.sample(self.data, min(sample_size, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

2. **Add new arguments**:

```python
parser.add_argument('--dataset-file', type=str,
                   default='data/unified_dataset.json',
                   help='Path to unified dataset JSON')
parser.add_argument('--sample-size', type=int, default=None,
                   help='Limit total samples processed')
parser.add_argument('--min-duration', type=float, default=5.0,
                   help='Minimum audio duration (seconds)')
parser.add_argument('--max-duration', type=float, default=600.0,
                   help='Maximum audio duration (seconds)')
```

3. **Update MusicBenchProcessor.load_and_split_dataset()**:

```python
def load_and_split_dataset(self):
    """Load unified dataset and create splits"""
    logger.info("Loading unified dataset...")

    # Load JSON
    dataset_loader = UnifiedDatasetLoader(
        self.config.dataset_file,
        sample_size=self.config.sample_size
    )

    # Filter by duration
    filtered = [
        item for item in dataset_loader.data
        if (self.config.min_duration <= item['duration'] <= self.config.max_duration)
    ]

    logger.info(f"Loaded {len(filtered)} tracks after filtering")

    # Create HuggingFace Dataset-like object
    # ... existing split logic
```

### 5. Requirements.txt Updates

**Add:**
```
yt-dlp>=2024.1.0
datasets>=2.14.0
soundfile>=0.12.1
```

## Next Steps - Quick Implementation Guide

### Step 1: Create unify_datasets.py (Priority: HIGH)

1. Create basic script structure
2. Implement Song Describer unifier (human captions)
3. Implement JamendoMaxCaps unifier (synthetic captions)
4. Implement FMA unifier (tag → caption synthesis)
5. Add sampling functionality
6. Test with small samples

### Step 2: Modify prepare_dataset.py (Priority: HIGH)

1. Add UnifiedDatasetLoader class
2. Add new CLI arguments
3. Replace MusicBench loader
4. Test with unified dataset
5. Ensure checkpoint/resume still works

### Step 3: End-to-End Testing (Priority: MEDIUM)

1. Download small dataset (Song Describer)
2. Unify to JSON
3. Prepare embeddings with --sample-size 100
4. Train with sample
5. Verify entire pipeline works

### Step 4: Documentation (Priority: MEDIUM)

1. Update main README
2. Add usage examples
3. Document caption synthesis strategies
4. Add troubleshooting guide

## Testing Checklist

- [ ] Download Song Describer dataset
- [ ] Download JamendoMaxCaps (or small subset)
- [ ] Download FMA small
- [ ] Unify all three datasets
- [ ] Sample 1,000 tracks
- [ ] Prepare embeddings
- [ ] Train for 1 epoch
- [ ] Verify no errors

## Caption Synthesis Examples

**FMA** (tags → caption):
```python
def synthesize_fma_caption(track):
    genre = track['genre']
    tags = track.get('tags', [])

    if tags:
        return f"A {genre} song featuring {', '.join(tags[:3])}"
    else:
        return f"A {genre} musical piece"
```

**MTG-Jamendo** (tags → caption):
```python
def synthesize_mtg_caption(track):
    tags = track['tags']
    mood = [t for t in tags if t in MOOD_TAGS]
    instruments = [t for t in tags if t in INSTRUMENT_TAGS]

    parts = []
    if mood:
        parts.append(f"a {mood[0]} piece")
    if instruments:
        parts.append(f"featuring {', '.join(instruments)}")

    return " ".join(parts) if parts else "a musical composition"
```

## File Organization After Full Implementation

```
python_services/
├── dataset_downloaders/
│   ├── __init__.py
│   ├── base.py
│   ├── song_describer.py
│   ├── jamendo_max_caps.py
│   ├── fma.py
│   ├── mtg_jamendo.py
│   ├── dali.py
│   └── clotho.py
├── download_datasets.py
├── unify_datasets.py          # TO BE CREATED
├── prepare_dataset.py          # TO BE MODIFIED
├── train.py
└── data/
    ├── raw/
    │   ├── song_describer/
    │   ├── jamendo_max_caps/
    │   └── fma/
    ├── unified_dataset.json   # Created by unify_datasets.py
    └── processed/
        └── embeddings.h5      # Created by prepare_dataset.py
```

## Current Status Summary

✅ **Completed:**
- Download infrastructure (base classes)
- Song Describer downloader
- JamendoMaxCaps downloader
- FMA downloader
- Main download script with CLI
- Comprehensive logging and checkpointing

🚧 **In Progress:**
- Dataset unification script
- prepare_dataset.py modifications

⏳ **Pending:**
- MTG-Jamendo, DALI, Clotho downloaders (lower priority)
- End-to-end testing
- Documentation

## Estimated Completion Time

- **unify_datasets.py**: 2-3 hours
- **prepare_dataset.py modifications**: 1-2 hours
- **Testing**: 1-2 hours
- **Documentation**: 1 hour

**Total**: ~6-8 hours for full implementation

## Notes

1. The stubs for MTG-Jamendo, DALI, and Clotho can be implemented later
2. Focus on Song Describer + JamendoMaxCaps + FMA first (covers most use cases)
3. The checkpoint/resume systems in both download and prepare are compatible
4. Can test with dry-run mode extensively before actual downloads
