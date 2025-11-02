"""
JamendoMaxCaps Dataset downloader.

Dataset: 28,185 tracks with synthetic captions
Source: https://huggingface.co/datasets/amaai-lab/JamendoMaxCaps
Paper: https://arxiv.org/abs/2502.07461
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .base import BaseDownloader


logger = logging.getLogger(__name__)


class JamendoMaxCapsDownloader(BaseDownloader):
    """Downloader for JamendoMaxCaps dataset"""

    DATASET_NAME = "jamendo_max_caps"
    HF_DATASET_ID = "amaai-lab/JamendoMaxCaps"

    def get_name(self) -> str:
        return self.DATASET_NAME

    def get_metadata_path(self) -> Path:
        return self.output_dir / "metadata"

    def download(self) -> bool:
        """
        Download JamendoMaxCaps dataset from HuggingFace.

        Returns:
            True if successful
        """
        logger.info(f"Starting {self.DATASET_NAME} download...")

        # Check if datasets library is available
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not found. Install with: pip install datasets")
            return False

        try:
            # Download dataset from HuggingFace
            logger.info(f"Downloading from HuggingFace: {self.HF_DATASET_ID}")
            logger.info("This may take a while...")

            if self.dry_run:
                logger.info("[DRY RUN] Would download dataset from HuggingFace")
                return True

            # Load dataset (HF handles caching automatically)
            dataset = load_dataset(
                self.HF_DATASET_ID,
                cache_dir=str(self.output_dir / "hf_cache")
            )

            logger.info(f"Dataset loaded. Splits: {list(dataset.keys())}")

            # Save metadata
            metadata_dir = self.get_metadata_path()
            metadata_dir.mkdir(exist_ok=True)

            # Export to JSON for easier access
            for split_name, split_data in dataset.items():
                logger.info(f"Processing {split_name} split ({len(split_data)} examples)...")

                metadata_file = metadata_dir / f"{split_name}.json"
                audio_dir = self.output_dir / "audio" / split_name
                audio_dir.mkdir(parents=True, exist_ok=True)

                # Save metadata
                examples = []
                for idx, example in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
                    # Extract audio if available
                    audio_path = None
                    if 'audio' in example and example['audio'] is not None:
                        # HF datasets stores audio as dict with 'path' and 'array'
                        audio_data = example['audio']

                        # Save audio file
                        audio_filename = f"{idx:06d}.mp3"
                        audio_path = audio_dir / audio_filename

                        # Write audio array to file using soundfile
                        if 'array' in audio_data and 'sampling_rate' in audio_data:
                            import soundfile as sf
                            sf.write(
                                audio_path,
                                audio_data['array'],
                                audio_data['sampling_rate']
                            )

                    # Create metadata entry
                    entry = {
                        'id': example.get('id', f'{split_name}_{idx}'),
                        'caption': example.get('caption', ''),
                        'audio_path': str(audio_path.relative_to(self.output_dir)) if audio_path else None,
                        'duration': example.get('duration'),
                        'tags': example.get('tags', []),
                        'artist': example.get('artist'),
                        'title': example.get('title'),
                        'license': example.get('license', 'CC-BY-SA 4.0')
                    }
                    examples.append(entry)

                    self.state.total_files += 1
                    if audio_path:
                        self.state.downloaded_files.add(str(audio_path.relative_to(self.output_dir)))
                        if audio_path.exists():
                            self.state.total_size_bytes += audio_path.stat().st_size

                # Save metadata JSON
                with open(metadata_file, 'w') as f:
                    json.dump(examples, f, indent=2)

                logger.info(f"Saved {len(examples)} examples to {metadata_file}")

            # Create README
            self._create_readme(dataset)

            # Print summary
            self.print_summary()

            # Clean up checkpoint
            self.delete_state()

            return True

        except Exception as e:
            logger.error(f"Failed to download {self.DATASET_NAME}: {e}", exc_info=True)
            return False

    def _create_readme(self, dataset):
        """Create README file with dataset information"""
        readme_path = self.output_dir / "README.md"

        if self.dry_run:
            return

        # Get dataset info
        total_examples = sum(len(split) for split in dataset.values())
        splits_info = "\n".join([f"- {name}: {len(split)} examples" for name, split in dataset.items()])

        readme_content = f"""# JamendoMaxCaps Dataset

## Overview
Synthetic captions for full-length music tracks from Jamendo.

- **Size**: {total_examples:,} tracks
- **Captions**: Automatically generated descriptive text
- **Duration**: Full tracks (variable length)
- **Source**: HuggingFace ({self.HF_DATASET_ID})
- **Paper**: https://arxiv.org/abs/2502.07461

## Splits
{splits_info}

## Directory Structure
```
{self.DATASET_NAME}/
├── metadata/          # Metadata JSON files
│   ├── train.json
│   ├── validation.json
│   └── test.json
├── audio/            # Audio files organized by split
│   ├── train/
│   ├── validation/
│   └── test/
├── hf_cache/         # HuggingFace cache
└── README.md         # This file
```

## Metadata Format
Each entry contains:
- `id`: Unique track ID
- `caption`: Synthetic description of the track
- `audio_path`: Relative path to audio file
- `duration`: Track duration in seconds
- `tags`: Musical tags/genres
- `artist`: Artist name
- `title`: Track title
- `license`: Creative Commons license (CC-BY-SA 4.0)

## Citation
```bibtex
@article{{jamendo_max_caps_2025,
  title={{JamendoMaxCaps: A Large-Scale Music Captioning Dataset}},
  author={{...}},
  journal={{arXiv preprint arXiv:2502.07461}},
  year={{2025}}
}}
```

## License
All tracks are licensed under Creative Commons (mostly CC-BY-SA 4.0).

## Download Info
- Downloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Downloader: dataset_downloaders/jamendo_max_caps.py
- Source: HuggingFace Datasets
"""

        with open(readme_path, 'w') as f:
            f.write(readme_content)

        logger.info(f"README created at {readme_path}")
