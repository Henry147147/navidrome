#!/usr/bin/env python3
"""
unify_datasets.py - Unified Music Dataset Converter

Converts downloaded music datasets into a unified JSON format for training.

Supported datasets:
- Song Describer: Human captions
- JamendoMaxCaps: Synthetic captions
- FMA: Tags → synthesized captions
- MTG-Jamendo: Tags → synthesized captions (stub)
- DALI: Lyrics as captions (stub)
- Clotho: Audio captions (stub)

Usage:
    # Unify all downloaded datasets
    python unify_datasets.py --output data/unified_dataset.json

    # Unify specific datasets
    python unify_datasets.py --datasets song_describer fma

    # Sample 10K tracks with stratified sampling
    python unify_datasets.py --max-samples 10000 --stratify-by dataset

    # Filter by duration
    python unify_datasets.py --min-duration 10 --max-duration 300
"""

import os
import sys
import json
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unify_datasets.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class UnifiedTrack:
    """Unified track representation"""
    id: str
    audio_path: str
    caption: str
    alt_captions: List[str]
    duration: float
    dataset: str
    artist: Optional[str] = None
    title: Optional[str] = None
    tags: Optional[List[str]] = None
    license: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        d = {
            'id': self.id,
            'audio_path': self.audio_path,
            'caption': self.caption,
            'alt_captions': self.alt_captions,
            'duration': self.duration,
            'dataset': self.dataset
        }
        if self.artist:
            d['artist'] = self.artist
        if self.title:
            d['title'] = self.title
        if self.tags:
            d['tags'] = self.tags
        if self.license:
            d['license'] = self.license
        return d


# ============================================================================
# Caption Synthesis Functions
# ============================================================================

def synthesize_caption_from_tags(
    genre: Optional[str] = None,
    tags: Optional[List[str]] = None,
    mood: Optional[str] = None,
    instruments: Optional[List[str]] = None
) -> str:
    """
    Synthesize natural language caption from tags.

    Args:
        genre: Musical genre
        tags: List of tags
        mood: Mood/atmosphere descriptor
        instruments: List of instruments

    Returns:
        Natural language caption
    """
    # Parse tags into categories if provided
    if tags and not (mood or instruments):
        mood_tags = {'upbeat', 'energetic', 'calm', 'dark', 'happy', 'sad', 'melancholic', 'aggressive'}
        instrument_tags = {'guitar', 'piano', 'drums', 'bass', 'synthesizer', 'violin', 'saxophone'}

        mood = next((t for t in tags if t.lower() in mood_tags), None)
        instruments = [t for t in tags if t.lower() in instrument_tags]

    # Build caption from available information
    parts = []

    if mood and genre:
        parts.append(f"A {mood} {genre} track")
    elif genre:
        parts.append(f"A {genre} song")
    elif mood:
        parts.append(f"A {mood} musical piece")
    else:
        parts.append("A musical composition")

    if instruments:
        if len(instruments) == 1:
            parts.append(f"featuring {instruments[0]}")
        elif len(instruments) == 2:
            parts.append(f"featuring {instruments[0]} and {instruments[1]}")
        else:
            parts.append(f"featuring {', '.join(instruments[:2])} and more")

    if not instruments and tags and len(tags) > 0:
        # Use first few tags as descriptors
        desc_tags = [t for t in tags if t.lower() not in {'instrumental', 'vocal'}][:2]
        if desc_tags:
            parts.append(f"with {' and '.join(desc_tags)} elements")

    caption = " ".join(parts)

    # Capitalize first letter
    if caption:
        caption = caption[0].upper() + caption[1:]

    return caption if caption else "A musical piece"


def synthesize_fma_caption(genre: str, tags: List[str]) -> str:
    """
    Synthesize caption specifically for FMA dataset.

    Args:
        genre: Musical genre
        tags: List of descriptive tags

    Returns:
        Natural language caption
    """
    if not tags:
        return f"A {genre} musical piece"

    # FMA-specific tag categorization
    moods = []
    instruments = []
    characteristics = []

    mood_keywords = {'upbeat', 'energetic', 'calm', 'mellow', 'dark', 'happy', 'sad', 'aggressive', 'peaceful'}
    instrument_keywords = {'guitar', 'piano', 'drums', 'bass', 'synth', 'strings', 'brass', 'vocals'}

    for tag in tags:
        tag_lower = tag.lower()
        if any(mood in tag_lower for mood in mood_keywords):
            moods.append(tag)
        elif any(inst in tag_lower for inst in instrument_keywords):
            instruments.append(tag)
        else:
            characteristics.append(tag)

    # Build caption
    if moods and instruments:
        mood_str = moods[0] if len(moods) == 1 else f"{moods[0]} and {moods[1]}"
        inst_str = instruments[0] if len(instruments) == 1 else ', '.join(instruments[:2])
        return f"A {mood_str} {genre} track featuring {inst_str}"
    elif instruments:
        inst_str = instruments[0] if len(instruments) == 1 else ', '.join(instruments[:3])
        return f"A {genre} song with {inst_str}"
    elif moods:
        return f"A {moods[0]} {genre} piece"
    elif characteristics:
        return f"A {genre} track with {characteristics[0]} style"
    else:
        return f"A {genre} musical composition"


# ============================================================================
# Dataset Parsers
# ============================================================================

class BaseParser:
    """Base class for dataset parsers"""

    def __init__(self, raw_dir: Path, data_root: Path):
        """
        Initialize parser.

        Args:
            raw_dir: Directory with raw dataset files
            data_root: Root data directory for resolving relative paths
        """
        self.raw_dir = raw_dir
        self.data_root = data_root

    def parse(self) -> List[UnifiedTrack]:
        """Parse dataset and return list of unified tracks"""
        raise NotImplementedError

    def resolve_audio_path(self, relative_path: str) -> str:
        """Convert relative path to absolute path"""
        if os.path.isabs(relative_path):
            return relative_path

        # Try relative to raw_dir first
        abs_path = self.raw_dir / relative_path
        if abs_path.exists():
            return str(abs_path.resolve())

        # Try relative to data_root
        abs_path = self.data_root / relative_path
        if abs_path.exists():
            return str(abs_path.resolve())

        # Return as-is (will fail validation later)
        return str(Path(relative_path).resolve())

    def get_audio_duration(self, audio_path: str) -> Optional[float]:
        """Get audio file duration in seconds"""
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.duration
        except Exception as e:
            logger.warning(f"Failed to get duration for {audio_path}: {e}")
            return None


class SongDescriberParser(BaseParser):
    """Parser for Song Describer dataset"""

    def parse(self) -> List[UnifiedTrack]:
        """Parse Song Describer metadata"""
        logger.info("Parsing Song Describer dataset...")

        metadata_file = self.raw_dir / "metadata.json"
        if not metadata_file.exists():
            logger.error(f"Metadata not found: {metadata_file}")
            return []

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        tracks = []
        audio_dir = self.raw_dir / "audio"

        # Handle both list and dict formats
        items = metadata if isinstance(metadata, list) else metadata.get('tracks', [])

        for item in tqdm(items, desc="Song Describer"):
            youtube_id = item.get('youtube_id') or item.get('ytid') or item.get('id')
            if not youtube_id:
                continue

            audio_path = audio_dir / f"{youtube_id}.mp3"
            if not audio_path.exists():
                logger.debug(f"Audio file not found: {audio_path}")
                continue

            # Get captions (Song Describer has 5 human captions)
            captions = item.get('captions', [])
            if isinstance(captions, str):
                captions = [captions]

            if not captions:
                # Fallback to description field
                desc = item.get('description') or item.get('caption')
                captions = [desc] if desc else []

            if not captions:
                logger.warning(f"No captions for {youtube_id}")
                continue

            # Use first caption as main, rest as alternatives
            main_caption = captions[0]
            alt_captions = captions[1:] if len(captions) > 1 else []

            # Get duration
            duration = item.get('duration')
            if not duration:
                duration = self.get_audio_duration(str(audio_path))

            track = UnifiedTrack(
                id=f"song_describer_{youtube_id}",
                audio_path=str(audio_path.resolve()),
                caption=main_caption,
                alt_captions=alt_captions,
                duration=duration or 120.0,  # Default ~2 min
                dataset="song_describer",
                artist=item.get('artist'),
                title=item.get('title'),
                license="YouTube"
            )
            tracks.append(track)

        logger.info(f"Parsed {len(tracks)} tracks from Song Describer")
        return tracks


class JamendoMaxCapsParser(BaseParser):
    """Parser for JamendoMaxCaps dataset"""

    def parse(self) -> List[UnifiedTrack]:
        """Parse JamendoMaxCaps metadata"""
        logger.info("Parsing JamendoMaxCaps dataset...")

        metadata_dir = self.raw_dir / "metadata"
        if not metadata_dir.exists():
            logger.error(f"Metadata directory not found: {metadata_dir}")
            return []

        tracks = []

        # Process each split (train, validation, test)
        for split_file in metadata_dir.glob("*.json"):
            split_name = split_file.stem

            with open(split_file, 'r') as f:
                split_data = json.load(f)

            for item in tqdm(split_data, desc=f"JamendoMaxCaps ({split_name})"):
                audio_path = item.get('audio_path')
                if not audio_path:
                    continue

                # Resolve path
                abs_audio_path = self.resolve_audio_path(audio_path)
                if not os.path.exists(abs_audio_path):
                    logger.debug(f"Audio file not found: {abs_audio_path}")
                    continue

                caption = item.get('caption', '')
                if not caption:
                    continue

                track_id = item.get('id', f"jamendo_{split_name}_{len(tracks)}")

                track = UnifiedTrack(
                    id=f"jamendo_{track_id}",
                    audio_path=abs_audio_path,
                    caption=caption,
                    alt_captions=[],  # Jamendo has single captions
                    duration=item.get('duration', 180.0),
                    dataset="jamendo_max_caps",
                    artist=item.get('artist'),
                    title=item.get('title'),
                    tags=item.get('tags', []),
                    license=item.get('license', 'CC-BY-SA 4.0')
                )
                tracks.append(track)

        logger.info(f"Parsed {len(tracks)} tracks from JamendoMaxCaps")
        return tracks


class FMAParser(BaseParser):
    """Parser for Free Music Archive dataset"""

    def parse(self) -> List[UnifiedTrack]:
        """Parse FMA metadata and synthesize captions"""
        logger.info("Parsing FMA dataset...")

        metadata_dir = self.raw_dir / "fma_metadata"
        tracks_csv = metadata_dir / "tracks.csv"

        if not tracks_csv.exists():
            logger.error(f"FMA metadata not found: {tracks_csv}")
            return []

        # Load tracks metadata
        # FMA uses multi-index CSV, need to handle carefully
        tracks_df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

        # Find audio files
        audio_dirs = [
            self.raw_dir / "fma_small",
            self.raw_dir / "fma_medium",
            self.raw_dir / "fma_large",
            self.raw_dir / "fma_full"
        ]

        # Build track ID to path mapping
        track_id_to_path = {}
        for audio_dir in audio_dirs:
            if not audio_dir.exists():
                continue

            for audio_file in audio_dir.rglob("*.mp3"):
                # FMA uses 6-digit zero-padded IDs
                track_id = int(audio_file.stem)
                track_id_to_path[track_id] = audio_file

        tracks = []

        for track_id, row in tqdm(tracks_df.iterrows(), desc="FMA", total=len(tracks_df)):
            if track_id not in track_id_to_path:
                continue

            audio_path = track_id_to_path[track_id]

            # Extract metadata (FMA has multi-level columns)
            try:
                # Genre (top-level genre)
                genre = row[('track', 'genre_top')] if ('track', 'genre_top') in row.index else "Unknown"

                # Tags (if available)
                tags = []
                if ('track', 'tags') in row.index:
                    tags_str = row[('track', 'tags')]
                    if pd.notna(tags_str):
                        tags = [t.strip() for t in str(tags_str).split(',')]

                # Artist and title
                artist = row[('artist', 'name')] if ('artist', 'name') in row.index else None
                title = row[('track', 'title')] if ('track', 'title') in row.index else None

                # Duration
                duration = row[('track', 'duration')] if ('track', 'duration') in row.index else None
                if pd.isna(duration):
                    duration = self.get_audio_duration(str(audio_path))

                # Synthesize caption from genre and tags
                caption = synthesize_fma_caption(genre, tags)

                track = UnifiedTrack(
                    id=f"fma_{track_id:06d}",
                    audio_path=str(audio_path.resolve()),
                    caption=caption,
                    alt_captions=[],
                    duration=duration or 180.0,
                    dataset="fma",
                    artist=str(artist) if pd.notna(artist) else None,
                    title=str(title) if pd.notna(title) else None,
                    tags=[genre] + tags if tags else [genre],
                    license="CC-BY"
                )
                tracks.append(track)

            except Exception as e:
                logger.warning(f"Failed to parse FMA track {track_id}: {e}")
                continue

        logger.info(f"Parsed {len(tracks)} tracks from FMA")
        return tracks


# ============================================================================
# Sampling Strategies
# ============================================================================

def sample_random(tracks: List[UnifiedTrack], max_samples: int, seed: int = 42) -> List[UnifiedTrack]:
    """Random sampling"""
    random.seed(seed)
    return random.sample(tracks, min(max_samples, len(tracks)))


def sample_stratified_by_dataset(
    tracks: List[UnifiedTrack],
    max_samples: int,
    seed: int = 42
) -> List[UnifiedTrack]:
    """Stratified sampling to balance across datasets"""
    random.seed(seed)

    # Group by dataset
    by_dataset = defaultdict(list)
    for track in tracks:
        by_dataset[track.dataset].append(track)

    # Calculate samples per dataset
    num_datasets = len(by_dataset)
    samples_per_dataset = max_samples // num_datasets

    # Sample from each dataset
    sampled = []
    for dataset_name, dataset_tracks in by_dataset.items():
        n_samples = min(samples_per_dataset, len(dataset_tracks))
        sampled.extend(random.sample(dataset_tracks, n_samples))

    # If we haven't reached max_samples, add more randomly
    if len(sampled) < max_samples:
        remaining = [t for t in tracks if t not in sampled]
        n_more = min(max_samples - len(sampled), len(remaining))
        sampled.extend(random.sample(remaining, n_more))

    return sampled


def sample_stratified_by_duration(
    tracks: List[UnifiedTrack],
    max_samples: int,
    seed: int = 42
) -> List[UnifiedTrack]:
    """Stratified sampling to balance short/medium/long tracks"""
    random.seed(seed)

    # Categorize by duration
    short = [t for t in tracks if t.duration < 60]
    medium = [t for t in tracks if 60 <= t.duration < 180]
    long = [t for t in tracks if t.duration >= 180]

    # Calculate samples per category (proportional to availability)
    total = len(short) + len(medium) + len(long)
    if total == 0:
        return []

    n_short = int(max_samples * len(short) / total)
    n_medium = int(max_samples * len(medium) / total)
    n_long = max_samples - n_short - n_medium

    # Sample from each category
    sampled = []
    sampled.extend(random.sample(short, min(n_short, len(short))))
    sampled.extend(random.sample(medium, min(n_medium, len(medium))))
    sampled.extend(random.sample(long, min(n_long, len(long))))

    return sampled


# ============================================================================
# Main Unification Logic
# ============================================================================

PARSERS = {
    'song_describer': SongDescriberParser,
    'jamendo_max_caps': JamendoMaxCapsParser,
    'fma': FMAParser,
    # Stubs for future implementation
    'mtg_jamendo': None,
    'dali': None,
    'clotho': None
}


def unify_datasets(
    datasets: List[str],
    data_root: Path,
    min_duration: float = 0.0,
    max_duration: float = float('inf'),
    max_samples: Optional[int] = None,
    samples_per_dataset: Optional[int] = None,
    stratify_by: Optional[str] = None,
    seed: int = 42,
    dry_run: bool = False
) -> List[UnifiedTrack]:
    """
    Unify multiple datasets into common format.

    Args:
        datasets: List of dataset names to unify
        data_root: Root data directory
        min_duration: Minimum track duration (seconds)
        max_duration: Maximum track duration (seconds)
        max_samples: Maximum total samples
        samples_per_dataset: Maximum samples per dataset
        stratify_by: Stratification strategy ('dataset', 'duration', or None)
        seed: Random seed
        dry_run: If True, don't actually process

    Returns:
        List of unified tracks
    """
    all_tracks = []

    # Parse each dataset
    for dataset_name in datasets:
        if dataset_name not in PARSERS:
            logger.warning(f"Unknown dataset: {dataset_name}")
            continue

        parser_class = PARSERS[dataset_name]
        if parser_class is None:
            logger.warning(f"Parser for {dataset_name} not yet implemented")
            continue

        raw_dir = data_root / "raw" / dataset_name
        if not raw_dir.exists():
            logger.warning(f"Dataset directory not found: {raw_dir}")
            continue

        if dry_run:
            logger.info(f"[DRY RUN] Would parse {dataset_name}")
            continue

        # Parse dataset
        parser = parser_class(raw_dir, data_root)
        tracks = parser.parse()

        # Filter by duration
        tracks = [
            t for t in tracks
            if min_duration <= t.duration <= max_duration
        ]

        # Sample per dataset if specified
        if samples_per_dataset and len(tracks) > samples_per_dataset:
            random.seed(seed)
            tracks = random.sample(tracks, samples_per_dataset)

        logger.info(f"Added {len(tracks)} tracks from {dataset_name} (after filtering)")
        all_tracks.extend(tracks)

    if dry_run:
        logger.info(f"[DRY RUN] Would unify {len(datasets)} datasets")
        return []

    # Apply global sampling
    if max_samples and len(all_tracks) > max_samples:
        logger.info(f"Sampling {max_samples} from {len(all_tracks)} total tracks...")

        if stratify_by == 'dataset':
            all_tracks = sample_stratified_by_dataset(all_tracks, max_samples, seed)
        elif stratify_by == 'duration':
            all_tracks = sample_stratified_by_duration(all_tracks, max_samples, seed)
        else:
            all_tracks = sample_random(all_tracks, max_samples, seed)

    return all_tracks


def save_unified_dataset(tracks: List[UnifiedTrack], output_file: Path):
    """Save unified tracks to JSON file"""
    logger.info(f"Saving {len(tracks)} tracks to {output_file}...")

    # Convert to dicts
    data = [track.to_dict() for track in tracks]

    # Write JSON
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    # Print statistics
    logger.info(f"\nUnified Dataset Statistics:")
    logger.info(f"  Total tracks: {len(tracks)}")

    # By dataset
    by_dataset = defaultdict(int)
    for track in tracks:
        by_dataset[track.dataset] += 1

    logger.info(f"\n  By dataset:")
    for dataset, count in sorted(by_dataset.items()):
        logger.info(f"    {dataset}: {count}")

    # Duration statistics
    durations = [t.duration for t in tracks]
    if durations:
        logger.info(f"\n  Duration:")
        logger.info(f"    Min: {min(durations):.1f}s")
        logger.info(f"    Max: {max(durations):.1f}s")
        logger.info(f"    Mean: {sum(durations)/len(durations):.1f}s")

    # File size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"\n  File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Unify music datasets into common JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=list(PARSERS.keys()),
        help='Datasets to unify (default: all available)'
    )

    parser.add_argument(
        '--data-root',
        type=str,
        default='data',
        help='Root data directory (default: data/)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/unified_dataset.json',
        help='Output JSON file path'
    )

    parser.add_argument(
        '--min-duration',
        type=float,
        default=0.0,
        help='Minimum track duration in seconds (default: 0)'
    )

    parser.add_argument(
        '--max-duration',
        type=float,
        default=float('inf'),
        help='Maximum track duration in seconds (default: unlimited)'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum total samples to include'
    )

    parser.add_argument(
        '--samples-per-dataset',
        type=int,
        default=None,
        help='Maximum samples per dataset'
    )

    parser.add_argument(
        '--stratify-by',
        type=str,
        choices=['dataset', 'duration'],
        default=None,
        help='Stratification strategy for sampling'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually processing'
    )

    args = parser.parse_args()

    # Determine which datasets to process
    if args.datasets:
        datasets_to_process = args.datasets
    else:
        # Find all available datasets
        data_root = Path(args.data_root)
        raw_dir = data_root / "raw"
        if raw_dir.exists():
            datasets_to_process = [
                d.name for d in raw_dir.iterdir()
                if d.is_dir() and d.name in PARSERS
            ]
        else:
            logger.error(f"Data root directory not found: {raw_dir}")
            return 1

    if not datasets_to_process:
        logger.error("No datasets specified or found")
        return 1

    logger.info(f"Unifying datasets: {datasets_to_process}")

    # Unify datasets
    tracks = unify_datasets(
        datasets=datasets_to_process,
        data_root=Path(args.data_root),
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_samples=args.max_samples,
        samples_per_dataset=args.samples_per_dataset,
        stratify_by=args.stratify_by,
        seed=args.seed,
        dry_run=args.dry_run
    )

    if args.dry_run:
        logger.info("\n[DRY RUN] No files were created")
        return 0

    if not tracks:
        logger.error("No tracks found to unify")
        return 1

    # Save to JSON
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_unified_dataset(tracks, output_file)

    logger.info(f"\n✓ Successfully created unified dataset: {output_file}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
