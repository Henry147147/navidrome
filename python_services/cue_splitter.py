"""
Utilities for splitting audio files into per-track FLACs using a CUE sheet.
"""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torchaudio
from cueparser import CueSheet

LOGGER = logging.getLogger("navidrome.cue_splitter")


_ILLEGAL_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\0]+')


def _normalize_component(value: str) -> str:
    normalized = value.replace("•", "&")
    normalized = normalized.replace("/", "_")
    normalized = normalized.replace("\\", "_")
    normalized = " ".join(normalized.split())
    return normalized.strip()


@dataclass(frozen=True)
class SplitTrack:
    """
    Describes the result of splitting a single track from a cuesheet.
    """

    index: int
    title: str
    artist: str
    album: str
    album_artist: str
    file_path: Path
    start_seconds: float
    duration_seconds: float

    def canonical_name(self) -> str:
        artist = _normalize_component(self.artist or "")
        title = _normalize_component(self.title or "")
        if artist and title:
            return f"{artist} - {title}"
        if title:
            return title
        if artist:
            return artist
        return f"Track {self.index:02d}"

    def destination_name(self) -> str:
        return self.file_path.name

    def to_response(self) -> Dict[str, str | int | float]:
        return {
            "path": str(self.file_path),
            "destName": self.destination_name(),
            "index": self.index,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "albumArtist": self.album_artist,
            "durationSeconds": self.duration_seconds,
        }


def _sanitize_filename(title: str, index: int) -> str:
    normalized = title.strip() or f"Track {index:02d}"
    normalized = _ILLEGAL_FILENAME_CHARS.sub("_", normalized)
    normalized = normalized.strip(". ")
    if not normalized:
        normalized = f"Track {index:02d}"
    return f"{index:02d} - {normalized}.flac"


def _read_cuesheet_text(cue_path: Path) -> str:
    try:
        raw_text = cue_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        LOGGER.debug("Failed to decode %s as UTF-8, retrying with cp1252", cue_path)
        raw_text = cue_path.read_text(encoding="cp1252", errors="ignore")
    normalized = raw_text.lstrip("\ufeff")
    return normalized


def _build_track_context(lines: Sequence[str]) -> List[Dict[str, Optional[str]]]:
    track_contexts: List[Dict[str, Optional[str]]] = []
    current_file: Optional[str] = None
    current_track: Optional[Dict[str, Optional[str]]] = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        try:
            parts = shlex.split(line, comments=False, posix=True)
        except ValueError:
            continue
        if not parts:
            continue
        keyword = parts[0].upper()
        if keyword == "FILE" and len(parts) >= 2:
            current_file = Path(parts[1]).name.lower()
        elif keyword == "TRACK" and len(parts) >= 2:
            try:
                track_number = int(parts[1])
            except ValueError:
                track_number = None
            current_track = {
                "number": track_number,
                "file": current_file,
                "title": None,
                "performer": None,
            }
            track_contexts.append(current_track)
        elif (
            keyword == "TITLE"
            and len(parts) >= 2
            and current_track is not None
            and current_track.get("title") is None
        ):
            current_track["title"] = parts[1]
        elif (
            keyword == "PERFORMER"
            and len(parts) >= 2
            and current_track is not None
            and current_track.get("performer") is None
        ):
            current_track["performer"] = parts[1]
    return track_contexts


def _cue_time_to_seconds(time_str: str) -> float:
    minute, second, frame = time_str.split(":")
    return int(minute) * 60 + int(second) + int(frame) / 75.0


def _load_audio_metadata(music_path: Path) -> tuple[int, int]:
    """
    Returns (sample_rate, total_frames)
    """
    info = torchaudio.info(str(music_path))
    sample_rate = info.sample_rate
    total_frames = info.num_frames
    if sample_rate <= 0 or total_frames <= 0:
        raise ValueError("Invalid audio metadata")
    return sample_rate, total_frames


def _load_audio_segment(
    music_path: Path,
    sample_rate: int,
    start_seconds: float,
    duration_seconds: float | None,
) -> torch.Tensor:
    frame_offset = max(int(round(start_seconds * sample_rate)), 0)
    if duration_seconds is None:
        num_frames = -1
    else:
        num_frames = max(int(round(duration_seconds * sample_rate)), 0)
    waveform, sr = torchaudio.load(
        str(music_path),
        frame_offset=frame_offset,
        num_frames=None if num_frames < 0 else num_frames,
    )
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform


def split_flac_with_cue(
    music_file: str,
    cue_file: str,
    *,
    output_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> List[SplitTrack]:
    """
    Split the provided music file into individual FLAC tracks described by the cuesheet.
    Returns a list of SplitTrack objects describing the generated files.
    """

    log = logger or LOGGER
    music_path = Path(music_file)
    cue_path = Path(cue_file)
    if not cue_path.exists():
        log.warning("Cue file does not exist: %s", cue_file)
        return []

    output_root = Path(output_dir) if output_dir is not None else music_path.parent
    output_root.mkdir(parents=True, exist_ok=True)

    cuesheet_text = _read_cuesheet_text(cue_path)
    lines = cuesheet_text.splitlines()
    sheet = CueSheet()
    sheet.setOutputFormat("", "%title%")
    sheet.setData(cuesheet_text)
    try:
        sheet.parse()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Failed to parse cuesheet %s: %s", cue_file, exc)
        return []

    track_contexts = _build_track_context(lines)
    candidate_name = music_path.name.lower()
    candidate_stem = music_path.stem.lower()

    try:
        sample_rate, total_frames = _load_audio_metadata(music_path)
    except Exception as exc:
        log.error("Unable to load audio metadata for %s: %s", music_file, exc)
        return []

    track_count = len(sheet.tracks)
    total_duration = total_frames / sample_rate
    album_title = (sheet.title or "").strip()
    album_artist = (sheet.performer or "").strip()

    results: List[SplitTrack] = []
    for idx, track in enumerate(sheet.tracks):
        context = track_contexts[idx] if idx < len(track_contexts) else {}
        context_file = (context or {}).get("file")
        if context_file:
            context_name = Path(str(context_file)).name.lower()
            context_stem = Path(str(context_file)).stem.lower()
            if context_name not in {candidate_name, candidate_stem} and context_stem not in {
                candidate_name,
                candidate_stem,
            }:
                log.debug(
                    "Skipping track %s due to mismatched FILE stanza (%s)",
                    track.number,
                    context_file,
                )
                continue

        if not track.offset:
            log.debug("Skipping track %s without INDEX 01 offset", track.number)
            continue

        try:
            start_seconds = _cue_time_to_seconds(track.offset)
        except ValueError:
            log.debug("Skipping track %s due to invalid offset %s", track.number, track.offset)
            continue

        if idx + 1 < track_count:
            next_track = sheet.tracks[idx + 1]
            if next_track.offset:
                try:
                    end_seconds = _cue_time_to_seconds(next_track.offset)
                except ValueError:
                    end_seconds = None
            else:
                end_seconds = None
        else:
            end_seconds = total_duration

        duration_seconds: Optional[float]
        if end_seconds is None:
            duration_seconds = None
        else:
            duration_seconds = max(end_seconds - start_seconds, 0.0)

        waveform = _load_audio_segment(
            music_path,
            sample_rate,
            start_seconds,
            duration_seconds,
        )
        if duration_seconds is None:
            duration_seconds = waveform.shape[-1] / sample_rate
        title = (track.title or context.get("title") or f"Track {track.number:02d}").strip()
        performer = (track.performer or "").strip()
        if not performer:
            performer = (context.get("performer") or "").strip()
        if not performer:
            performer = album_artist

        dest_name = _sanitize_filename(title, track.number or (idx + 1))
        track_path = output_root / dest_name
        try:
            torchaudio.save(str(track_path), waveform, sample_rate, format="FLAC")
        except Exception as exc:
            log.error("Failed to write split track %s: %s", dest_name, exc)
            continue

        try:
            from mutagen.flac import FLAC  # local import to avoid module cost if unused

            tags = FLAC(str(track_path))
            tags["title"] = [title]
            if performer:
                tags["artist"] = [performer]
                if album_artist:
                    tags["albumartist"] = [album_artist]
            if album_title:
                tags["album"] = [album_title]
            tags["tracknumber"] = [str(track.number or (idx + 1))]
            tags.save()
        except Exception as exc:
            log.debug("Unable to save metadata for %s: %s", track_path, exc)

        results.append(
            SplitTrack(
                index=track.number or (idx + 1),
                title=title,
                artist=performer,
                album=album_title,
                album_artist=album_artist,
                file_path=track_path,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
            )
        )

    return results


__all__ = ["SplitTrack", "split_flac_with_cue"]
