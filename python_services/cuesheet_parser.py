"""
Utilities for parsing CUE sheet files used by the embedding pipeline.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Iterable, List, Optional, Set

from cueparser import CueSheet

from models import TrackSegment

logger = logging.getLogger("navidrome.cuesheet")


def cue_time_to_seconds(time_str: str) -> float:
    """
    Convert a CUE time string (MM:SS:FF) into absolute seconds.
    """
    minute, second, frame = time_str.split(":")
    return int(minute) * 60 + int(second) + int(frame) / 75.0


def parse_cuesheet_tracks(
    cue_path: Path,
    music_file: str,
    *,
    candidate_names: Optional[Iterable[str]] = None,
) -> List[TrackSegment]:
    """
    Parse a cuesheet looking for track boundaries that correspond to the target file.

    Returns a list of TrackSegment entries ordered by their start time.
    """
    target_path = Path(music_file)
    target_name = target_path.name.lower()
    target_stem = target_path.stem.lower()
    candidate_name_set: Set[str] = {target_name}
    candidate_stem_set: Set[str] = {target_stem}
    if candidate_names:
        for name in candidate_names:
            if not name:
                continue
            candidate_name_set.add(Path(name).name.lower())
            candidate_stem_set.add(Path(name).stem.lower())
    logger.debug(
        "Parsing cuesheet %s for target file %s (candidates=%s)",
        cue_path,
        target_name,
        sorted(candidate_name_set),
    )
    try:
        raw_text = cue_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.debug("Failed to decode %s as UTF-8, retrying with cp1252", cue_path)
        raw_text = cue_path.read_text(encoding="cp1252", errors="ignore")
    normalized = raw_text.lstrip("\ufeff")
    if normalized != raw_text:
        logger.debug("Removed UTF-8 BOM from cuesheet %s", cue_path)
    lines = normalized.splitlines()
    removed_leading = 0
    while lines and not lines[0].strip():
        lines.pop(0)
        removed_leading += 1
    if removed_leading:
        logger.debug(
            "Stripped %d leading blank lines from cuesheet %s",
            removed_leading,
            cue_path,
        )
    if not lines:
        logger.debug("Cuesheet %s contains no data after normalization", cue_path)
        return []
    cuesheet = "\n".join(lines)
    sheet = CueSheet()
    sheet.setOutputFormat("", "%title%")
    sheet.setData(cuesheet)
    try:
        sheet.parse()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to parse cuesheet %s: %s", cue_path, exc)
        return []
    logger.debug(
        "cueparser extracted %d tracks for %s (sheet file=%s)",
        len(sheet.tracks),
        target_name,
        (sheet.file or "").lower(),
    )

    track_contexts: List[dict] = []
    current_file: Optional[str] = None
    current_track: Optional[dict] = None
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
            current_track = {"number": track_number, "file": current_file}
            track_contexts.append(current_track)
        elif (
            keyword == "TITLE"
            and len(parts) >= 2
            and current_track is not None
            and current_track.get("title") is None
        ):
            current_track["title"] = parts[1]

    if len(track_contexts) != len(sheet.tracks):
        logger.debug(
            "Cuesheet %s track count mismatch: parsed %d tracks, context %d",
            cue_path,
            len(sheet.tracks),
            len(track_contexts),
        )

    collected: List[dict] = []
    for track, context in zip(sheet.tracks, track_contexts):
        context_file = context.get("file")
        if context_file:
            context_name = Path(context_file).name.lower()
            context_stem = Path(context_file).stem.lower()
            if (
                context_name not in candidate_name_set
                and context_stem not in candidate_stem_set
            ):
                logger.debug(
                    "Skipping track %s due to file mismatch (context=%s, targets=%s)",
                    context.get("number") or track.number,
                    context_file,
                    sorted(candidate_name_set),
                )
                continue
        if not track.offset:
            logger.debug(
                "Skipping track %s because no offset present in cuesheet",
                context.get("number") or track.number,
            )
            continue
        try:
            start = cue_time_to_seconds(track.offset)
        except ValueError:
            logger.debug(
                "Skipping track %s due to invalid offset format: %s",
                context.get("number") or track.number,
                track.offset,
            )
            continue
        title = track.title or context.get("title")
        context_number = context.get("number")
        number = context_number if context_number is not None else track.number
        collected.append(
            {
                "number": int(number) if number is not None else track.number,
                "title": title,
                "start": start,
            }
        )

    collected.sort(key=lambda entry: entry["start"])
    result: List[TrackSegment] = []
    for idx, entry in enumerate(collected):
        end = collected[idx + 1]["start"] if idx + 1 < len(collected) else None
        title = entry["title"] or f"Track {entry['number']:02d}"
        result.append(
            TrackSegment(
                index=entry["number"],
                title=title,
                start=entry["start"],
                end=end,
            )
        )
    logger.debug("Parsed %d cue tracks for %s", len(result), target_name)
    return result


__all__ = [
    "cue_time_to_seconds",
    "parse_cuesheet_tracks",
]
