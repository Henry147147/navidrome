#!/usr/bin/env python3
import argparse
import errno
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm


FRAME_RATE = 75.0
AUDIO_EXTENSIONS = {
    ".flac",
    ".wav",
    ".wave",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".oga",
    ".opus",
    ".wma",
    ".alac",
    ".ape",
    ".wv",
    ".aif",
    ".aiff",
    ".mka",
    ".m4b",
}


@dataclass
class CueTrack:
    number: int
    title: str = ""
    performer: str = ""
    index_seconds: Optional[float] = None


def parse_timecode(raw: str) -> Optional[float]:
    parts = raw.strip().split(":")
    if len(parts) == 3:
        mm, ss, ff = parts
        try:
            return int(mm) * 60 + int(ss) + int(ff) / FRAME_RATE
        except ValueError:
            return None
    if len(parts) == 4:
        hh, mm, ss, ff = parts
        try:
            return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ff) / FRAME_RATE
        except ValueError:
            return None
    return None


def extract_quoted_or_rest(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    if '"' in raw:
        first = raw.find('"')
        last = raw.rfind('"')
        if last > first:
            return raw[first + 1 : last]
    return raw.strip()


def parse_cue(cue_path: Path) -> Tuple[Dict[str, List[CueTrack]], str, str]:
    files: Dict[str, List[CueTrack]] = {}
    global_performer = ""
    global_title = ""
    current_file: Optional[str] = None
    current_track: Optional[CueTrack] = None

    with cue_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.upper().startswith("REM"):
                continue

            upper = line.upper()
            if upper.startswith("FILE "):
                rest = line[5:].strip()
                file_value = ""
                if '"' in rest:
                    file_value = extract_quoted_or_rest(rest)
                else:
                    tokens = rest.split()
                    if tokens:
                        file_value = tokens[0]
                if file_value:
                    current_file = file_value
                    files.setdefault(current_file, [])
                current_track = None
                continue

            if upper.startswith("TRACK "):
                tokens = line.split()
                if len(tokens) >= 2 and tokens[1].isdigit():
                    track_no = int(tokens[1])
                    current_track = CueTrack(number=track_no)
                    if current_file is None:
                        current_file = ""
                        files.setdefault(current_file, [])
                    files[current_file].append(current_track)
                continue

            if upper.startswith("TITLE "):
                value = extract_quoted_or_rest(line[6:])
                if current_track is not None:
                    current_track.title = value
                else:
                    global_title = value
                continue

            if upper.startswith("PERFORMER "):
                value = extract_quoted_or_rest(line[10:])
                if current_track is not None:
                    current_track.performer = value
                else:
                    global_performer = value
                continue

            if upper.startswith("INDEX 01"):
                tokens = line.split()
                if len(tokens) >= 3 and current_track is not None:
                    timestamp = parse_timecode(tokens[2])
                    current_track.index_seconds = timestamp
                continue

    return files, global_performer, global_title


def resolve_audio_path(cue_dir: Path, file_value: str) -> Optional[Path]:
    if not file_value:
        return None
    candidate = Path(file_value)
    if not candidate.is_absolute():
        candidate = cue_dir / candidate
    if candidate.exists():
        return candidate

    lower_name = candidate.name.lower()
    try:
        for entry in cue_dir.iterdir():
            if entry.is_file() and entry.name.lower() == lower_name:
                return entry
    except FileNotFoundError:
        return None
    return None


def safe_filename(name: str) -> str:
    if not name:
        return ""
    invalid = '<>:/\\|?*"'
    cleaned = "".join("_" if c in invalid else c for c in name)
    cleaned = cleaned.strip().strip(".")
    return cleaned or "Track"


def truncate_filename(name: str, max_length: int) -> str:
    if len(name) <= max_length:
        return name
    return name[:max_length].rstrip(" ._")


def build_output_path(out_dir: Path, base: str, suffix: str) -> Path:
    safe_base = safe_filename(base)
    try:
        max_name = os.pathconf(str(out_dir), "PC_NAME_MAX")
        max_path = os.pathconf(str(out_dir), "PC_PATH_MAX")
    except (OSError, ValueError):
        max_name = 255
        max_path = 4096

    max_base_len = max(1, max_name - len(suffix))
    remaining_path = max_path - (len(str(out_dir)) + 1 + len(suffix))
    if remaining_path > 0:
        max_base_len = min(max_base_len, remaining_path)
    safe_base = truncate_filename(safe_base, max_base_len)
    return out_dir / f"{safe_base}{suffix}"


def ensure_dir_within_limits(out_dir: Path) -> Path:
    try:
        max_name = os.pathconf(str(out_dir.parent), "PC_NAME_MAX")
        max_path = os.pathconf(str(out_dir.parent), "PC_PATH_MAX")
    except (OSError, ValueError):
        max_name = 255
        max_path = 4096
    leaf = out_dir.name
    safe_leaf = safe_filename(leaf)
    safe_leaf = truncate_filename(safe_leaf, max_name)
    base = out_dir.parent / safe_leaf
    if len(str(base)) > max_path:
        remaining = max_path - (len(str(out_dir.parent)) + 1)
        safe_leaf = truncate_filename(safe_leaf, max(1, remaining))
        base = out_dir.parent / safe_leaf
    return base


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


def probe_duration(path: Path) -> Optional[float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def estimate_cue_length_seconds(tracks: List[CueTrack]) -> Optional[float]:
    ordered = [t for t in sorted(tracks, key=lambda t: t.number) if t.index_seconds is not None]
    if len(ordered) < 2:
        return None
    times = [t.index_seconds for t in ordered if t.index_seconds is not None]
    if any(times[idx] < times[idx - 1] for idx in range(1, len(times))):
        return None
    diffs = [times[idx] - times[idx - 1] for idx in range(1, len(times)) if times[idx] - times[idx - 1] > 0]
    if not diffs:
        return None
    diffs_sorted = sorted(diffs)
    mid = len(diffs_sorted) // 2
    if len(diffs_sorted) % 2 == 1:
        median = diffs_sorted[mid]
    else:
        median = (diffs_sorted[mid - 1] + diffs_sorted[mid]) / 2
    return times[-1] + median


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(round(seconds - (minutes * 60)))
    if secs >= 60:
        minutes += 1
        secs = 0
    return f"{minutes:02d}:{secs:02d}"


def codec_args(ext: str, reencode: bool) -> Tuple[List[str], Optional[str]]:
    ext = ext.lower()
    if not reencode:
        return ["-c", "copy"], None
    if ext == ".flac":
        return ["-c:a", "flac"], None
    if ext in {".wav", ".wave"}:
        return ["-c:a", "pcm_s16le"], None
    if ext == ".mp3":
        return ["-c:a", "libmp3lame"], None
    if ext == ".m4a":
        return ["-c:a", "aac"], None
    return ["-c", "copy"], "Unknown extension for re-encode; falling back to stream copy."


def split_album(
    cue_path: Path,
    audio_path: Path,
    tracks: List[CueTrack],
    out_dir: Path,
    album_name: str,
    default_performer: str,
    reencode: bool,
    overwrite: bool,
) -> None:
    tracks = [track for track in tracks if track.index_seconds is not None]
    if not tracks:
        tqdm.write(f"No indexed tracks found in {cue_path}")
        return

    tracks.sort(key=lambda t: t.index_seconds or 0.0)
    out_dir = ensure_dir_within_limits(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_tracks = len(tracks)
    codec, codec_warning = codec_args(audio_path.suffix, reencode)
    if codec_warning:
        tqdm.write(f"{audio_path}: {codec_warning}")

    for idx, track in enumerate(tracks):
        start = track.index_seconds or 0.0
        end = tracks[idx + 1].index_seconds if idx + 1 < total_tracks else None
        duration = end - start if end is not None else None

        title = track.title.strip() if track.title else f"Track {track.number:02d}"
        output_path = build_output_path(out_dir, title, audio_path.suffix)
        try:
            if output_path.exists() and not overwrite:
                continue
        except OSError as exc:
            if exc.errno != errno.ENAMETOOLONG:
                raise
            output_path = build_output_path(out_dir, f"{title}-{track.number:02d}", audio_path.suffix)

        artist = track.performer.strip() if track.performer else default_performer

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        cmd.append("-y" if overwrite else "-n")
        cmd += ["-i", str(audio_path)]
        cmd += ["-ss", f"{start:.3f}"]
        if duration is not None and duration > 0:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-map_metadata", "0", "-vn"]
        cmd += codec
        cmd += ["-metadata", f"title={title}"]
        cmd += ["-metadata", f"album={album_name}"]
        cmd += ["-metadata", f"track={track.number}"]
        cmd += ["-metadata", f"tracktotal={total_tracks}"]
        if artist:
            cmd += ["-metadata", f"artist={artist}"]
        cmd += [str(output_path)]

        try:
            subprocess.run(cmd, check=True)
        except OSError as exc:
            if exc.errno == errno.ENAMETOOLONG:
                fallback = build_output_path(out_dir, f"{title}-{track.number:02d}", audio_path.suffix)
                cmd[-1] = str(fallback)
                subprocess.run(cmd, check=True)
            else:
                raise


def gather_files(source_root: Path, dest_root: Path) -> Tuple[List[Path], List[Path]]:
    cue_files: List[Path] = []
    copy_files: List[Path] = []

    for root, _, files in os.walk(source_root):
        root_path = Path(root)
        for name in files:
            path = root_path / name
            if dest_root in path.parents:
                continue
            if path.suffix.lower() == ".cue":
                cue_files.append(path)
            else:
                copy_files.append(path)

    return cue_files, copy_files


def copy_non_cue_files(
    files: List[Path],
    source_root: Path,
    dest_root: Path,
    overwrite: bool,
    max_copy: Optional[int],
) -> None:
    if max_copy is not None:
        files = files[:max_copy]
    for src in tqdm(files, desc="Copying files", unit="file"):
        rel = src.relative_to(source_root)
        dst = dest_root / rel
        if dst.exists() and not overwrite:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            if exc.errno != errno.ENAMETOOLONG:
                raise
            truncated = build_output_path(dst.parent, dst.stem, dst.suffix)
            if truncated.exists() and not overwrite:
                continue
            shutil.copy2(src, truncated)


def build_duration_index(
    audio_paths: List[Path],
    duration_cache: Dict[Path, Optional[float]],
    bucket_size: float,
) -> Dict[int, List[Tuple[Path, float]]]:
    buckets: Dict[int, List[Tuple[Path, float]]] = {}
    for path in audio_paths:
        if path not in duration_cache:
            duration_cache[path] = probe_duration(path)
        duration = duration_cache[path]
        if duration is None:
            continue
        bucket = int(duration / bucket_size)
        buckets.setdefault(bucket, []).append((path, duration))
    return buckets


def render_track_titles(tracks: List[CueTrack]) -> List[str]:
    titles: List[str] = []
    for track in sorted(tracks, key=lambda t: t.number):
        title = track.title.strip() if track.title else f"Track {track.number:02d}"
        titles.append(title)
    return titles


def build_llm_messages(
    cue_path: Path,
    tracks: List[CueTrack],
    candidates: List[Tuple[Path, float]],
    estimated: float,
) -> List[Dict[str, str]]:
    track_titles = render_track_titles(tracks)
    if len(track_titles) > 15:
        track_display = track_titles[:12] + ["..."] + track_titles[-3:]
    else:
        track_display = track_titles

    candidate_lines = []
    for path, duration in candidates:
        diff = abs(duration - estimated)
        candidate_lines.append(
            f"- {path.name} (duration {format_duration(duration)}, diff {diff:.1f}s)"
        )

    example_1 = (
        "Example 1:\n"
        "CUE: AC-DC - Back In Black.cue\n"
        "Tracks: Hells Bells, Shoot to Thrill, What Do You Do for Money Honey\n"
        "Candidates:\n"
        "- AC-DC - Back In Black.flac (duration 41:58, diff 0.3s)\n"
        "- AC-DC - Highway To Hell.flac (duration 39:35, diff 143.0s)\n"
        "Estimated length: 41:58\n"
        "Return JSON with choice, confidence, reason."
    )
    example_2 = (
        "Example 2:\n"
        "CUE: Live in Hyde Park.cue\n"
        "Tracks: Intro, Somewhere, Live in Hyde Park\n"
        "Candidates:\n"
        "- Live at River Plate.flac (duration 73:10, diff 5.0s)\n"
        "- Live in Hyde Park.flac (duration 73:12, diff 7.0s)\n"
        "Estimated length: 73:05\n"
        "If ambiguous or unsure, choose NONE."
    )

    prompt = (
        "You are a cautious assistant that matches CUE sheets to audio files. "
        "Choose the single best match or NONE if not confident. "
        "Only choose when the cue name and track list strongly align with a candidate filename. "
        "Respond in JSON with keys: choice, confidence (0-1], reason (short). "
        "The choice must exactly match a candidate filename or be NONE."
    )

    user_payload = (
        f"CUE: {cue_path.name}\n"
        f"Tracks: {', '.join(track_display)}\n"
        "Candidates:\n"
        + "\n".join(candidate_lines)
        + f"\nEstimated length: {format_duration(estimated)}"
    )

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": example_1},
        {
            "role": "assistant",
            "content": '{"choice":"AC-DC - Back In Black.flac","confidence":0.92,"reason":"Album name and track list match the candidate file."}',
        },
        {"role": "user", "content": example_2},
        {
            "role": "assistant",
            "content": '{"choice":"NONE","confidence":0.35,"reason":"Both candidates are live recordings with similar durations; not confident."}',
        },
        {"role": "user", "content": user_payload},
    ]


def extract_json_object(text: str) -> Optional[dict]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    blob = stripped[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


def choose_match_with_llm(
    cue_path: Path,
    tracks: List[CueTrack],
    candidates: List[Tuple[Path, float]],
    estimated: float,
    endpoint: str,
    model: str,
    timeout: float,
    min_confidence: float,
) -> Optional[Path]:
    messages = build_llm_messages(cue_path, tracks, candidates, estimated)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 8192,
    }
    try:
        response = requests.post(
            f"{endpoint.rstrip('/')}/chat/completions", json=payload, timeout=timeout
        )
    except requests.RequestException as exc:
        tqdm.write(f"LLM request failed for {cue_path.name}: {exc}")
        return None
    if response.status_code != 200:
        tqdm.write(f"LLM request failed for {cue_path.name}: HTTP {response.status_code}")
        return None
    data = response.json()
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    tqdm.write(f"LLM response for {cue_path.name}: {content}")
    parsed = extract_json_object(content)
    if not parsed:
        return None
    choice = str(parsed.get("choice", "")).strip()
    confidence = parsed.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    if confidence_value < min_confidence:
        return None
    if choice.upper() == "NONE":
        return None
    by_name = {path.name: path for path, _ in candidates}
    if choice in by_name:
        return by_name[choice]
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1][0]
    for name, path in by_name.items():
        if name.lower() == choice.lower():
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split albums by CUE files and copy non-CUE files to a destination."
    )
    parser.add_argument(
        "--source",
        default="~/projects/navidrome/music",
        help="Source music folder (default: %(default)s)",
    )
    parser.add_argument(
        "--dest",
        default="~/projects/navidrome/new_music",
        help="Destination folder (default: %(default)s)",
    )
    parser.add_argument(
        "--max-cues",
        type=int,
        default=None,
        help="Limit the number of cue files processed (for testing)",
    )
    parser.add_argument(
        "--max-copy",
        type=int,
        default=None,
        help="Limit the number of non-cue files copied (for testing)",
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Skip copying non-cue files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Only copy non-cue files; skip splitting",
    )
    parser.add_argument(
        "--no-reencode",
        action="store_true",
        help="Use stream copy instead of re-encoding",
    )
    parser.add_argument(
        "--match-by-duration",
        action="store_true",
        help="Match cue files to audio by estimated duration when filenames differ",
    )
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=0.5,
        help="Tolerance in seconds when matching by duration (default: %(default)s)",
    )
    parser.add_argument(
        "--match-by-llm",
        action="store_true",
        help="Use an OpenAI-compatible model to auto-pick duration matches",
    )
    parser.add_argument(
        "--llm-endpoint",
        default="http://192.168.1.185:8808/v1",
        help="OpenAI-compatible API endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--llm-model",
        default="glm",
        help="Model name for the OpenAI-compatible API (default: %(default)s)",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=30.0,
        help="LLM request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--llm-min-confidence",
        type=float,
        default=0.75,
        help="Minimum confidence to accept LLM choice (default: %(default)s)",
    )

    args = parser.parse_args()

    source_root = Path(os.path.expanduser(args.source)).resolve()
    dest_root = Path(os.path.expanduser(args.dest)).resolve()

    if not source_root.exists():
        print(f"Source path does not exist: {source_root}", file=sys.stderr)
        return 2

    if not args.copy_only and not ffmpeg_available():
        print("ffmpeg not found in PATH. Please install ffmpeg.", file=sys.stderr)
        return 2

    dest_root.mkdir(parents=True, exist_ok=True)

    cue_files, copy_files = gather_files(source_root, dest_root)
    if args.max_cues is not None:
        cue_files = cue_files[: args.max_cues]

    audio_by_dir: Dict[Path, List[Path]] = {}
    for path in copy_files:
        if is_audio_file(path):
            audio_by_dir.setdefault(path.parent, []).append(path)

    handled_cue_entries: set[Tuple[Path, str]] = set()
    matched_audio: set[Path] = set()
    unmatched_entries: List[Tuple[Path, str, List[CueTrack], str, Path]] = []

    if not args.copy_only:
        for cue_path in tqdm(cue_files, desc="Splitting CUEs", unit="cue"):
            files, global_performer, _global_title = parse_cue(cue_path)
            cue_dir = cue_path.parent

            if not files:
                tqdm.write(f"No FILE entries found in {cue_path}")
                continue

            for file_value, tracks in files.items():
                cue_key = (cue_path, file_value)
                if cue_key in handled_cue_entries:
                    continue
                audio_path = resolve_audio_path(cue_dir, file_value)
                if audio_path is None:
                    if args.match_by_duration:
                        unmatched_entries.append(
                            (cue_path, file_value, tracks, global_performer, cue_dir)
                        )
                    else:
                        tqdm.write(f"Audio file not found for {cue_path}: {file_value}")
                    continue

                album_name = audio_path.stem
                output_dir = dest_root / cue_path.parent.relative_to(source_root) / album_name
                try:
                    split_album(
                        cue_path,
                        audio_path,
                        tracks,
                        output_dir,
                        album_name,
                        global_performer,
                        reencode=not args.no_reencode,
                        overwrite=args.overwrite,
                    )
                    handled_cue_entries.add(cue_key)
                    matched_audio.add(audio_path)
                except subprocess.CalledProcessError as exc:
                    tqdm.write(f"ffmpeg failed for {audio_path}: {exc}")

    if args.match_by_duration and unmatched_entries:
        if not ffprobe_available():
            print("ffprobe not found in PATH. Please install ffmpeg/ffprobe.", file=sys.stderr)
            return 2
        duration_cache: Dict[Path, Optional[float]] = {}
        bucket_size = args.duration_tolerance if args.duration_tolerance > 0 else 0.01
        unmatched_by_dir: Dict[Path, List[Tuple[Path, str, List[CueTrack], str, Path]]] = {}
        for entry in unmatched_entries:
            unmatched_by_dir.setdefault(entry[4], []).append(entry)

        for cue_dir, entries in tqdm(
            list(unmatched_by_dir.items()), desc="Matching by duration", unit="dir"
        ):
            candidate_audio = [p for p in audio_by_dir.get(cue_dir, []) if p not in matched_audio]
            if not candidate_audio:
                continue
            duration_index = build_duration_index(candidate_audio, duration_cache, bucket_size)

            for cue_path, file_value, tracks, global_performer, _ in entries:
                cue_key = (cue_path, file_value)
                if cue_key in handled_cue_entries:
                    continue
                estimated = estimate_cue_length_seconds(tracks)
                if estimated is None:
                    continue
                bucket = int(estimated / bucket_size)
                candidates: List[Tuple[Path, float]] = []
                for bucket_id in (bucket - 1, bucket, bucket + 1):
                    candidates.extend(duration_index.get(bucket_id, []))
                matches: Dict[Path, float] = {}
                for path, duration in candidates:
                    if path in matched_audio:
                        continue
                    if abs(duration - estimated) <= args.duration_tolerance:
                        matches[path] = duration
                if not matches:
                    continue
                ordered_matches = sorted(
                    matches.items(), key=lambda item: abs(item[1] - estimated)
                )
                if args.match_by_llm:
                    chosen = choose_match_with_llm(
                        cue_path,
                        tracks,
                        ordered_matches,
                        estimated,
                        endpoint=args.llm_endpoint,
                        model=args.llm_model,
                        timeout=args.llm_timeout,
                        min_confidence=args.llm_min_confidence,
                    )
                    if chosen is None:
                        handled_cue_entries.add(cue_key)
                        continue
                    audio_path = chosen
                    album_name = audio_path.stem
                    output_dir = dest_root / cue_path.parent.relative_to(source_root) / album_name
                    try:
                        split_album(
                            cue_path,
                            audio_path,
                            tracks,
                            output_dir,
                            album_name,
                            global_performer,
                            reencode=not args.no_reencode,
                            overwrite=args.overwrite,
                        )
                        handled_cue_entries.add(cue_key)
                        matched_audio.add(audio_path)
                    except subprocess.CalledProcessError as exc:
                        tqdm.write(f"ffmpeg failed for {audio_path}: {exc}")
                    continue
                if len(ordered_matches) == 1:
                    audio_path, duration = ordered_matches[0]
                    tqdm.write(
                        f"Duration match for {cue_path.name}: {audio_path.name} "
                        f"(~{format_duration(estimated)} vs {format_duration(duration)})"
                    )
                    resp = input("Process this match? [y/N]: ").strip().lower()
                    if resp in {"y", "yes"}:
                        album_name = audio_path.stem
                        output_dir = dest_root / cue_path.parent.relative_to(source_root) / album_name
                        try:
                            split_album(
                                cue_path,
                                audio_path,
                                tracks,
                                output_dir,
                                album_name,
                                global_performer,
                                reencode=not args.no_reencode,
                                overwrite=args.overwrite,
                            )
                            handled_cue_entries.add(cue_key)
                            matched_audio.add(audio_path)
                        except subprocess.CalledProcessError as exc:
                            tqdm.write(f"ffmpeg failed for {audio_path}: {exc}")
                    else:
                        handled_cue_entries.add(cue_key)
                else:
                    tqdm.write(
                        f"Multiple duration matches for {cue_path.name} "
                        f"(~{format_duration(estimated)}):"
                    )
                    for idx, (audio_path, duration) in enumerate(ordered_matches, start=1):
                        tqdm.write(f"  {idx}) {audio_path.name} ({format_duration(duration)})")
                    resp = input("Select number to process, or 's' to skip: ").strip().lower()
                    if resp.isdigit():
                        choice = int(resp)
                        if 1 <= choice <= len(ordered_matches):
                            audio_path, duration = ordered_matches[choice - 1]
                            album_name = audio_path.stem
                            output_dir = (
                                dest_root / cue_path.parent.relative_to(source_root) / album_name
                            )
                            try:
                                split_album(
                                    cue_path,
                                    audio_path,
                                    tracks,
                                    output_dir,
                                    album_name,
                                    global_performer,
                                    reencode=not args.no_reencode,
                                    overwrite=args.overwrite,
                                )
                                handled_cue_entries.add(cue_key)
                                matched_audio.add(audio_path)
                            except subprocess.CalledProcessError as exc:
                                tqdm.write(f"ffmpeg failed for {audio_path}: {exc}")
                        else:
                            handled_cue_entries.add(cue_key)
                    else:
                        handled_cue_entries.add(cue_key)

    if not args.skip_copy:
        copy_non_cue_files(
            copy_files,
            source_root,
            dest_root,
            overwrite=args.overwrite,
            max_copy=args.max_copy,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    #python scripts/split_cue_albums.py --source ~/projects/navidrome/music --dest /mnt/data/share/hosted/new_music --max-cues 1 --skip-copy --match-by-duration --duration-tolerance 1
