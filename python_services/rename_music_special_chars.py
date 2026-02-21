#!/usr/bin/env python3
"""Rename files to be portable across Linux/Windows filename rules."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm

INVALID_WINDOWS_CHARS = set('<>:"/\\|?*')
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


@dataclass(frozen=True)
class RenamePlanItem:
    source: Path
    target: Path
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class ConflictItem:
    item: RenamePlanItem
    conflict_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class RenameError:
    source: Path
    target: Path
    stage: str
    error: str


def sanitize_stem(stem: str) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    output_chars: List[str] = []
    saw_invalid_char = False
    saw_control_char = False

    for char in stem:
        if ord(char) < 32:
            output_chars.append("_")
            saw_control_char = True
        elif char in INVALID_WINDOWS_CHARS:
            output_chars.append("_")
            saw_invalid_char = True
        else:
            output_chars.append(char)

    if saw_invalid_char:
        reasons.append("invalid_char")
    if saw_control_char:
        reasons.append("control_char")

    sanitized = "".join(output_chars)
    trimmed = sanitized.rstrip(" .")
    if trimmed != sanitized:
        reasons.append("trailing_space_or_dot")
        sanitized = trimmed

    if not sanitized:
        sanitized = "unnamed"
        reasons.append("empty_stem")

    if sanitized.upper() in WINDOWS_RESERVED_NAMES:
        sanitized = f"{sanitized}_"
        reasons.append("reserved_name")

    return sanitized, reasons


def scan_regular_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in tqdm(root.rglob("*"), desc="Scanning files", unit="entry"):
        if path.is_file():
            files.append(path)
    return files


def build_rename_plan(files: Iterable[Path]) -> List[RenamePlanItem]:
    planned: List[RenamePlanItem] = []
    for path in tqdm(list(files), desc="Planning renames", unit="file"):
        sanitized_stem, reasons = sanitize_stem(path.stem)
        if sanitized_stem == path.stem:
            continue
        target = path.with_name(f"{sanitized_stem}{path.suffix}")
        planned.append(
            RenamePlanItem(
                source=path,
                target=target,
                reasons=tuple(reasons),
            )
        )
    return planned


def detect_conflicts(
    planned: Sequence[RenamePlanItem], all_files: Sequence[Path]
) -> Tuple[List[RenamePlanItem], List[ConflictItem]]:
    source_paths = {item.source for item in planned}
    unchanged_paths = {path for path in all_files if path not in source_paths}

    target_to_items: Dict[Path, List[RenamePlanItem]] = {}
    for item in planned:
        target_to_items.setdefault(item.target, []).append(item)

    conflict_reasons_by_source: Dict[Path, List[str]] = {}
    for target, items in target_to_items.items():
        if len(items) > 1:
            for item in items:
                conflict_reasons_by_source.setdefault(item.source, []).append(
                    "multiple_sources_same_target"
                )
        if target in unchanged_paths:
            for item in items:
                conflict_reasons_by_source.setdefault(item.source, []).append(
                    "target_exists_unchanged_file"
                )
        elif target.exists() and target not in source_paths:
            for item in items:
                conflict_reasons_by_source.setdefault(item.source, []).append(
                    "target_exists"
                )

    ready: List[RenamePlanItem] = []
    conflicts: List[ConflictItem] = []
    for item in planned:
        conflict_reasons = conflict_reasons_by_source.get(item.source, [])
        if not conflict_reasons:
            ready.append(item)
            continue
        conflicts.append(
            ConflictItem(
                item=item,
                conflict_reasons=tuple(sorted(set(conflict_reasons))),
            )
        )

    return ready, conflicts


def create_temp_path(source: Path) -> Path:
    while True:
        candidate = source.with_name(f".rename_tmp_{uuid.uuid4().hex}{source.suffix}")
        if not candidate.exists():
            return candidate


def apply_two_phase_renames(
    items: Sequence[RenamePlanItem],
) -> Tuple[List[RenamePlanItem], List[RenameError]]:
    phase_one_successes: List[Tuple[RenamePlanItem, Path]] = []
    errors: List[RenameError] = []

    for item in tqdm(items, desc="Phase 1 temp renames", unit="file"):
        temp_path = create_temp_path(item.source)
        try:
            item.source.rename(temp_path)
            phase_one_successes.append((item, temp_path))
        except OSError as exc:
            errors.append(
                RenameError(
                    source=item.source,
                    target=item.target,
                    stage="phase_1",
                    error=str(exc),
                )
            )

    renamed: List[RenamePlanItem] = []
    for item, temp_path in tqdm(phase_one_successes, desc="Phase 2 final renames", unit="file"):
        try:
            temp_path.rename(item.target)
            renamed.append(item)
        except OSError as exc:
            errors.append(
                RenameError(
                    source=item.source,
                    target=item.target,
                    stage="phase_2",
                    error=str(exc),
                )
            )
            try:
                if not item.source.exists():
                    temp_path.rename(item.source)
            except OSError as rollback_exc:
                errors.append(
                    RenameError(
                        source=item.source,
                        target=item.target,
                        stage="rollback",
                        error=str(rollback_exc),
                    )
                )

    return renamed, errors


def render_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def emit_summary(summary: Dict[str, int]) -> None:
    tqdm.write("Summary:")
    for key in ("scanned", "planned", "renamed", "unchanged", "skipped_conflict", "errors"):
        tqdm.write(f"  {key}: {summary[key]}")


def emit_samples(
    root: Path,
    planned: Sequence[RenamePlanItem],
    conflicts: Sequence[ConflictItem],
    limit: int = 10,
) -> None:
    if planned:
        tqdm.write("Sample planned renames:")
        for item in planned[:limit]:
            tqdm.write(
                f"  {render_path(item.source, root)} -> {render_path(item.target, root)} "
                f"(reasons: {', '.join(item.reasons)})"
            )
    if conflicts:
        tqdm.write("Sample conflicts:")
        for conflict in conflicts[:limit]:
            item = conflict.item
            tqdm.write(
                f"  {render_path(item.source, root)} -> {render_path(item.target, root)} "
                f"(conflicts: {', '.join(conflict.conflict_reasons)})"
            )


def write_report(
    report_path: Path,
    root: Path,
    dry_run: bool,
    summary: Dict[str, int],
    planned: Sequence[RenamePlanItem],
    renamed: Sequence[RenamePlanItem],
    conflicts: Sequence[ConflictItem],
    errors: Sequence[RenameError],
) -> None:
    data = {
        "root": str(root),
        "mode": "dry_run" if dry_run else "apply",
        "summary": summary,
        "planned": [
            {
                "source": render_path(item.source, root),
                "target": render_path(item.target, root),
                "reasons": list(item.reasons),
            }
            for item in planned
        ],
        "applied": [
            {
                "source": render_path(item.source, root),
                "target": render_path(item.target, root),
            }
            for item in renamed
        ],
        "skipped_conflict": [
            {
                "source": render_path(conflict.item.source, root),
                "target": render_path(conflict.item.target, root),
                "reasons": list(conflict.item.reasons),
                "conflict_reasons": list(conflict.conflict_reasons),
            }
            for conflict in conflicts
        ],
        "errors": [
            {
                "source": render_path(err.source, root),
                "target": render_path(err.target, root),
                "stage": err.stage,
                "error": err.error,
            }
            for err in errors
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rename files under a root directory when names violate Windows/Linux "
            "filesystem compatibility rules."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory to scan recursively",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply renames (default is dry-run)",
    )
    parser.add_argument(
        "--report-json",
        help="Optional output path for a JSON report",
    )

    args = parser.parse_args(argv)

    root = Path(os.path.expanduser(args.root)).resolve()
    if not root.exists():
        print(f"Root path does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"Root path is not a directory: {root}", file=sys.stderr)
        return 2

    files = scan_regular_files(root)
    planned = build_rename_plan(files)
    ready, conflicts = detect_conflicts(planned, files)

    renamed: List[RenamePlanItem] = []
    errors: List[RenameError] = []
    if args.apply:
        renamed, errors = apply_two_phase_renames(ready)
    else:
        tqdm.write("Dry-run mode: no files were renamed. Use --apply to apply changes.")

    summary = {
        "scanned": len(files),
        "planned": len(planned),
        "renamed": len(renamed),
        "unchanged": len(files) - len(planned),
        "skipped_conflict": len(conflicts),
        "errors": len(errors),
    }
    emit_summary(summary)
    emit_samples(root, planned, conflicts)

    if args.report_json:
        report_path = Path(os.path.expanduser(args.report_json)).resolve()
        write_report(
            report_path=report_path,
            root=root,
            dry_run=not args.apply,
            summary=summary,
            planned=planned,
            renamed=renamed,
            conflicts=conflicts,
            errors=errors,
        )
        tqdm.write(f"Wrote JSON report: {report_path}")

    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
