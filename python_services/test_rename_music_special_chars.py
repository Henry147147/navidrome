#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rename_music_special_chars as sanitizer


class RenameMusicSpecialCharsTests(unittest.TestCase):
    def test_em_dash_unchanged_without_option(self) -> None:
        sanitized, reasons = sanitizer.sanitize_stem("Song—Live")
        self.assertEqual(sanitized, "Song—Live")
        self.assertEqual(reasons, [])

    def test_em_dash_replaced_with_option(self) -> None:
        sanitized, reasons = sanitizer.sanitize_stem(
            "Song—Live", replace_problematic_chars=True
        )
        self.assertEqual(sanitized, "Song-Live")
        self.assertIn("dash_variant", reasons)

    def test_problematic_unicode_chars_replaced_with_option(self) -> None:
        sanitized, reasons = sanitizer.sanitize_stem(
            "A\u00A0B\u200BC", replace_problematic_chars=True
        )
        self.assertEqual(sanitized, "A B_C")
        self.assertIn("unicode_whitespace", reasons)
        self.assertIn("format_or_surrogate_char", reasons)

    def test_invalid_character_replacement(self) -> None:
        sanitized, reasons = sanitizer.sanitize_stem("song?:*")
        self.assertEqual(sanitized, "song___")
        self.assertIn("invalid_char", reasons)

    def test_control_character_replacement(self) -> None:
        sanitized, reasons = sanitizer.sanitize_stem("ab\x01cd")
        self.assertEqual(sanitized, "ab_cd")
        self.assertIn("control_char", reasons)

    def test_trailing_space_and_dot_trim(self) -> None:
        sanitized, reasons = sanitizer.sanitize_stem("track. ")
        self.assertEqual(sanitized, "track")
        self.assertIn("trailing_space_or_dot", reasons)

    def test_reserved_name_adjustment(self) -> None:
        sanitized, reasons = sanitizer.sanitize_stem("CON")
        self.assertEqual(sanitized, "CON_")
        self.assertIn("reserved_name", reasons)

    def test_extension_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "track?.flac"
            original.write_text("x", encoding="utf-8")

            code = sanitizer.run(["--root", str(root), "--apply"])

            self.assertEqual(code, 0)
            self.assertFalse(original.exists())
            self.assertTrue((root / "track_.flac").exists())

    def test_conflict_skip_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "same?.mp3"
            second = root / "same*.mp3"
            report = root / "report.json"
            first.write_text("one", encoding="utf-8")
            second.write_text("two", encoding="utf-8")

            code = sanitizer.run(
                ["--root", str(root), "--apply", "--report-json", str(report)]
            )

            self.assertEqual(code, 0)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertFalse((root / "same_.mp3").exists())

            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["skipped_conflict"], 2)
            self.assertEqual(len(data["skipped_conflict"]), 2)

    def test_dry_run_does_not_change_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "demo?.mp3"
            report = root / "dry_run.json"
            original.write_text("demo", encoding="utf-8")

            code = sanitizer.run(["--root", str(root), "--report-json", str(report)])

            self.assertEqual(code, 0)
            self.assertTrue(original.exists())
            self.assertFalse((root / "demo_.mp3").exists())
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["renamed"], 0)

    def test_apply_mode_performs_renames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "bad:name.mp3"
            report = root / "apply.json"
            original.write_text("content", encoding="utf-8")

            code = sanitizer.run(
                ["--root", str(root), "--apply", "--report-json", str(report)]
            )

            self.assertEqual(code, 0)
            self.assertFalse(original.exists())
            self.assertTrue((root / "bad_name.mp3").exists())
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["renamed"], 1)
            self.assertEqual(len(data["applied"]), 1)

    def test_two_phase_rename_handles_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a = root / "a.txt"
            source_b = root / "b.txt"
            target_b = root / "b.txt"
            target_c = root / "c.txt"

            source_a.write_text("A", encoding="utf-8")
            source_b.write_text("B", encoding="utf-8")

            items = [
                sanitizer.RenamePlanItem(source=source_a, target=target_b, reasons=()),
                sanitizer.RenamePlanItem(source=source_b, target=target_c, reasons=()),
            ]

            renamed, errors = sanitizer.apply_two_phase_renames(items)

            self.assertEqual(len(errors), 0)
            self.assertEqual(len(renamed), 2)
            self.assertFalse(source_a.exists())
            self.assertTrue(target_b.exists())
            self.assertTrue(target_c.exists())
            self.assertEqual(target_b.read_text(encoding="utf-8"), "A")
            self.assertEqual(target_c.read_text(encoding="utf-8"), "B")

    def test_apply_mode_with_problematic_char_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "track—name.flac"
            report = root / "extended.json"
            original.write_text("x", encoding="utf-8")

            code = sanitizer.run(
                [
                    "--root",
                    str(root),
                    "--apply",
                    "--replace-problematic-chars",
                    "--report-json",
                    str(report),
                ]
            )

            self.assertEqual(code, 0)
            self.assertFalse(original.exists())
            self.assertTrue((root / "track-name.flac").exists())
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(data["options"]["replace_problematic_chars"])
            self.assertEqual(data["summary"]["renamed"], 1)


if __name__ == "__main__":
    unittest.main()
