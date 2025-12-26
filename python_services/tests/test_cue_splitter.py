import pytest
import torch
import torchaudio

from cue_splitter import (
    _build_track_context,
    _cue_time_to_seconds,
    _read_cuesheet_text,
    _sanitize_filename,
    split_flac_with_cue,
)
from mutagen.flac import FLAC


def test_split_flac_with_cue_creates_tracks(tmp_path):
    sample_rate = 44100
    total_seconds = 4
    samples = sample_rate * total_seconds
    waveform = torch.zeros((1, samples))
    source_path = tmp_path / "album.flac"
    torchaudio.save(str(source_path), waveform, sample_rate, format="FLAC")

    cue_text = """
PERFORMER "Album Artist"
TITLE "Test Album"
FILE "album.flac" WAVE
  TRACK 01 AUDIO
    TITLE "Intro"
    PERFORMER "Track Artist"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "Outro"
    INDEX 01 00:02:00
"""
    cue_path = tmp_path / "album.cue"
    cue_path.write_text(cue_text.strip(), encoding="utf-8")

    tracks = split_flac_with_cue(str(source_path), str(cue_path))

    assert len(tracks) == 2
    first, second = tracks

    assert first.title == "Intro"
    assert first.artist == "Track Artist"
    assert first.album == "Test Album"
    assert first.album_artist == "Album Artist"
    assert first.file_path.exists()
    assert first.duration_seconds == pytest.approx(2.0, abs=0.05)

    assert second.title == "Outro"
    assert second.artist == "Album Artist"
    assert second.album == "Test Album"
    assert second.file_path.exists()
    assert second.duration_seconds == pytest.approx(2.0, abs=0.05)

    first_tags = FLAC(str(first.file_path))
    assert first_tags["title"] == ["Intro"]
    assert first_tags["artist"] == ["Track Artist"]
    assert first_tags["album"] == ["Test Album"]
    assert first_tags["tracknumber"] == ["1"]

    second_tags = FLAC(str(second.file_path))
    assert second_tags["title"] == ["Outro"]
    assert second_tags["artist"] == ["Album Artist"]
    assert second_tags["album"] == ["Test Album"]
    assert second_tags["tracknumber"] == ["2"]


def test_sanitize_filename_handles_illegal_chars():
    name = _sanitize_filename('My:/Track*Name?', 3)
    assert name.startswith("03 - ")
    assert ":" not in name
    assert "*" not in name


def test_cue_time_to_seconds_parses_frames():
    assert _cue_time_to_seconds("01:02:37") == pytest.approx(62.493, rel=1e-3)


def test_build_track_context_parses_track_metadata():
    lines = [
        'FILE "album.flac" WAVE',
        "  TRACK 01 AUDIO",
        '    TITLE "Intro"',
        '    PERFORMER "Artist"',
        "  TRACK 02 AUDIO",
        '    TITLE "Outro"',
    ]
    context = _build_track_context(lines)
    assert context[0]["title"] == "Intro"
    assert context[0]["performer"] == "Artist"
    assert context[1]["title"] == "Outro"


def test_read_cuesheet_text_strips_bom(tmp_path):
    cue_path = tmp_path / "test.cue"
    cue_path.write_text("\ufeffFILE \"album.flac\" WAVE", encoding="utf-8")
    text = _read_cuesheet_text(cue_path)
    assert text.startswith("FILE")
