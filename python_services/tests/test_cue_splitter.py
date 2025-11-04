import pytest
import torch
import torchaudio

from cue_splitter import split_flac_with_cue
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
