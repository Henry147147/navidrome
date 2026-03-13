#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python_services.muq_milvus import EmbeddingRow, MilvusEmbeddingStore
from python_services.muq_pipeline import BatchJobManager
from python_services.muq_provider import MuQProvider, MuQProviderConfig, _pool_music_flamingo_audio_tokens
from python_services.muq_service import MuQService
from python_services.muq_shared import (
    AUDIO_SAMPLE_RATE,
    COLLECTION_MUSIC_FLAMINGO_AUDIO,
    COLLECTION_MUQ_AUDIO,
    COLLECTION_MUQ_MULAN,
    LEGACY_COLLECTIONS,
    MODEL_MUSIC_FLAMINGO_AUDIO,
    MODEL_MUQ_AUDIO,
    MODEL_MUQ_MULAN,
    TrackInfo,
    normalize_model_names,
)


class _FakeBackend:
    def __init__(self) -> None:
        self.dropped: list[str] = []
        self.ensured: list[tuple[str, int]] = []
        self.upserts: list[tuple[str, list[EmbeddingRow]]] = []
        self.flushed: list[str] = []
        self.existing_by_collection: dict[str, set[str]] = {}
        self.queries: list[tuple[str, list[str]]] = []

    def drop_collection(self, name: str) -> None:
        self.dropped.append(name)

    def ensure_collection(self, name: str, dimension: int) -> None:
        self.ensured.append((name, dimension))

    def upsert_rows(self, name: str, rows: list[EmbeddingRow]) -> None:
        self.upserts.append((name, rows))

    def query_names(self, name: str, names: list[str]) -> set[str]:
        self.queries.append((name, list(names)))
        return set(self.existing_by_collection.get(name, set())).intersection(names)

    def flush(self, name: str) -> None:
        self.flushed.append(name)


class _FakeProvider:
    def __init__(self) -> None:
        self.audio_calls: list[tuple[list[str], str]] = []
        self.text_calls: list[tuple[list[str], str, int]] = []

    def embed_audio_files(self, paths: list[str], model: str) -> list[list[float]]:
        self.audio_calls.append((list(paths), model))
        if model == MODEL_MUSIC_FLAMINGO_AUDIO:
            return [[0.0, 0.6, 0.8, 0.0] for _ in paths]
        if model == MODEL_MUQ_AUDIO:
            return [[0.6, 0.8, 0.0] for _ in paths]
        return [[1.0, 0.0] for _ in paths]

    def embed_texts(self, texts: list[str], *, model: str, dimensions: int = 0) -> list[list[float]]:
        self.text_calls.append((list(texts), model, dimensions))
        rows = [[float(index + 1), float(index + 2), float(index + 3)] for index in range(len(texts))]
        if dimensions > 0:
            rows = [row[:dimensions] for row in rows]
        return rows


class _SlowProvider(_FakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def embed_audio_files(self, paths: list[str], model: str) -> list[list[float]]:
        self.started.set()
        self.release.wait(1.0)
        return super().embed_audio_files(paths, model)


class _FakeBatchManager:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool]] = []

    def start(self, *, models: list[str], clear_existing: bool) -> dict[str, object]:
        self.calls.append((list(models), clear_existing))
        return {"status": "started", "models": list(models), "clear_existing": clear_existing}

    def progress(self) -> dict[str, object]:
        return {"status": "idle"}

    def cancel(self) -> dict[str, object]:
        return {"status": "cancelled"}


class MuQSharedTests(unittest.TestCase):
    def test_explicit_music_flamingo_model_is_preserved(self) -> None:
        self.assertEqual(
            normalize_model_names(["music_flamingo_audio", "audio", "lyrics", "description", "qwen8b"]),
            [MODEL_MUSIC_FLAMINGO_AUDIO, MODEL_MUQ_AUDIO, MODEL_MUQ_MULAN],
        )

    def test_legacy_flamingo_alias_still_maps_to_muq_audio(self) -> None:
        self.assertEqual(
            normalize_model_names(["flamingo", "music-flamingo", "music_flamingo"]),
            [MODEL_MUQ_AUDIO],
        )


class MuQProviderTests(unittest.TestCase):
    def test_load_audio_file_resamples_to_24khz_and_mixes_to_mono(self) -> None:
        stereo = [[1.0, 3.0, 5.0], [3.0, 5.0, 7.0]]
        expected_mono = [2.0, 4.0, 6.0]
        expected_resampled = [0.0, 0.5, 1.0, 1.5]

        with patch("python_services.muq_provider.Path.is_file", return_value=True), patch(
            "python_services.muq_provider.librosa.load", return_value=(stereo, 48_000)
        ), patch(
            "python_services.muq_provider.librosa.resample",
            return_value=expected_resampled,
        ) as mock_resample:
            provider = MuQProvider(
                MuQProviderConfig(
                    audio_model_ref="audio-ref",
                    mulan_model_ref="mulan-ref",
                    cache_dir="",
                    device="cpu",
                )
            )
            waveform = provider.load_audio_file("track.flac")

        self.assertTrue(torch.equal(waveform, torch.tensor(expected_resampled, dtype=torch.float32)))
        self.assertEqual(mock_resample.call_args.kwargs["orig_sr"], 48_000)
        self.assertEqual(mock_resample.call_args.kwargs["target_sr"], AUDIO_SAMPLE_RATE)
        self.assertEqual(mock_resample.call_args.args[0].tolist(), expected_mono)

    def test_embed_audio_waveforms_pads_batch_and_normalizes_masked_pooling(self) -> None:
        provider = MuQProvider(
            MuQProviderConfig(
                audio_model_ref="audio-ref",
                mulan_model_ref="mulan-ref",
                cache_dir="",
                device="cpu",
                audio_batch_size=2,
            )
        )

        hidden_states = torch.tensor(
            [
                [[3.0, 4.0], [3.0, 4.0]],
                [[0.0, 5.0], [9.0, 9.0]],
            ],
            dtype=torch.float32,
        )
        feature_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)

        class _FakeMuQAudioModel:
            def __init__(self) -> None:
                self.model = SimpleNamespace(
                    conformer=SimpleNamespace(
                        _get_feature_vector_attention_mask=lambda _length, _mask: feature_mask,
                    )
                )
                self.last_batch = None
                self.last_attention_mask = None

            def __call__(self, batch: torch.Tensor, attention_mask: torch.Tensor, output_hidden_states: bool = True):
                self.last_batch = batch.detach().cpu()
                self.last_attention_mask = attention_mask.detach().cpu()
                return SimpleNamespace(last_hidden_state=hidden_states)

        fake_model = _FakeMuQAudioModel()

        with patch.object(provider, "_load_muq_audio_model", return_value=fake_model):
            vectors = provider.embed_audio_waveforms(
                [torch.tensor([1.0, 2.0]), torch.tensor([9.0])],
                MODEL_MUQ_AUDIO,
            )

        self.assertEqual(fake_model.last_batch.tolist(), [[1.0, 2.0], [9.0, 0.0]])
        self.assertEqual(fake_model.last_attention_mask.tolist(), [[1, 1], [1, 0]])
        self.assertAlmostEqual(vectors[0][0], 0.6, places=6)
        self.assertAlmostEqual(vectors[0][1], 0.8, places=6)
        self.assertAlmostEqual(vectors[1][0], 0.0, places=6)
        self.assertAlmostEqual(vectors[1][1], 1.0, places=6)

    def test_embed_audio_files_flattens_and_normalizes_music_flamingo_embeddings(self) -> None:
        provider = MuQProvider(
            MuQProviderConfig(
                flamingo_model_ref="flamingo-ref",
                audio_model_ref="audio-ref",
                mulan_model_ref="mulan-ref",
                cache_dir="",
                device="cpu",
            )
        )

        class _FakeMusicFlamingoModel:
            def __init__(self) -> None:
                self.paths: list[str] = []

            def extract_embedding(self, path: str) -> torch.Tensor:
                self.paths.append(path)
                return torch.tensor([3.0, 4.0, 0.0, 0.0], dtype=torch.float32)

        fake_model = _FakeMusicFlamingoModel()

        with patch.object(provider, "_load_music_flamingo_model", return_value=fake_model):
            vectors = provider.embed_audio_files(["track.flac"], MODEL_MUSIC_FLAMINGO_AUDIO)

        self.assertEqual(fake_model.paths, ["track.flac"])
        self.assertEqual(len(vectors), 1)
        self.assertAlmostEqual(vectors[0][0], 0.6, places=6)
        self.assertAlmostEqual(vectors[0][1], 0.8, places=6)
        self.assertAlmostEqual(vectors[0][2], 0.0, places=6)
        self.assertAlmostEqual(vectors[0][3], 0.0, places=6)

    def test_pool_music_flamingo_tokens_uses_only_valid_frames(self) -> None:
        projected = torch.tensor(
            [
                [
                    [1.0, 1.0],
                    [3.0, 3.0],
                    [9.0, 9.0],
                ]
            ],
            dtype=torch.float32,
        )
        input_features_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)

        pooled = _pool_music_flamingo_audio_tokens(projected, input_features_mask)

        self.assertEqual(tuple(pooled.shape), (1, 2))
        self.assertTrue(torch.equal(pooled[0], torch.tensor([2.0, 2.0], dtype=torch.float32)))

    def test_embed_audio_files_averages_multiple_music_flamingo_windows(self) -> None:
        provider = MuQProvider(
            MuQProviderConfig(
                flamingo_model_ref="flamingo-ref",
                audio_model_ref="audio-ref",
                mulan_model_ref="mulan-ref",
                cache_dir="",
                device="cpu",
            )
        )

        class _FakeWindowedMusicFlamingoModel:
            def extract_embedding(self, path: str) -> torch.Tensor:
                self.last_path = path
                return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        fake_model = _FakeWindowedMusicFlamingoModel()

        with patch.object(provider, "_load_music_flamingo_model", return_value=fake_model):
            vectors = provider.embed_audio_files(["track.flac"], MODEL_MUSIC_FLAMINGO_AUDIO)

        self.assertEqual(fake_model.last_path, "track.flac")
        self.assertEqual(len(vectors), 1)
        self.assertAlmostEqual(vectors[0][0], 0.707106, places=5)
        self.assertAlmostEqual(vectors[0][1], 0.707106, places=5)


class MilvusEmbeddingStoreTests(unittest.TestCase):
    def test_reset_for_run_drops_legacy_collections_when_clearing(self) -> None:
        backend = _FakeBackend()
        store = MilvusEmbeddingStore(
            backend=backend,
            dimensions={MODEL_MUSIC_FLAMINGO_AUDIO: 4, MODEL_MUQ_AUDIO: 3, MODEL_MUQ_MULAN: 2},
        )

        store.reset_for_run(["music_flamingo_audio", "description"], clear_existing=True)

        self.assertEqual(
            set(backend.dropped),
            {COLLECTION_MUSIC_FLAMINGO_AUDIO, COLLECTION_MUQ_MULAN, *LEGACY_COLLECTIONS},
        )
        self.assertEqual(
            set(backend.ensured),
            {(COLLECTION_MUSIC_FLAMINGO_AUDIO, 4), (COLLECTION_MUQ_MULAN, 2)},
        )

    def test_upsert_embeddings_validates_dimension(self) -> None:
        backend = _FakeBackend()
        store = MilvusEmbeddingStore(
            backend=backend,
            dimensions={MODEL_MUSIC_FLAMINGO_AUDIO: 4, MODEL_MUQ_AUDIO: 3, MODEL_MUQ_MULAN: 2},
        )

        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            store.upsert_embeddings(
                MODEL_MUQ_MULAN,
                [EmbeddingRow(name="track-1", embedding=[1.0, 2.0, 3.0], model_id=MODEL_MUQ_MULAN)],
            )

    def test_existing_names_returns_subset_for_requested_model(self) -> None:
        backend = _FakeBackend()
        backend.existing_by_collection[COLLECTION_MUQ_AUDIO] = {"track-1", "track-3"}
        store = MilvusEmbeddingStore(
            backend=backend,
            dimensions={MODEL_MUSIC_FLAMINGO_AUDIO: 4, MODEL_MUQ_AUDIO: 3, MODEL_MUQ_MULAN: 2},
        )

        existing = store.existing_names(MODEL_MUQ_AUDIO, ["track-1", "track-2", "track-3"])

        self.assertEqual(existing, {"track-1", "track-3"})
        self.assertEqual(backend.queries, [(COLLECTION_MUQ_AUDIO, ["track-1", "track-2", "track-3"])])


class BatchJobManagerTests(unittest.TestCase):
    def test_batch_run_processes_requested_models(self) -> None:
        tracks = [
            TrackInfo(id="1", path="a.flac", title="Song A", artist="Artist A", full_path="/music/a.flac"),
            TrackInfo(id="2", path="b.flac", title="Song B", artist="Artist B", full_path="/music/b.flac"),
        ]
        provider = _FakeProvider()
        backend = _FakeBackend()
        store = MilvusEmbeddingStore(
            backend=backend,
            dimensions={MODEL_MUSIC_FLAMINGO_AUDIO: 4, MODEL_MUQ_AUDIO: 3, MODEL_MUQ_MULAN: 2},
        )
        manager = BatchJobManager(track_loader=lambda: tracks, provider=provider, store=store)

        started = manager.start(models=["music_flamingo_audio", "audio", "qwen8b"], clear_existing=True)
        self.assertEqual(started["status"], "started")

        manager._thread.join(timeout=5.0)  # type: ignore[union-attr]
        progress = manager.progress()

        self.assertEqual(progress["status"], "completed")
        self.assertEqual(progress["processed_tracks"], 2)
        self.assertEqual(progress["failed_tracks"], 0)
        self.assertEqual(progress["total_operations"], 6)
        self.assertEqual(progress["processed_operations"], 6)
        self.assertEqual(len(provider.audio_calls), 6)
        self.assertEqual(
            {name for name, _ in backend.upserts},
            {COLLECTION_MUSIC_FLAMINGO_AUDIO, COLLECTION_MUQ_AUDIO, COLLECTION_MUQ_MULAN},
        )

    def test_cancel_marks_job_cancelled(self) -> None:
        tracks = [
            TrackInfo(id="1", path="a.flac", title="Song A", artist="Artist A", full_path="/music/a.flac"),
            TrackInfo(id="2", path="b.flac", title="Song B", artist="Artist B", full_path="/music/b.flac"),
            TrackInfo(id="3", path="c.flac", title="Song C", artist="Artist C", full_path="/music/c.flac"),
        ]
        provider = _SlowProvider()
        backend = _FakeBackend()
        store = MilvusEmbeddingStore(
            backend=backend,
            dimensions={MODEL_MUSIC_FLAMINGO_AUDIO: 4, MODEL_MUQ_AUDIO: 3, MODEL_MUQ_MULAN: 2},
        )
        manager = BatchJobManager(track_loader=lambda: tracks, provider=provider, store=store)

        manager.start(models=["muq_audio"], clear_existing=False)
        self.assertTrue(provider.started.wait(1.0))
        cancelling = manager.cancel()
        self.assertEqual(cancelling["status"], "cancelling")
        provider.release.set()

        deadline = time.time() + 2.0
        progress = manager.progress()
        while progress["status"] in {"started", "running", "cancelling"} and time.time() < deadline:
            time.sleep(0.01)
            progress = manager.progress()

        self.assertEqual(progress["status"], "cancelled")


class MuQServiceTests(unittest.TestCase):
    def test_post_embeddings_returns_openai_compatible_payload(self) -> None:
        provider = _FakeProvider()
        service = MuQService(provider=provider, batch_manager=_FakeBatchManager())

        status, payload = service.post_embeddings(
            {
                "input": ["first prompt", "second prompt"],
                "model": "qwen8b",
                "dimensions": 2,
            }
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["model"], MODEL_MUQ_MULAN)
        self.assertEqual(provider.text_calls, [(["first prompt", "second prompt"], MODEL_MUQ_MULAN, 2)])
        self.assertEqual([item["index"] for item in payload["data"]], [0, 1])
        self.assertEqual(payload["data"][0]["embedding"], [1.0, 2.0])
        self.assertEqual(payload["data"][1]["embedding"], [2.0, 3.0])

    def test_post_batch_start_passthroughs_payload_to_batch_manager(self) -> None:
        manager = _FakeBatchManager()
        service = MuQService(provider=_FakeProvider(), batch_manager=manager)

        status, payload = service.post_batch_start(
            {
                "models": ["flamingo", "lyrics"],
                "clearExisting": True,
            }
        )

        self.assertEqual(status, 200)
        self.assertEqual(manager.calls, [(["flamingo", "lyrics"], True)])
        self.assertEqual(payload["status"], "started")


if __name__ == "__main__":
    unittest.main()
