import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import recommender_api
import batch_embedding_job
from schemas import RecommendationRequest, RecommendationSeed


class DummyResolver:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def name_to_id(self, name: str):
        return self.mapping.get(name)


class DummySearcher:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def get_embeddings_by_name(self, names, model=None):
        return {name: self.embeddings.get(name) for name in names}


def test_env_flag_parsing(monkeypatch):
    monkeypatch.setenv("NAVIDROME_FLAG", "true")
    assert recommender_api._env_flag("NAVIDROME_FLAG") is True
    monkeypatch.setenv("NAVIDROME_FLAG", "0")
    assert recommender_api._env_flag("NAVIDROME_FLAG") is False


def test_resolve_path_uses_env(monkeypatch, tmp_path):
    custom = tmp_path / "custom.db"
    monkeypatch.setenv("NAVIDROME_DB_PATH", str(custom))
    assert recommender_api._resolve_path("NAVIDROME_DB_PATH", "navidrome.db") == custom


def test_process_hits_applies_diversity_and_exclusions():
    resolver = DummyResolver({"Artist - Skip": "skip"})
    engine = recommender_api.RecommendationEngine(
        searcher=None, name_resolver=resolver, debug_logging=False
    )
    seed = RecommendationSeed(track_id="seed", weight=1.0, source="recent")
    hits = [
        {"name": "Artist - Keep", "distance": 0.8, "track_id": "keep", "models": ["muq"]},
        {"name": "Artist - Skip", "distance": 0.9},  # resolved via resolver
    ]

    candidate_scores = {}
    candidate_reason = {}
    candidate_models = {}

    engine._process_hits(
        hits=hits,
        seed=seed,
        exclude_ids={"skip"},
        diversity=0.1,
        candidate_scores=candidate_scores,
        candidate_reason=candidate_reason,
        candidate_models=candidate_models,
    )

    assert "keep" in candidate_scores
    assert candidate_scores["keep"] == pytest.approx(0.8 * 0.9)
    assert candidate_reason["keep"] == "seed:recent"
    assert candidate_models["keep"] == ["muq"]


def test_apply_negative_prompt_penalty_reduces_scores():
    embeddings = {"Artist - Track": np.array([1.0, 0.0])}
    searcher = DummySearcher(embeddings)
    resolver = DummyResolver()
    engine = recommender_api.RecommendationEngine(
        searcher=searcher, name_resolver=resolver
    )

    request = RecommendationRequest(
        user_id="u1",
        user_name="User",
        limit=10,
        mode="recent",
        seeds=[],
        models=["muq"],
        negative_prompts=["avoid"],
        negative_embeddings={"muq": [[1.0, 0.0]]},
        negative_prompt_penalty=0.8,
    )

    scores = {"t1": 1.0}
    names = {"t1": "Artist - Track"}

    negative = engine._apply_negative_prompt_penalty(scores, names, request)

    assert negative["t1"] == pytest.approx(1.0)
    assert scores["t1"] == pytest.approx(0.8)


def test_apply_negative_prompt_penalty_no_prompts_noop():
    searcher = DummySearcher({})
    engine = recommender_api.RecommendationEngine(
        searcher=searcher, name_resolver=DummyResolver()
    )
    request = RecommendationRequest(
        user_id="u1",
        user_name="User",
        limit=10,
        mode="recent",
        seeds=[],
        models=["muq"],
        negative_prompts=[],
        negative_embeddings=None,
    )

    scores = {"t1": 1.0}
    names = {"t1": "Artist - Track"}

    negative = engine._apply_negative_prompt_penalty(scores, names, request)

    assert negative == {}
    assert scores["t1"] == pytest.approx(1.0)


def test_playlist_endpoint_uses_engine():
    class DummyEngine:
        def recommend(self, request):
            return recommender_api.RecommendationResponse(tracks=[])

    app = FastAPI()
    app.include_router(recommender_api.build_recommender_router(DummyEngine()))
    client = TestClient(app)

    payload = {
        "user_id": "u1",
        "user_name": "User",
        "limit": 5,
        "mode": "recent",
        "seeds": [],
        "models": ["muq"],
    }
    response = client.post("/playlist/recent", json=payload)

    assert response.status_code == 200
    assert response.json()["tracks"] == []


def test_batch_endpoints(monkeypatch, tmp_path):
    db_path = tmp_path / "navidrome.db"
    db_path.write_text("")
    music_root = tmp_path / "music"
    music_root.mkdir()

    monkeypatch.setenv("NAVIDROME_DB_PATH", str(db_path))
    monkeypatch.setenv("NAVIDROME_MUSIC_ROOT", str(music_root))

    class DummyProgress:
        def __init__(self):
            self.total_tracks = 1
            self.total_operations = 1
            self.processed_tracks = 0
            self.processed_operations = 0
            self.failed_tracks = 0
            self.current_track = None
            self.current_model = None
            self.status = "running"
            self.estimated_completion = None
            self.last_error = None

    class DummyJob:
        def __init__(self):
            self.progress = DummyProgress()
            self.cancelled = False

        def run(self, *_args, **_kwargs):
            pass

        def get_progress(self):
            return self.progress

        def cancel(self):
            self.cancelled = True

    state = {"job": None}

    def start_job(**_kw):
        state["job"] = DummyJob()
        return state["job"]

    def get_job():
        return state["job"]

    monkeypatch.setattr(batch_embedding_job, "start_batch_job", start_job)
    monkeypatch.setattr(batch_embedding_job, "get_current_job", get_job)

    class DummyEngine:
        def recommend(self, request):
            return recommender_api.RecommendationResponse(tracks=[])

    app = FastAPI()
    app.include_router(recommender_api.build_recommender_router(DummyEngine()))
    client = TestClient(app)

    response = client.post("/batch/start", json={"models": ["muq"], "clearExisting": True})
    assert response.status_code == 200

    progress = client.get("/batch/progress")
    assert progress.status_code == 200
    assert progress.json()["status"] == "running"

    cancel = client.post("/batch/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["status"] == "cancelling"
