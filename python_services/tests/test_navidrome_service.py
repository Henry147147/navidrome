import importlib
import sys
import types

from fastapi.testclient import TestClient


def _install_dummy_modules(monkeypatch, milvus_uri_target: str):
    """
    Inject lightweight stand-ins for heavy dependencies so navidrome_service
    can be imported without talking to real services.
    """

    # Dummy pymilvus
    class DummyMilvusClient:
        last_uri = None

        def __init__(self, uri):
            DummyMilvusClient.last_uri = uri

        def load_collection(self, *_, **__):
            return None

    pymilvus = types.ModuleType("pymilvus")
    pymilvus.MilvusClient = DummyMilvusClient
    monkeypatch.setitem(sys.modules, "pymilvus", pymilvus)

    # Dummy python_embed_server
    from fastapi import APIRouter

    class DummyEmbedSocketServer:
        def __init__(self, milvus_client=None):
            self.milvus_client = milvus_client
            self.enable_descriptions = True
            self.socket_path = "/tmp/navidrome-test.sock"

        def serve_forever(self):
            return None

    def build_embed_router(_server):
        router = APIRouter()

        @router.get("/embed-dummy")
        def _dummy():
            return {"status": "ok"}

        return router

    pes = types.ModuleType("python_embed_server")
    pes.EmbedSocketServer = DummyEmbedSocketServer
    pes.build_embed_router = build_embed_router
    monkeypatch.setitem(sys.modules, "python_embed_server", pes)

    # Dummy recommender_api
    def build_engine(*_, **__):
        return object()

    def build_recommender_router(_engine):
        router = APIRouter()

        @router.get("/recommender-dummy")
        def _dummy():
            return {"status": "ok"}

        return router

    rec_mod = types.ModuleType("recommender_api")
    rec_mod.build_engine = build_engine
    rec_mod.build_recommender_router = build_recommender_router
    monkeypatch.setitem(sys.modules, "recommender_api", rec_mod)

    # Dummy text_embedding_service
    class DummyService:
        device = "cpu"
        use_stubs = True

    def build_text_embedding_router(_service):
        router = APIRouter()

        @router.get("/text-dummy")
        def _dummy():
            return {"status": "ok"}

        return router

    text_mod = types.ModuleType("text_embedding_service")
    text_mod.text_service = DummyService()
    text_mod.build_text_embedding_router = build_text_embedding_router
    monkeypatch.setitem(sys.modules, "text_embedding_service", text_mod)

    # Ensure environment is set before import
    monkeypatch.setenv("NAVIDROME_MILVUS_DB_PATH", milvus_uri_target)
    monkeypatch.delenv("NAVIDROME_MILVUS_URI", raising=False)


def test_create_app_uses_local_milvus_path(monkeypatch, tmp_path):
    # Prepare dummy dependencies and env
    db_path = tmp_path / "milvus.db"
    _install_dummy_modules(monkeypatch, str(db_path))

    # Force reload after inserting dummy modules
    sys.modules.pop("navidrome_service", None)
    navidrome_service = importlib.import_module("navidrome_service")

    app = navidrome_service.create_app()
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["text_embedding"]["device"] == "cpu"
    assert payload["embedding"]["descriptions"] is True

    # Verify Milvus client saw the file-based URI
    dummy_client = sys.modules["pymilvus"].MilvusClient
    assert dummy_client.last_uri == str(db_path)
