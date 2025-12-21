"""
Unified Navidrome Python service exposing:
- Text embedding endpoints
- Recommendation endpoints
- Audio embedding via Unix socket (primary) and HTTP (legacy)

Run with:
    python3 navidrome_service.py
or via uvicorn:
    uvicorn navidrome_service:app --host 0.0.0.0 --port 9002
"""

import argparse
import logging
import os
import threading
from fastapi import FastAPI
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for type checking; runtime imports are inside create_app.
    from pymilvus import MilvusClient
    from python_embed_server import EmbedSocketServer
    from recommender_api import RecommendationEngine
    from text_embedding_service import TextEmbeddingService


def _service_port() -> int:
    """Resolve the port for the unified service with backwards compatibility."""

    for key in (
        "NAVIDROME_SERVICE_PORT",
        "NAVIDROME_RECOMMENDER_PORT",
        "TEXT_EMBEDDING_PORT",
    ):
        val = os.getenv(key)
        if val:
            try:
                return int(val)
            except ValueError:
                continue
    return 9002


def _configure_logging(verbose: bool) -> str:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return "debug" if verbose else "info"


def _resolve_milvus_uri() -> str:
    """
    Prefer a local Milvus Lite database file when NAVIDROME_MILVUS_DB_PATH is set.
    Falls back to NAVIDROME_MILVUS_URI or the default remote Milvus endpoint.
    """

    db_path = os.getenv("NAVIDROME_MILVUS_DB_PATH")
    if db_path:
        expanded = Path(db_path).expanduser().resolve()
        # Ensure parent directory exists so Milvus Lite can create the file.
        expanded.parent.mkdir(parents=True, exist_ok=True)
        # Milvus Lite expects a plain file path (no scheme). A new DB file will
        # be created automatically if it doesn't already exist.
        return str(expanded)

    return os.getenv("NAVIDROME_MILVUS_URI", "http://localhost:19530")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Navidrome AI Service",
        version="2.0.0",
        description="Combined service for embeddings and recommendations.",
    )

    milvus_uri = _resolve_milvus_uri()
    # Make sure downstream modules that still read NAVIDROME_MILVUS_URI pick up the
    # resolved value (including the local file:// form).
    os.environ["NAVIDROME_MILVUS_URI"] = milvus_uri

    # Local imports to avoid connecting to Milvus when invoked with --help.
    from pymilvus import MilvusClient
    from python_embed_server import EmbedSocketServer, build_embed_router
    from recommender_api import build_engine, build_recommender_router
    from text_embedding_service import build_text_embedding_router, text_service

    milvus_client = MilvusClient(uri=milvus_uri)

    # Shared engines instantiated once
    embed_server = EmbedSocketServer(milvus_client=milvus_client)
    recommender_engine = build_engine(
        milvus_client=milvus_client, milvus_uri=milvus_uri
    )

    app.include_router(build_text_embedding_router(text_service))
    app.include_router(build_recommender_router(recommender_engine))
    app.include_router(build_embed_router(embed_server))

    # Start Unix socket server in background thread for Go client communication
    logger = logging.getLogger("navidrome.service")
    socket_thread = threading.Thread(
        target=embed_server.serve_forever,
        daemon=True,
        name="embed-socket-server",
    )
    socket_thread.start()
    logger.info("Started embedding socket server at %s", embed_server.socket_path)

    @app.get("/health")
    async def unified_health():
        return {
            "status": "ok",
            "text_embedding": {
                "device": text_service.device,
                "use_stubs": text_service.use_stubs,
            },
            "recommender": "ready",
            "embedding": {"descriptions": embed_server.enable_descriptions},
        }

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Navidrome AI unified service")
    parser.add_argument(
        "--milvus-db-path",
        help=(
            "Path to a Milvus Lite .db file (overrides NAVIDROME_MILVUS_DB_PATH "
            "and NAVIDROME_MILVUS_URI). Useful when running with a local, "
            "copied Milvus database instead of a remote server."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args, _unknown = parser.parse_known_args()

    uvicorn_log_level = _configure_logging(args.verbose)

    if args.milvus_db_path:
        os.environ["NAVIDROME_MILVUS_DB_PATH"] = args.milvus_db_path
        # Ensure any previously-set URI is ignored in favor of the DB file.
        os.environ.pop("NAVIDROME_MILVUS_URI", None)

    # Recreate app to pick up CLI-provided environment overrides
    app = create_app()

    import uvicorn

    port = _service_port()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=uvicorn_log_level)

else:
    # When imported by uvicorn (`uvicorn navidrome_service:app`), build the app
    # immediately using environment variables.
    app = create_app()
