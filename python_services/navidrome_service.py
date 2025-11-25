"""
Unified Navidrome Python service exposing:
- Text embedding endpoints
- Recommendation endpoints
- Audio embedding/upload endpoint (formerly Unix socket)

Run with:
    python3 navidrome_service.py
or via uvicorn:
    uvicorn navidrome_service:app --host 0.0.0.0 --port 9002
"""

import logging
import os
from fastapi import FastAPI

from python_embed_server import EmbedSocketServer, build_embed_router
from recommender_api import build_engine, build_recommender_router
from text_embedding_service import build_text_embedding_router, text_service


def _service_port() -> int:
    """Resolve the port for the unified service with backwards compatibility."""

    for key in ("NAVIDROME_SERVICE_PORT", "NAVIDROME_RECOMMENDER_PORT", "TEXT_EMBEDDING_PORT"):
        val = os.getenv(key)
        if val:
            try:
                return int(val)
            except ValueError:
                continue
    return 9002


def create_app() -> FastAPI:
    app = FastAPI(
        title="Navidrome AI Service",
        version="2.0.0",
        description="Combined service for embeddings, recommendations, and uploads.",
    )

    # Shared engines instantiated once
    embed_server = EmbedSocketServer()
    recommender_engine = build_engine()

    app.include_router(build_text_embedding_router(text_service))
    app.include_router(build_recommender_router(recommender_engine))
    app.include_router(build_embed_router(embed_server))

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


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import uvicorn

    port = _service_port()
    uvicorn.run(app, host="0.0.0.0", port=port)
