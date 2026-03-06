from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from python_services.muq_milvus import MilvusEmbeddingStore
from python_services.muq_pipeline import BatchJobManager
from python_services.muq_provider import MuQProvider, MuQProviderConfig
from python_services.muq_shared import (
    MODEL_MUQ_MULAN,
    default_database_path,
    default_milvus_uri,
    default_music_dir,
    get_all_music,
    normalize_model_name,
)


class ServiceError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status_code = status_code


class MuQService:
    def __init__(self, *, provider: MuQProvider, batch_manager: BatchJobManager) -> None:
        self.provider = provider
        self.batch_manager = batch_manager

    def post_embeddings(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        raw_input = payload.get("input")
        if isinstance(raw_input, str):
            texts = [raw_input]
        elif isinstance(raw_input, list) and all(isinstance(item, str) for item in raw_input):
            texts = list(raw_input)
        else:
            raise ServiceError("input must be a string or string array")

        requested_model = payload.get("model", MODEL_MUQ_MULAN)
        model = normalize_model_name(str(requested_model))
        if model != MODEL_MUQ_MULAN:
            raise ServiceError("text embeddings are only available for muq_mulan")

        dimensions = int(payload.get("dimensions") or 0)
        embeddings = self.provider.embed_texts(texts, model=model, dimensions=dimensions)
        response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": embedding,
                }
                for index, embedding in enumerate(embeddings)
            ],
            "model": model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
        return HTTPStatus.OK, response

    def post_batch_start(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        raw_models = payload.get("models", [])
        if raw_models is None:
            raw_models = []
        if not isinstance(raw_models, list) or not all(isinstance(item, str) for item in raw_models):
            raise ServiceError("models must be an array of strings")
        clear_existing = bool(payload.get("clearExisting", False))
        return HTTPStatus.OK, self.batch_manager.start(models=raw_models, clear_existing=clear_existing)

    def get_batch_progress(self) -> tuple[int, dict[str, Any]]:
        return HTTPStatus.OK, self.batch_manager.progress()

    def post_batch_cancel(self) -> tuple[int, dict[str, Any]]:
        return HTTPStatus.OK, self.batch_manager.cancel()


class MuQRequestHandler(BaseHTTPRequestHandler):
    server_version = "MuQService/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/batch/progress":
            self._dispatch(self.server.service.get_batch_progress)  # type: ignore[attr-defined]
            return
        self._write_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/v1/embeddings":
            self._dispatch(self.server.service.post_embeddings, self._read_json_body())  # type: ignore[attr-defined]
            return
        if self.path == "/batch/start":
            self._dispatch(self.server.service.post_batch_start, self._read_json_body())  # type: ignore[attr-defined]
            return
        if self.path == "/batch/cancel":
            self._dispatch(self.server.service.post_batch_cancel)  # type: ignore[attr-defined]
            return
        self._write_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, format: str, *args: object) -> None:
        logging.info("%s - %s", self.address_string(), format % args)

    def _dispatch(self, handler, *args: object) -> None:
        try:
            status, body = handler(*args)
        except ServiceError as exc:
            self._write_error(exc.status_code, str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            logging.exception("MuQ service request failed")
            self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        self._write_json(status, body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ServiceError("invalid JSON payload") from exc
        if not isinstance(payload, dict):
            raise ServiceError("JSON payload must be an object")
        return payload

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_error(self, status: int, message: str) -> None:
        self._write_json(
            status,
            {"error": {"message": message, "type": "invalid_request_error", "code": status}},
        )


def create_default_service(*, db_path: str, music_dir: str, milvus_uri: str) -> MuQService:
    provider = MuQProvider(MuQProviderConfig.from_env())
    store = MilvusEmbeddingStore(uri=milvus_uri)
    batch_manager = BatchJobManager(
        track_loader=lambda: get_all_music(db_path, music_dir=music_dir),
        provider=provider,
        store=store,
    )
    return MuQService(provider=provider, batch_manager=batch_manager)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuQ embedding service for Navidrome")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9002, type=int)
    parser.add_argument("--db-path", default=default_database_path())
    parser.add_argument("--music-dir", default=default_music_dir())
    parser.add_argument("--milvus-uri", default=default_milvus_uri())
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    service = create_default_service(db_path=args.db_path, music_dir=args.music_dir, milvus_uri=args.milvus_uri)
    server = ThreadingHTTPServer((args.host, args.port), MuQRequestHandler)
    server.service = service  # type: ignore[attr-defined]
    logging.info("Starting MuQ service on http://%s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
