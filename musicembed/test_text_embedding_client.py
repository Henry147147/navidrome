#!/usr/bin/env python3
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_embedding_client import LlamaCppEmbeddingClient


class _DummyResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class LlamaCppEmbeddingClientTests(unittest.TestCase):
    def test_success_openai_response_reorders_by_index_and_normalizes(self) -> None:
        payload = {
            "data": [
                {"index": 1, "embedding": [0.0, 5.0, 0.0]},
                {"index": 0, "embedding": [3.0, 4.0, 0.0]},
            ]
        }

        with patch("text_embedding_client.request.urlopen", return_value=_DummyResponse(payload)):
            client = LlamaCppEmbeddingClient("http://127.0.0.1:9002")
            vectors = client.embed_documents(["first", "second"], dimensions=3)

        self.assertEqual(len(vectors), 2)
        self.assertAlmostEqual(vectors[0][0], 0.6, places=6)
        self.assertAlmostEqual(vectors[0][1], 0.8, places=6)
        self.assertAlmostEqual(vectors[0][2], 0.0, places=6)
        self.assertAlmostEqual(vectors[1][0], 0.0, places=6)
        self.assertAlmostEqual(vectors[1][1], 1.0, places=6)
        self.assertAlmostEqual(vectors[1][2], 0.0, places=6)

    def test_sends_dimensions_in_request_payload(self) -> None:
        payload = {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

        with patch("text_embedding_client.request.urlopen", return_value=_DummyResponse(payload)) as mock_urlopen:
            client = LlamaCppEmbeddingClient("http://127.0.0.1:9002")
            client.embed_documents(["hello"], dimensions=2)

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["input"], ["hello"])
        self.assertEqual(body["encoding_format"], "float")
        self.assertEqual(body["dimensions"], 2)

    def test_legacy_embedding_fallback_parses(self) -> None:
        payload = {"embedding": [10.0, 0.0]}

        with patch("text_embedding_client.request.urlopen", return_value=_DummyResponse(payload)):
            client = LlamaCppEmbeddingClient("http://127.0.0.1:9002")
            vectors = client.embed_documents(["legacy"], dimensions=2)

        self.assertEqual(len(vectors), 1)
        self.assertAlmostEqual(vectors[0][0], 1.0, places=6)
        self.assertAlmostEqual(vectors[0][1], 0.0, places=6)

    def test_non_200_json_error_surfaces_message(self) -> None:
        error_body = io.BytesIO(json.dumps({"error": {"message": "invalid input"}}).encode("utf-8"))
        http_error = HTTPError(
            url="http://127.0.0.1:9002/v1/embeddings",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=error_body,
        )

        with patch("text_embedding_client.request.urlopen", side_effect=http_error):
            client = LlamaCppEmbeddingClient("http://127.0.0.1:9002")
            with self.assertRaisesRegex(RuntimeError, "invalid input"):
                client.embed_documents(["bad"], dimensions=2)

    def test_non_200_non_json_error_returns_status(self) -> None:
        error_body = io.BytesIO(b"upstream unavailable")
        http_error = HTTPError(
            url="http://127.0.0.1:9002/v1/embeddings",
            code=502,
            msg="Bad Gateway",
            hdrs=None,
            fp=error_body,
        )

        with patch("text_embedding_client.request.urlopen", side_effect=http_error):
            client = LlamaCppEmbeddingClient("http://127.0.0.1:9002")
            with self.assertRaisesRegex(RuntimeError, "status 502"):
                client.embed_documents(["bad"], dimensions=2)

    def test_dimension_exact_match_is_preserved(self) -> None:
        payload = {"data": [{"index": 0, "embedding": [1.0, 2.0, 2.0]}]}

        with patch("text_embedding_client.request.urlopen", return_value=_DummyResponse(payload)):
            client = LlamaCppEmbeddingClient("http://127.0.0.1:9002")
            vectors = client.embed_documents(["exact"], dimensions=3)

        self.assertEqual(len(vectors[0]), 3)

    def test_dimension_larger_vector_is_truncated(self) -> None:
        payload = {"data": [{"index": 0, "embedding": [1.0, 2.0, 2.0, 100.0]}]}

        with patch("text_embedding_client.request.urlopen", return_value=_DummyResponse(payload)):
            client = LlamaCppEmbeddingClient("http://127.0.0.1:9002")
            vectors = client.embed_documents(["truncate"], dimensions=3)

        self.assertEqual(len(vectors[0]), 3)

    def test_dimension_smaller_vector_errors(self) -> None:
        payload = {"data": [{"index": 0, "embedding": [1.0, 2.0]}]}

        with patch("text_embedding_client.request.urlopen", return_value=_DummyResponse(payload)):
            client = LlamaCppEmbeddingClient("http://127.0.0.1:9002")
            with self.assertRaisesRegex(RuntimeError, "too small"):
                client.embed_documents(["small"], dimensions=3)

    def test_l2_normalization_is_applied(self) -> None:
        payload = {"data": [{"index": 0, "embedding": [3.0, 4.0]}]}

        with patch("text_embedding_client.request.urlopen", return_value=_DummyResponse(payload)):
            client = LlamaCppEmbeddingClient("http://127.0.0.1:9002")
            vectors = client.embed_documents(["norm"], dimensions=2)

        self.assertAlmostEqual(vectors[0][0], 0.6, places=6)
        self.assertAlmostEqual(vectors[0][1], 0.8, places=6)

    def test_empty_input_returns_empty_without_http_call(self) -> None:
        with patch("text_embedding_client.request.urlopen") as mock_urlopen:
            client = LlamaCppEmbeddingClient("http://127.0.0.1:9002")
            vectors = client.embed_documents([], dimensions=2)

        self.assertEqual(vectors, [])
        mock_urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
