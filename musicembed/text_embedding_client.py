import json
import math
from typing import Any, List, Sequence
from urllib import error, request


class MuQEmbeddingClient:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        normalized_base_url = base_url.strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("text embedding base URL is required")
        if timeout_seconds <= 0:
            raise ValueError("text embedding timeout must be > 0")

        self.base_url = normalized_base_url
        self.endpoint = f"{normalized_base_url}/v1/embeddings"
        self.timeout_seconds = float(timeout_seconds)

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        model: str = "muq_mulan",
        dimensions: int = 0,
    ) -> List[List[float]]:
        if not texts:
            return []

        payload: dict[str, Any] = {
            "input": list(texts),
            "encoding_format": "float",
            "model": model,
        }
        if dimensions > 0:
            payload["dimensions"] = dimensions

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw_data = resp.read()
        except error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            message = self._extract_error_message(body_text)
            if message:
                raise RuntimeError(f"text embedding service returned status {exc.code}: {message}") from exc
            raise RuntimeError(f"text embedding service returned status {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"call embeddings endpoint: {exc.reason}") from exc

        try:
            parsed = json.loads(raw_data.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("text embedding response was not valid JSON") from exc

        service_error = self._extract_error_message(parsed)
        if service_error:
            raise RuntimeError(f"text embedding service returned error: {service_error}")

        embeddings = self._parse_embeddings(parsed)
        if not embeddings:
            raise RuntimeError("text embedding response did not contain an embedding")
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"text embedding response size mismatch: expected {len(texts)} got {len(embeddings)}"
            )

        normalized_embeddings: List[List[float]] = []
        for embedding in embeddings:
            adjusted = self._normalize_embedding_dimension(embedding, dimensions)
            normalized_embeddings.append(self._l2_normalize(adjusted))
        return normalized_embeddings

    @staticmethod
    def _parse_embeddings(payload: Any) -> List[List[float]]:
        if not isinstance(payload, dict):
            return []

        data = payload.get("data")
        if isinstance(data, list) and data:
            entries: List[tuple[int, int, List[float]]] = []
            indexed_count = 0
            for pos, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                embedding = item.get("embedding")
                if not isinstance(embedding, list) or not embedding:
                    continue
                try:
                    vector = [float(value) for value in embedding]
                except (TypeError, ValueError):
                    continue

                entry_index = item.get("index")
                if isinstance(entry_index, int):
                    entries.append((entry_index, pos, vector))
                    indexed_count += 1
                else:
                    entries.append((pos, pos, vector))

            if not entries:
                return []

            if indexed_count == len(entries):
                entries.sort(key=lambda item: (item[0], item[1]))
            else:
                entries.sort(key=lambda item: item[1])
            return [item[2] for item in entries]

        embedding = payload.get("embedding")
        if isinstance(embedding, list) and embedding:
            try:
                return [[float(value) for value in embedding]]
            except (TypeError, ValueError):
                return []

        return []

    @staticmethod
    def _normalize_embedding_dimension(embedding: Sequence[float], desired_dim: int) -> List[float]:
        vector = [float(value) for value in embedding]
        if desired_dim <= 0:
            return vector
        if len(vector) == desired_dim:
            return vector
        if len(vector) > desired_dim:
            return vector[:desired_dim]
        raise RuntimeError(f"embedding dimension too small: expected {desired_dim} got {len(vector)}")

    @staticmethod
    def _l2_normalize(embedding: Sequence[float]) -> List[float]:
        vector = [float(value) for value in embedding]
        norm_sq = sum(value * value for value in vector)
        if norm_sq <= 0:
            return vector
        inv_norm = 1.0 / math.sqrt(norm_sq)
        return [value * inv_norm for value in vector]

    @staticmethod
    def _extract_error_message(payload: Any) -> str:
        parsed: Any = payload
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                return ""

        if not isinstance(parsed, dict):
            return ""

        err = parsed.get("error")
        if isinstance(err, dict):
            message = err.get("message")
            if isinstance(message, str):
                return message.strip()
        return ""


LlamaCppEmbeddingClient = MuQEmbeddingClient
