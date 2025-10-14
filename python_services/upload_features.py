"""
Utilities for applying upload settings to embedding payloads.

This module centralizes all upload-related features for the embedding server,
keeping python_embed_server.py focused on socket orchestration and model logic.
"""

import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from urllib import error, request

from models import _REASONING_LEVELS, SongEmbedding, UploadSettings

LOGGER = logging.getLogger("navidrome.upload_features")


class OpenAiRenamer:
    """
    Handles communication with an OpenAI-compatible endpoint to rename tracks.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        reasoning_level: str,
        logger: logging.Logger,
        timeout: int = 45,
    ) -> None:
        self.logger = logger
        self.timeout = timeout
        self.model = model
        base_endpoint = endpoint.strip()
        base_endpoint = base_endpoint.rstrip("/")
        if base_endpoint.endswith("/v1/chat/completions"):
            self.url = base_endpoint
        elif base_endpoint.endswith("/v1"):
            self.url = f"{base_endpoint}/chat/completions"
        else:
            self.url = f"{base_endpoint}/v1/chat/completions"
        self.reasoning_level = (
            reasoning_level if reasoning_level in _REASONING_LEVELS else "default"
        )
        self.api_key = os.getenv("NAVIDROME_OPENAI_API_KEY") or os.getenv(
            "OPENAI_API_KEY"
        )
        if not self.api_key:
            self.logger.info(
                "OpenAI API key not found in NAVIDROME_OPENAI_API_KEY or OPENAI_API_KEY. "
                "Continuing without authentication headers; ensure your endpoint does not "
                "require a bearer token (LM Studio servers typically do not)."
            )

    def rename_segments(self, name: str, system_prompt: str) -> str:
        prompt = system_prompt.strip() if system_prompt else ""
        context_lines = [name]

        system_messages = [
            "You are helping rename uploaded music track segments.",
            "Respond ONLY with compact JSON following this schema: " '{"title": title}',
        ]
        if prompt:
            system_messages.append(
                f"The caller supplied additional requirements: {prompt}"
            )
        system_message = " ".join(filter(None, system_messages))
        user_message = "Here are the original tracks:\n" + "\n".join(context_lines)

        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.2,
            "max_tokens": 40000,
        }

        response = self._perform_request(request_body)
        if not response:
            return name

        choices = response.get("choices")
        if not choices:
            self.logger.warning("OpenAI response contained no choices: %s", response)
            return name

        first_choice = choices[0]
        message = first_choice.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            self.logger.warning("Unexpected OpenAI response content field: %s", content)

        parsed = self._parse_response_content(content or "{}") or {}
        new_name = parsed.get("title", name).strip()
        return new_name

    def _perform_request(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(self.url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                response_bytes = resp.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            self.logger.error(
                "OpenAI rename request failed (%s): %s", exc.code, detail.strip()
            )
            return None
        except error.URLError as exc:
            self.logger.error("OpenAI rename request errored: %s", exc)
            return None

        try:
            return json.loads(response_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self.logger.error("Failed to decode OpenAI response JSON: %s", exc)
            return None

    @staticmethod
    def _parse_response_content(content: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*}", content, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


class SimilaritySearchStub:
    """
    Placeholder implementation for similarity search integration.

    The real implementation can replace this class while keeping the pipeline
    contract unchanged.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def identify_duplicates(
        self, embedding_payload: SongEmbedding, threshold: float
    ) -> List[str]:
        print(embedding_payload)
        return []
        segments = embedding_payload.get("segments")
        if not isinstance(segments, list):
            return []
        self.logger.info(
            "Similarity search stub invoked for %d segments (threshold=%.3f). "
            "No duplicate filtering performed.",
            len(segments),
            threshold,
        )
        return []


class UploadFeaturePipeline:
    """
    Applies the configured upload features to an embedding payload.
    """

    def __init__(
        self,
        *,
        similarity_searcher: Optional[SimilaritySearchStub] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or LOGGER
        self.similarity_searcher = similarity_searcher or SimilaritySearchStub(
            self.logger
        )

    def rename(self, name, settings: UploadSettings) -> str:
        name_path = Path(name)
        name_suffix = name_path.suffix
        name = name_path.stem
        if settings.rename_enabled:
            new_name = self._apply_renaming(name, settings)
            self.logger.info(f"Renamed {name} to {new_name} via OpenAI endpoint.")
            return str(Path(new_name).with_suffix(name_suffix))
            
        return str(name_path)

    def scan_for_dups(
        self, embeddings: List[SongEmbedding], settings: UploadSettings
    ) -> List[str]:
        removed_list = []
        if not settings.similarity_search_enabled:
            return removed_list

        for embed in embeddings:
            duplicates = self._apply_similarity_filter(embed, settings)
            if duplicates:
                removed_list.append(embed.name)
                self.logger.info(
                    "Embedding %s flagged as duplicate of %s",
                    embed.name,
                    ", ".join(duplicates),
                )
        return removed_list

    def _apply_renaming(self, name: str, settings: UploadSettings) -> str:
        renamer = OpenAiRenamer(
            endpoint=settings.openai_endpoint,
            model=settings.openai_model,
            reasoning_level=settings.reasoning_level,
            logger=self.logger,
        )

        new_name = renamer.rename_segments(name, settings.renaming_prompt)
        return new_name

    def _apply_similarity_filter(
        self, embedding_payload: SongEmbedding, settings: UploadSettings
    ) -> List[str]:
        duplicates = self.similarity_searcher.identify_duplicates(
            embedding_payload,
            settings.dedup_threshold,
        )
        return duplicates
