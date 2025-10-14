"""
Utilities for applying upload settings to embedding payloads.

This module centralizes all upload-related features for the embedding server,
keeping python_embed_server.py focused on socket orchestration and model logic.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
from urllib import error, request

LOGGER = logging.getLogger("navidrome.upload_features")

_TRUTHY_STRINGS: Set[str] = {"1", "true", "yes", "on"}
_REASONING_LEVELS: Set[str] = {"none", "low", "medium", "high", "default"}


@dataclass
class UploadSettings:
    rename_enabled: bool = False
    renaming_prompt: str = ""
    openai_endpoint: str = ""
    openai_model: str = ""
    similarity_search_enabled: bool = False
    dedup_threshold: float = 0.85
    reasoning_level: str = "default"

    @classmethod
    def from_payload(cls, payload: Optional[dict]) -> "UploadSettings":
        if not isinstance(payload, dict):
            return cls()

        defaults = cls()

        def _coerce_string(value: Any) -> str:
            if value is None:
                return ""
            return str(value).strip()

        def _coerce_bool(value: Any) -> bool:
            if isinstance(value, str):
                return value.strip().lower() in _TRUTHY_STRINGS
            return bool(value)

        raw_threshold = payload.get("dedupThreshold", defaults.dedup_threshold)
        try:
            dedup_threshold = float(raw_threshold)
        except (TypeError, ValueError):
            dedup_threshold = defaults.dedup_threshold
        else:
            dedup_threshold = min(max(dedup_threshold, 0.0), 1.0)

        raw_reasoning = payload.get("reasoningLevel", defaults.reasoning_level)
        reasoning = cls._normalize_reasoning(raw_reasoning, defaults.reasoning_level)

        return cls(
            rename_enabled=_coerce_bool(payload.get("renameEnabled")),
            renaming_prompt=_coerce_string(payload.get("renamingPrompt")),
            openai_endpoint=_coerce_string(payload.get("openAiEndpoint")),
            openai_model=_coerce_string(payload.get("openAiModel")),
            similarity_search_enabled=_coerce_bool(
                payload.get("similaritySearchEnabled")
            ),
            dedup_threshold=dedup_threshold,
            reasoning_level=reasoning,
        )

    @staticmethod
    def _normalize_reasoning(value: Any, default: str) -> str:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return default
            if normalized in _REASONING_LEVELS:
                return normalized
            if normalized in {"off", "disabled"}:
                return "none"
            if normalized in {"standard", "normal"}:
                return "default"
        return default


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
        self.model = model or "gpt-4o-mini"
        base_endpoint = endpoint.strip() or "https://api.openai.com"
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
        self.api_key = (
            os.getenv("NAVIDROME_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        if not self.api_key:
            self.logger.warning(
                "OpenAI API key not found in NAVIDROME_OPENAI_API_KEY or OPENAI_API_KEY; "
                "automatic renaming will be skipped."
            )

    def rename_segments(
        self, segments: Sequence[dict], system_prompt: str
    ) -> Dict[int, str]:
        if not segments:
            self.logger.debug("No segments supplied for renaming; skipping request.")
            return {}

        if not self.api_key:
            return {}

        prompt = system_prompt.strip() if system_prompt else ""
        context_lines = []
        for segment in segments:
            index = segment.get("index")
            title = segment.get("title") or f"Track {index}"
            duration = segment.get("duration_seconds")
            duration_text = ""
            if isinstance(duration, (int, float)) and duration > 0:
                duration_text = f" ({duration:.0f}s)"
            context_lines.append(f"{index}: {title}{duration_text}")

        reasoning_text = ""
        if self.reasoning_level == "none":
            reasoning_text = (
                "The caller does not want additional reasoning beyond following the instructions."
            )
        elif self.reasoning_level in {"low", "medium", "high"}:
            reasoning_text = (
                f"Apply approximately {self.reasoning_level} reasoning effort while crafting the names."
            )
        else:
            reasoning_text = "Use the model's default reasoning behaviour."

        system_messages = [
            "You are helping rename uploaded music track segments.",
            "Respond ONLY with compact JSON following this schema: "
            '{"tracks": [{"index": number, "title": string}]}.',
            reasoning_text,
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
            "max_tokens": 400,
        }

        response = self._perform_request(request_body)
        if not response:
            return {}

        choices = response.get("choices")
        if not choices:
            self.logger.warning("OpenAI response contained no choices: %s", response)
            return {}

        first_choice = choices[0]
        message = first_choice.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            self.logger.warning("Unexpected OpenAI response content field: %s", content)
            return {}

        parsed = self._parse_response_content(content)
        if not parsed:
            self.logger.warning("Unable to parse OpenAI rename response as JSON.")
            return {}

        tracks = parsed.get("tracks")
        if not isinstance(tracks, list):
            self.logger.warning("OpenAI rename response missing 'tracks' array.")
            return {}

        rename_map: Dict[int, str] = {}
        for item in tracks:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            title = item.get("title")
            if not isinstance(title, str):
                continue
            cleaned = title.strip()
            if not cleaned:
                continue
            rename_map[index] = cleaned

        missing = {segment.get("index") for segment in segments} - rename_map.keys()
        if missing:
            self.logger.debug(
                "OpenAI rename response omitted %d tracks; original titles retained.",
                len(missing),
            )
        return rename_map

    def _perform_request(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
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

        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
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
        self, embedding_payload: Dict[str, Any], threshold: float
    ) -> List[int]:
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

    def apply(self, embedding_payload: Dict[str, Any], settings: UploadSettings) -> Dict[str, Any]:
        if not isinstance(embedding_payload, dict):
            raise TypeError("embedding_payload must be a dictionary.")

        segments = embedding_payload.get("segments")
        if not isinstance(segments, list):
            self.logger.debug("Embedding payload missing segments array; skipping.")
            return embedding_payload

        if settings.rename_enabled:
            rename_count = self._apply_renaming(segments, settings)
            if rename_count:
                self.logger.info("Renamed %d segments via OpenAI.", rename_count)

        if settings.similarity_search_enabled:
            removed = self._apply_similarity_filter(embedding_payload, settings)
            if removed:
                self.logger.info("Removed %d duplicate segments.", removed)

        return embedding_payload

    def _apply_renaming(
        self, segments: List[dict], settings: UploadSettings
    ) -> int:
        renamer = OpenAiRenamer(
            endpoint=settings.openai_endpoint,
            model=settings.openai_model,
            reasoning_level=settings.reasoning_level,
            logger=self.logger,
        )

        rename_map = renamer.rename_segments(segments, settings.renaming_prompt)
        if not rename_map:
            return 0

        updated = 0
        for segment in segments:
            try:
                index = int(segment.get("index"))
            except (TypeError, ValueError):
                continue
            new_title = rename_map.get(index)
            if not new_title:
                continue
            original_title = segment.get("title")
            if original_title == new_title:
                continue
            if original_title and "original_title" not in segment:
                segment["original_title"] = original_title
            segment["title"] = new_title
            segment["generated_title"] = new_title
            updated += 1

        return updated

    def _apply_similarity_filter(
        self, embedding_payload: Dict[str, Any], settings: UploadSettings
    ) -> int:
        duplicates = self.similarity_searcher.identify_duplicates(
            embedding_payload, settings.dedup_threshold
        )
        if not duplicates:
            return 0

        segments = embedding_payload.get("segments")
        if not isinstance(segments, list):
            return 0

        duplicate_indexes: Set[int] = set()
        for value in duplicates:
            try:
                duplicate_indexes.add(int(value))
            except (TypeError, ValueError):
                continue

        if not duplicate_indexes:
            return 0

        original_count = len(segments)
        embedding_payload["segments"] = [
            segment
            for segment in segments
            if int(segment.get("index", -1)) not in duplicate_indexes
        ]
        return original_count - len(embedding_payload["segments"])
