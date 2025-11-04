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
from database_query import MilvusSimilaritySearcher

LOGGER = logging.getLogger("navidrome.upload_features")


def best_effort_parse_metadata(path: Optional[str]) -> Dict[str, Any]:
    """
    Attempt to extract useful audio metadata without failing loudly.
    Returns a dictionary with normalized keys if any metadata is available.
    """

    def _to_text_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            collected: List[str] = []
            for item in value:
                collected.extend(_to_text_list(item))
            return collected
        if hasattr(value, "text"):
            try:
                return _to_text_list(getattr(value, "text"))
            except Exception:
                return []
        if hasattr(value, "value"):
            try:
                return _to_text_list(getattr(value, "value"))
            except Exception:
                return []
        if isinstance(value, bytes):
            try:
                return [value.decode("utf-8")]
            except UnicodeDecodeError:
                return [value.decode("latin-1", errors="ignore")]
        text = str(value).strip()
        return [text] if text else []

    metadata: Dict[str, Any] = {}
    if not path:
        return metadata

    try:
        file_path = Path(path)
    except (TypeError, ValueError):
        return metadata

    if not file_path.exists():
        LOGGER.debug("Metadata parse skipped; file does not exist: %s", path)
        return metadata

    try:
        from mutagen import File as MutagenFile  # type: ignore
    except ImportError:
        LOGGER.debug(
            "mutagen library not available; cannot parse metadata for %s", path
        )
        return metadata

    try:
        audio_file = MutagenFile(str(file_path))
    except Exception as exc:
        LOGGER.debug("mutagen failed to read %s: %s", path, exc)
        return metadata

    if audio_file is None:
        LOGGER.debug("mutagen returned no data for %s", path)
        return metadata

    tags = getattr(audio_file, "tags", None)
    if not tags:
        return metadata

    normalized: Dict[str, List[str]] = {}
    try:
        items = tags.items()
    except Exception:
        items = []

    for key, value in items:
        key_str = str(key).lower()
        values = _to_text_list(value)
        if not values:
            continue
        normalized.setdefault(key_str, [])
        normalized[key_str].extend(values)

    def pick_values(*candidates: str) -> List[str]:
        collected: List[str] = []
        for candidate in candidates:
            collected.extend(normalized.get(candidate, []))
        return collected

    def pick_first(*candidates: str) -> Optional[str]:
        for candidate in candidates:
            values = normalized.get(candidate)
            if values:
                for raw in values:
                    text = raw.strip()
                    if text:
                        return text
        return None

    artist_tokens: List[str] = []
    for raw_artist in pick_values("artist", "artists", "albumartist", "tpe1", "tpe2"):
        splits = re.split(r"[;/,&]+", raw_artist)
        for token in splits:
            cleaned = token.strip()
            if cleaned:
                artist_tokens.append(cleaned)
    if artist_tokens:
        # Remove duplicates while preserving order and limit to three to avoid noisy metadata floods.
        seen = set()
        deduped: List[str] = []
        for artist in artist_tokens:
            key = artist.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(artist)
            if len(deduped) == 3:
                break
        if deduped:
            metadata["artists"] = deduped

    album = pick_first("album", "talb")
    if album:
        metadata["album"] = album

    title = pick_first("title", "tit2")
    if title:
        metadata["title"] = title

    track_raw = pick_first("tracknumber", "trck")
    if track_raw:
        match = re.match(r"(\d+)", track_raw)
        metadata["track_number"] = match.group(1) if match else track_raw

    disc_raw = pick_first("discnumber", "tpos")
    if disc_raw:
        match = re.match(r"(\d+)", disc_raw)
        metadata["disc_number"] = match.group(1) if match else disc_raw

    date = pick_first("date", "year", "tdrc")
    if date:
        metadata["date"] = date

    genres = []
    for raw_genre in pick_values("genre", "genres", "tcon"):
        splits = re.split(r"[;/,&]+", raw_genre)
        for token in splits:
            cleaned = token.strip()
            if cleaned:
                genres.append(cleaned)
    if genres:
        seen_genres = set()
        deduped_genres: List[str] = []
        for genre in genres:
            key = genre.lower()
            if key in seen_genres:
                continue
            seen_genres.add(key)
            deduped_genres.append(genre)
            if len(deduped_genres) == 5:
                break
        if deduped_genres:
            metadata["genres"] = deduped_genres

    return metadata


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

    def rename_segments(
        self, name: str, system_prompt: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        prompt = system_prompt.strip() if system_prompt else ""
        metadata = metadata or {}

        system_messages = [
            "You are helping rename uploaded music track segments.",
            "Respond ONLY with compact JSON following this schema: " '{"title": title}',
        ]
        if metadata:
            system_messages.append(
                "Audio metadata is auto-extracted and may be polluted or outdated. "
                "Treat it as hints only and avoid inventing extra artist names."
            )
        if prompt:
            system_messages.append(
                f"The caller supplied additional requirements: {prompt}"
            )
        system_message = " ".join(filter(None, system_messages))

        def _format_metadata(data: Dict[str, Any]) -> str:
            ordered_keys = [
                "title",
                "artists",
                "album",
                "track_number",
                "disc_number",
                "date",
                "genres",
            ]
            summary_lines: List[str] = []
            handled = set()
            for key in ordered_keys:
                if key not in data:
                    continue
                handled.add(key)
                label = key.replace("_", " ").title()
                value = data[key]
                if isinstance(value, list):
                    display = ", ".join(value)
                else:
                    display = str(value)
                summary_lines.append(f"- {label}: {display}")
            for key in sorted(data.keys()):
                if key in handled:
                    continue
                label = key.replace("_", " ").title()
                value = data[key]
                if isinstance(value, list):
                    display = ", ".join(value)
                else:
                    display = str(value)
                summary_lines.append(f"- {label}: {display}")
            return "\n".join(summary_lines)

        user_lines = [
            "Here is the original audio context for renaming:",
            f"- Uploaded file name: {name}",
        ]
        if metadata:
            user_lines.extend(
                [
                    "- Best-effort metadata (may be inaccurate):",
                    _format_metadata(metadata),
                ]
            )
        user_lines.append(
            "Return only JSON with a concise, listener-friendly title that respects the hints above."
        )

        user_message = "\n".join(user_lines)

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


class UploadFeaturePipeline:
    """
    Applies the configured upload features to an embedding payload.
    """

    def __init__(
        self,
        *,
        similarity_searcher: MilvusSimilaritySearcher,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or LOGGER
        self.similarity_searcher: MilvusSimilaritySearcher = similarity_searcher

    def rename(
        self,
        name: str,
        settings: UploadSettings,
        *,
        music_file: Optional[str] = None,
    ) -> str:
        name_path = Path(name)
        name_suffix = name_path.suffix
        name = name_path.stem
        if settings.rename_enabled:
            source_for_metadata = music_file or str(name_path)
            metadata: Dict[str, Any] = {}
            if settings.use_metadata:
                metadata = best_effort_parse_metadata(source_for_metadata)
            new_name = self._apply_renaming(name, settings, metadata)
            self.logger.info(
                "Renamed %s to %s via OpenAI endpoint (metadata=%s).",
                name,
                new_name,
                bool(metadata),
            )
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

    def _apply_renaming(
        self, name: str, settings: UploadSettings, metadata: Dict[str, Any]
    ) -> str:
        renamer = OpenAiRenamer(
            endpoint=settings.openai_endpoint,
            model=settings.openai_model,
            reasoning_level=settings.reasoning_level,
            logger=self.logger,
        )

        new_name = renamer.rename_segments(name, settings.renaming_prompt, metadata)
        return new_name

    def _apply_similarity_filter(
        self, embedding_payload: SongEmbedding, settings: UploadSettings
    ) -> List[str]:
        duplicates = self.similarity_searcher.identify_duplicates(
            embedding_payload,
            settings.dedup_threshold,
        )
        return duplicates
