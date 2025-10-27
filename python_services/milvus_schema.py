"""Helpers for ensuring Milvus collections expose the fields Navidrome expects."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from pymilvus import DataType, MilvusClient


class MilvusSchemaManager:
    """Ensures that required Milvus schema fields exist before inserting data."""

    def __init__(
        self,
        client: MilvusClient,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.client = client
        self.logger = logger or logging.getLogger("navidrome.milvus_schema")
        self._availability_cache: Dict[Tuple[str, str], bool] = {}

    def ensure_track_id_field(self, collection_name: str) -> bool:
        """Make sure the given collection accepts a ``track_id`` VARCHAR field."""

        return self.ensure_varchar_field(
            collection_name=collection_name,
            field_name="track_id",
            max_length=512,
        )

    def ensure_varchar_field(
        self,
        *,
        collection_name: str,
        field_name: str,
        max_length: int = 512,
        nullable: bool = True,
    ) -> bool:
        """Check for (and lazily add) a VARCHAR field on the target collection."""

        cache_key = (collection_name, field_name)
        cached = self._availability_cache.get(cache_key)
        if cached is True:
            return True

        try:
            description = self.client.describe_collection(
                collection_name=collection_name
            )
        except Exception:  # pragma: no cover - defensive logging
            self.logger.exception(
                "Failed to describe Milvus collection %s", collection_name
            )
            self._availability_cache[cache_key] = False
            return False

        fields = (
            description.get("schema", {}).get("fields")
            if isinstance(description, dict)
            else None
        )
        if fields and any(field.get("name") == field_name for field in fields):
            self._availability_cache[cache_key] = True
            return True

        add_field = getattr(self.client, "add_collection_field", None)
        if add_field is None:
            self.logger.warning(
                "Milvus client cannot add field %s to %s; operation unsupported",
                field_name,
                collection_name,
            )
            self._availability_cache[cache_key] = False
            return False

        try:
            add_field(
                collection_name=collection_name,
                field_name=field_name,
                data_type=DataType.VARCHAR,
                max_length=max_length,
                nullable=nullable,
            )
            self.logger.info(
                "Added missing field %s to Milvus collection %s",
                field_name,
                collection_name,
            )
        except Exception:  # pragma: no cover - defensive logging
            self.logger.exception(
                "Unable to add field %s to Milvus collection %s",
                field_name,
                collection_name,
            )
            self._availability_cache[cache_key] = False
            return False

        try:
            updated = self.client.describe_collection(collection_name=collection_name)
        except Exception:  # pragma: no cover - defensive logging
            self.logger.warning(
                "Milvus did not immediately expose schema update for %s.%s; assuming success",
                collection_name,
                field_name,
            )
            self._availability_cache[cache_key] = True
            return True

        updated_fields = (
            updated.get("schema", {}).get("fields")
            if isinstance(updated, dict)
            else None
        )
        if updated_fields and any(
            field.get("name") == field_name for field in updated_fields
        ):
            self._availability_cache[cache_key] = True
            return True

        self.logger.warning(
            "Field %s still absent from Milvus collection %s after attempting to add it",
            field_name,
            collection_name,
        )
        self._availability_cache[cache_key] = False
        return False


__all__ = ["MilvusSchemaManager"]

