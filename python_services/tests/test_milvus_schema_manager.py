import logging
from typing import Any, Dict, List

import pytest

from milvus_schema import MilvusSchemaManager


class FakeMilvusClient:
    def __init__(self, initial_fields: List[str], supports_add: bool = True) -> None:
        self._fields = list(initial_fields)
        self.supports_add = supports_add
        self.add_calls: List[Dict[str, Any]] = []

    def describe_collection(self, collection_name: str) -> Dict[str, Any]:
        return {"schema": {"fields": [{"name": name} for name in self._fields]}}

    def add_collection_field(
        self,
        *,
        collection_name: str,
        field_name: str,
        data_type: Any,
        max_length: int,
        nullable: bool,
    ) -> None:
        if not self.supports_add:
            raise RuntimeError("add_collection_field disabled")
        self.add_calls.append(
            {
                "collection_name": collection_name,
                "field_name": field_name,
                "data_type": data_type,
                "max_length": max_length,
                "nullable": nullable,
            }
        )
        self._fields.append(field_name)


@pytest.fixture()
def logger() -> logging.Logger:
    return logging.getLogger("navidrome.tests")


def test_schema_manager_detects_existing_field(logger: logging.Logger) -> None:
    client = FakeMilvusClient(["name", "embedding", "track_id"])
    manager = MilvusSchemaManager(client, logger=logger)
    assert manager.ensure_track_id_field("embedding") is True
    assert client.add_calls == []


def test_schema_manager_adds_missing_field_once(logger: logging.Logger) -> None:
    client = FakeMilvusClient(["name", "embedding"])
    manager = MilvusSchemaManager(client, logger=logger)

    assert manager.ensure_track_id_field("embedding") is True
    assert len(client.add_calls) == 1
    # Cached result avoids reissuing the schema change.
    assert manager.ensure_track_id_field("embedding") is True
    assert len(client.add_calls) == 1


def test_schema_manager_handles_unsupported_add(logger: logging.Logger) -> None:
    client = FakeMilvusClient(["name", "embedding"], supports_add=False)
    client.add_collection_field = None  # type: ignore[attr-defined]
    manager = MilvusSchemaManager(client, logger=logger)

    assert manager.ensure_track_id_field("embedding") is False
    assert client.add_calls == []

