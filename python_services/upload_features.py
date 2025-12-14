"""
Upload feature helpers focused on duplicate detection.

Renaming support was removed; this module now only handles similarity search
for uploaded tracks before persisting embeddings.
"""

import logging
from typing import List, Optional

from database_query import MilvusSimilaritySearcher
from models import SongEmbedding, UploadSettings

LOGGER = logging.getLogger("navidrome.upload_features")


class UploadFeaturePipeline:
    """Applies duplicate filtering to embedding payloads."""

    def __init__(
        self,
        *,
        similarity_searcher: MilvusSimilaritySearcher,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or LOGGER
        self.similarity_searcher: MilvusSimilaritySearcher = similarity_searcher

    def scan_for_dups(
        self, embeddings: List[SongEmbedding], settings: UploadSettings
    ) -> List[str]:
        """Return names flagged as duplicates when similarity search is enabled."""
        removed_list: List[str] = []
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

    def _apply_similarity_filter(
        self, embedding_payload: SongEmbedding, settings: UploadSettings
    ) -> List[str]:
        return self.similarity_searcher.identify_duplicates(
            embedding_payload,
            settings.dedup_threshold,
        )
