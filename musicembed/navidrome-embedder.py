#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python_services.muq_milvus import EmbeddingRow, MilvusEmbeddingStore
from python_services.muq_pipeline import save_track_embedding
from python_services.muq_provider import MuQProvider, MuQProviderConfig
from python_services.muq_shared import (
    default_database_path,
    default_milvus_uri,
    default_music_dir,
    get_all_music,
    normalize_model_names,
    track_storage_key,
)

VERSION = "2.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Navidrome audio embedder")
    parser.add_argument("--db-path", default=default_database_path(), help="Path to the Navidrome sqlite database")
    parser.add_argument("--music-dir", default=default_music_dir(), help="Base music directory for relative paths")
    parser.add_argument("--milvus-uri", default=default_milvus_uri(), help="Milvus connection URI")
    parser.add_argument("--output-dir", default="./embeddings", help="Directory to save .pt embedding payloads")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["music_flamingo_audio", "muq_audio", "muq_mulan"],
        help="Embedding models to generate (music_flamingo_audio, muq_audio, muq_mulan)",
    )
    parser.add_argument("--batch-size", default=4, type=int, help="Audio batch size for MuQ inference")
    parser.add_argument("--limit", default=0, type=int, help="Limit the number of tracks processed")
    parser.add_argument("--clear-existing", action="store_true", help="Drop and recreate the target collections first")
    parser.add_argument("--no-milvus", action="store_true", help="Skip Milvus writes and only save local payloads")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.version:
        print(f"navidrome-embedder version {VERSION}")
        return

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    models = normalize_model_names(args.models)
    tracks = get_all_music(args.db_path, music_dir=args.music_dir, limit=args.limit)
    if not tracks:
        logging.info("No tracks found")
        return

    provider_config = MuQProviderConfig.from_env()
    provider = MuQProvider(
        MuQProviderConfig(
            flamingo_model_ref=provider_config.flamingo_model_ref,
            audio_model_ref=provider_config.audio_model_ref,
            mulan_model_ref=provider_config.mulan_model_ref,
            cache_dir=provider_config.cache_dir,
            device=provider_config.device,
            audio_batch_size=max(1, args.batch_size),
            text_batch_size=provider_config.text_batch_size,
            max_audio_seconds=provider_config.max_audio_seconds,
        )
    )
    store = None if args.no_milvus else MilvusEmbeddingStore(uri=args.milvus_uri)
    if store is not None:
        store.reset_for_run(models, clear_existing=args.clear_existing)

    processed = 0
    failed = 0
    for model in models:
        for batch in chunked(tracks, max(1, args.batch_size)):
            try:
                embeddings = provider.embed_audio_files([track.full_path for track in batch], model)
                for track, embedding in zip(batch, embeddings):
                    save_track_embedding(args.output_dir, track, model, embedding)
                    if store is not None:
                        store.upsert_embeddings(
                            model,
                            [
                                EmbeddingRow(
                                    name=track_storage_key(track),
                                    embedding=embedding,
                                    model_id=model,
                                )
                            ],
                        )
                    processed += 1
            except Exception as exc:
                logging.warning("Batch failed for %s on %d tracks: %s", model, len(batch), exc)
                for track in batch:
                    try:
                        embedding = provider.embed_audio_files([track.full_path], model)[0]
                        save_track_embedding(args.output_dir, track, model, embedding)
                        if store is not None:
                            store.upsert_embeddings(
                                model,
                                [
                                    EmbeddingRow(
                                        name=track_storage_key(track),
                                        embedding=embedding,
                                        model_id=model,
                                    )
                                ],
                            )
                        processed += 1
                    except Exception as item_exc:
                        failed += 1
                        logging.error("Failed to embed %s with %s: %s", track.full_path, model, item_exc)
        if store is not None:
            store.flush(model)
        provider.unload_model(model)

    logging.info("Embedding run completed. processed=%d failed=%d", processed, failed)


def chunked(values: Sequence[object], size: int) -> list[Sequence[object]]:
    if size <= 0:
        return [values]
    return [values[index : index + size] for index in range(0, len(values), size)]


if __name__ == "__main__":
    main()
