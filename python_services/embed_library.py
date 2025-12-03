"""
Standalone script to embed the full Navidrome library into a Milvus Lite DB file.

Usage:
    python python_services/embed_library.py \
        --db-path ./navidrome.db \
        --music-root ./music \
        --milvus-db-path ./milvus.db \
        --models muq qwen3 \
        --no-clear-existing
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure local modules are importable when run from repo root
sys.path.append(str(Path(__file__).parent))

from batch_embedding_job import start_batch_job, get_current_job  # noqa: E402
from gpu_settings import load_gpu_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed entire Navidrome library into Milvus Lite DB.")
    parser.add_argument("--db-path", required=True, help="Path to navidrome.db (SQLite).")
    parser.add_argument("--music-root", required=True, help="Path to the music library root directory.")
    parser.add_argument(
        "--milvus-db-path",
        required=True,
        help="Path to Milvus Lite database file. Created if missing.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["muq", "qwen3"],
        help="Embedding models to run (default: muq qwen3).",
    )
    parser.add_argument(
        "--clear-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear existing embeddings before re-embedding (default: true).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save progress every N tracks (default: 50).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("embed_library")

    db_path = Path(args.db_path).expanduser().resolve()
    music_root = Path(args.music_root).expanduser().resolve()
    milvus_db_path = Path(args.milvus_db_path).expanduser().resolve()

    if not db_path.exists():
        logger.error("Navidrome DB not found: %s", db_path)
        return 1
    if not music_root.exists():
        logger.error("Music root not found: %s", music_root)
        return 1

    milvus_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure Milvus Lite for downstream modules
    os.environ["NAVIDROME_MILVUS_DB_PATH"] = str(milvus_db_path)
    os.environ["NAVIDROME_MILVUS_URI"] = str(milvus_db_path)

    gpu_settings = load_gpu_settings()
    logger.info(
        "Starting batch embedding: db=%s music_root=%s milvus_db=%s models=%s clear_existing=%s",
        db_path,
        music_root,
        milvus_db_path,
        args.models,
        args.clear_existing,
    )

    job = start_batch_job(
        db_path=str(db_path),
        music_root=str(music_root),
        milvus_uri=str(milvus_db_path),
        checkpoint_interval=args.checkpoint_interval,
        gpu_settings=gpu_settings,
    )

    # Run synchronously
    job.run(models=args.models, clearExisting=args.clear_existing)

    progress = job.get_progress()
    logger.info(
        "Completed: tracks=%s/%s failures=%s status=%s",
        progress.processed_tracks,
        progress.total_tracks,
        progress.failed_tracks,
        progress.status,
    )
    if progress.last_error:
        logger.warning("Last error: %s", progress.last_error)
    return 0


if __name__ == "__main__":
    sys.exit(main())
