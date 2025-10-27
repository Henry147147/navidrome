import json
from dataclasses import dataclass, asdict
from hashlib import sha224
from json import JSONDecodeError
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import torch
from pymilvus import MilvusClient, DataType

EMBEDS = Path("./embeds")
FLUSH_TO_DB = 10000

"""DATABASE SETUP"""


def make_schemas(client: MilvusClient):
    embeding_schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    embeding_schema.add_field("name", DataType.VARCHAR, is_primary=True, max_length=512)
    embeding_schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=512)
    embeding_schema.add_field("window", DataType.INT32)
    embeding_schema.add_field("hop", DataType.INT32)
    embeding_schema.add_field("sample_rate", DataType.INT32)
    embeding_schema.add_field("offset", DataType.FLOAT)
    embeding_schema.add_field(
        "chunk_ids", DataType.ARRAY, element_type=DataType.INT64, max_capacity=600
    )

    chunked_embed_schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    chunked_embed_schema.add_field("id", DataType.INT64, is_primary=True)
    chunked_embed_schema.add_field("parent_id", DataType.VARCHAR, max_length=512)
    chunked_embed_schema.add_field("start_seconds", DataType.FLOAT)
    chunked_embed_schema.add_field("end_seconds", DataType.FLOAT)
    chunked_embed_schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=512)

    client.create_collection("embedding", schema=embeding_schema)
    client.create_collection("chunked_embedding", schema=chunked_embed_schema)


def make_idxs(client: MilvusClient):
    embedding_ix = MilvusClient.prepare_index_params()
    embedding_ix.add_index(field_name="name", index_type="INVERTED")  # VARCHAR
    embedding_ix.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",  # or L2/IP
        params={"M": 50, "efConstruction": 250},
    )
    client.create_index("embedding", embedding_ix)

    # ----- indexes for collection: chunked_embedding -----
    chunked_embedding_ix = MilvusClient.prepare_index_params()
    chunked_embedding_ix.add_index(field_name="id", index_type="INVERTED")  # INT64
    chunked_embedding_ix.add_index(
        field_name="parent_id", index_type="INVERTED"  # VARCHAR
    )
    chunked_embedding_ix.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 50, "efConstruction": 250},
    )
    client.create_index("chunked_embedding", chunked_embedding_ix)


@dataclass
class SongEmbedding:
    name: str
    embedding: torch.FloatTensor
    window: int
    hop: int
    sample_rate: int
    offset: float
    chunk_ids: list


@dataclass
class ChunkedEmbedding:
    id: int
    parent_id: str
    start_seconds: float
    end_seconds: float
    embedding: torch.FloatTensor

def remove_directory_tree(start_directory: Path):
    for path in start_directory.iterdir():
        if path.is_file():
            path.unlink()
        else:
            remove_directory_tree(path)
    start_directory.rmdir()

# for chunk in tqdm(list(EMBEDS.rglob("**chunks.json"))):
def load_from_json(chunk: Path) -> Tuple[SongEmbedding, List[ChunkedEmbedding]]:
    name = str(chunk.relative_to(EMBEDS).parent)
    data = json.loads(chunk.read_text())
    window_seconds = data["window_seconds"]
    hop_seconds = data["hop_seconds"]
    sample_rate = data["sample_rate"]
    offset_seconds = data["offset_seconds"]
    main_embedding = torch.load(chunk.parent / "track.pt")
    all_chunks = torch.load(chunk.parent / "chunks.pt")
    chunks = data["chunks"]
    chunk_data = []

    for echunk in chunks:
        index = echunk["index"]
        start_seconds = echunk["start_seconds"]
        end_seconds = echunk["end_seconds"]
        id_to_hash_str = f"{name}\n{index}\n{start_seconds}\n{end_seconds}".encode(
            "utf-8"
        )
        hash_obj = sha224()
        hash_obj.update(id_to_hash_str)
        hash_bytes = hash_obj.digest()
        _id: np.int64 = np.frombuffer(hash_bytes[:8], dtype=np.int64)[0] # type: ignore
        chunk_data.append(
            ChunkedEmbedding(
                id=_id,
                parent_id=name,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                embedding=all_chunks[index],
            )
        )

    song_embed = SongEmbedding(
        name=name,
        embedding=main_embedding,
        window=window_seconds,
        hop=hop_seconds,
        sample_rate=sample_rate,
        offset=offset_seconds,
        chunk_ids=[i.id for i in chunk_data],
    )
    return song_embed, chunk_data

def insert_all(client: MilvusClient, root_embeds: Path):
    all_embeds = []
    all_chunks = []
    for chunk in tqdm(list(root_embeds.rglob("**chunks.json"))):
        try:
            main, chunks = load_from_json(chunk)
        except JSONDecodeError:
            continue
        all_embeds.append(asdict(main))
        all_chunks.extend([asdict(c) for c in chunks])
        if len(all_embeds) > FLUSH_TO_DB:
            print("Flushing main embeddings")
            client.upsert("embedding", all_embeds)
            all_embeds.clear()
        if len(all_chunks) > FLUSH_TO_DB:
            print("Flushing chunking embeddings")
            client.upsert("chunked_embedding", all_chunks)
            all_chunks.clear()
    client.upsert("embedding", all_embeds)
    client.upsert("chunked_embedding", all_chunks)


if __name__ == "__main__":
    print("Dont really want to re run this do you?")
    return
    client = MilvusClient(uri="http://localhost:19530")
    client.drop_collection("embedding")
    client.drop_collection("chunked_embedding")
    make_schemas(client)
    make_idxs(client)
    insert_all(client, root_embeds=EMBEDS)
    client.flush("embedding")
    client.flush("chunked_embedding")
    print("embedding", client.get_collection_stats("embedding"))
    print("chunked_embedding", client.get_collection_stats("chunked_embedding"))
