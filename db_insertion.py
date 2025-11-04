import json
from dataclasses import dataclass, asdict
from json import JSONDecodeError
from pathlib import Path
from typing import List

import torch
from pymilvus import DataType, MilvusClient
from tqdm import tqdm

EMBEDS = Path("./embeds")
FLUSH_TO_DB = 10000

"""DATABASE SETUP"""


def make_schemas(client: MilvusClient):
    embedding_schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    embedding_schema.add_field(
        "name", DataType.VARCHAR, is_primary=True, max_length=512
    )
    embedding_schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=512)
    embedding_schema.add_field("offset", DataType.FLOAT)
    embedding_schema.add_field("model_id", DataType.VARCHAR, max_length=256)

    client.create_collection("embedding", schema=embedding_schema)


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


@dataclass
class SongEmbedding:
    name: str
    embedding: torch.FloatTensor
    offset: float
    model_id: str = ""

def remove_directory_tree(start_directory: Path):
    for path in start_directory.iterdir():
        if path.is_file():
            path.unlink()
        else:
            remove_directory_tree(path)
    start_directory.rmdir()

# for chunk in tqdm(list(EMBEDS.rglob("**chunks.json"))):
def load_from_json(chunk: Path) -> SongEmbedding:
    name = str(chunk.relative_to(EMBEDS).parent)
    data = json.loads(chunk.read_text())
    main_embedding = torch.load(chunk.parent / "track.pt")
    offset_seconds = data.get("offset_seconds", 0.0)
    song_embed = SongEmbedding(
        name=name,
        embedding=main_embedding,
        offset=offset_seconds,
        model_id=data.get("model_id", ""),
    )
    return song_embed

def insert_all(client: MilvusClient, root_embeds: Path):
    all_embeds = []
    for chunk in tqdm(list(root_embeds.rglob("**chunks.json"))):
        try:
            main = load_from_json(chunk)
        except JSONDecodeError:
            continue
        all_embeds.append(asdict(main))
        if len(all_embeds) > FLUSH_TO_DB:
            print("Flushing main embeddings")
            client.upsert("embedding", all_embeds)
            all_embeds.clear()
    if all_embeds:
        client.upsert("embedding", all_embeds)


if __name__ == "__main__":
    print("Dont really want to re run this do you?")
    return
    client = MilvusClient(uri="http://localhost:19530")
    client.drop_collection("embedding")
    make_schemas(client)
    make_idxs(client)
    insert_all(client, root_embeds=EMBEDS)
    client.flush("embedding")
    print("embedding", client.get_collection_stats("embedding"))
