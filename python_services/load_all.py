#!/usr/bin/env python3
from typing import List
import json

import tqdm
from pymilvus import MilvusClient

MILVUS_URI = "http://localhost:19530"
QUERY_BATCH = 2_000

MAIN = "embedding"
COLLECTIONS = [MAIN, "clustered", "chunked_embedding"]

def fetch_all_embeddings(client: MilvusClient, collections: List[str]):
    for collection in collections:
        client.load_collection(collection)
        it = client.query_iterator(
            collection_name=collection,
            batch_size=QUERY_BATCH,
        )

        while True:
            batch = it.next()
            if not batch:
                it.close()
                break

def get_all_names(client: MilvusClient):
    client.load_collection(MAIN)
    it = client.query_iterator(
        collection_name=MAIN,
        batch_size=QUERY_BATCH,
        output_fields=["name"]
    )
    names = []
    while True:
        batch = it.next()
        if not batch:
            it.close()
            break
        for row in batch:
            names.append(row["name"])
    return names

def _get_all_from_it(iterator):
    result = []
    while True:
        batch = iterator.next()
        if not batch:
            iterator.close()
            break
        result.extend(batch)
    return result

def find_closest_vector(client: MilvusClient, song: str, embedding: list, top_k: int):
    client.load_collection(MAIN)
    search_params = {"metric_type": "COSINE", "params": {"ef": max(64, top_k)}}  # if IVF_* use {"nprobe": 16}

    res = client.search(
        collection_name=MAIN,
        anns_field="embedding",
        data=[embedding],
        limit=top_k,
        output_fields=["name"],
        filter="name not in {names}",
        filter_params={"names": [song]},
        search_params=search_params,
    )
    return res


def find_duplicates(client: MilvusClient, threshold=0.999):
    songs = get_all_names(client)
    BATCH_SIZE = 1000
    all_embeddings = []
    client.load_collection(MAIN)
    for start in range(0, len(songs), BATCH_SIZE):
        batch_songs = songs[start:start + BATCH_SIZE]
        if not batch_songs:
            continue
        batch_embeddings = get_song_embedding(client, batch_songs, ensure_loaded=False)
        all_embeddings.extend(batch_embeddings)
    duplicate_songs = {}
    duplicate_set = set()
    for song_embedding in tqdm.tqdm(all_embeddings):
        song = song_embedding["name"]
        if song in duplicate_set:
            continue
        embedding = song_embedding["embedding"]
        closest_songs = find_closest_vector(client, song, embedding, top_k=100)[0]
        matching_songs = list(map(lambda s: s["name"], filter(lambda data: data["distance"] >= threshold, closest_songs)))
        if len(matching_songs) > 0:
            for song in matching_songs:
                duplicate_set.add(song)
            duplicate_songs[song] = matching_songs
    print(len(duplicate_set))
    return duplicate_songs


def search_songs(songs: List[str], name_filters: List[str], match_lower=True, matcher=all):
    matched = filter(
        lambda song: matcher(
            name_filter in (song if not match_lower else song.lower())
            for name_filter in name_filters
        ),
        songs,
    )
    return list(matched)

def get_song_embedding(client: MilvusClient, songs: List[str], ensure_loaded: bool = True):
    if ensure_loaded:
        client.load_collection(MAIN)
    if not songs:
        return []
    result = client.query(
        collection_name=MAIN,
        filter="name in {names}",
        filter_params={"names": songs},
        output_fields=["name", "embedding"],
    )
    return result

def main():
    client = MilvusClient(uri=MILVUS_URI)
    names = get_all_names(client)
    songs = ["Orchestre de la Suisse Romande & Ernest Ansermet - The Nutcracker, op. 71_ Act II, Tableau 3. No. 14 Pas de deux"]
    beach_b = search_songs(names, ["beach", "bunny"])
    songs.extend(search_songs(beach_b, ["good", "girls"]))
    songs.extend(search_songs(beach_b, ["nice", "guys"]))
    songs.extend(search_songs(beach_b, ["love", "sick"]))
    mumford = search_songs(names, ["mumford", "sons"])
    songs.extend(search_songs(mumford, ["cold", "arms"]))

    results = get_song_embedding(client, songs)
    embeddings = {}
    for result in results:
        name = result["name"]
        embedding = result["embedding"]
        embeddings[name] = embedding
    for name, embedding in embeddings.items():
        print(name)
        results = find_closest_vector(client, name, embedding, 5)
        for result in results:
            for song in result:
                print(song)
    if False:
        print("dups")
        import json
        with open("dedupe_scan.json", "w") as file:
            json.dump(find_duplicates(client), file)


        embeddings = {}

        # find_closest_vector(client, "")


if __name__ == "__main__":
    main()
