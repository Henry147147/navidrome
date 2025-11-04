import sqlite3

from track_name_resolver import TrackNameResolver


def _create_db(tmp_path):
    db_path = tmp_path / "navidrome.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE media_file (id TEXT PRIMARY KEY, artist TEXT, title TEXT)"
        )
        conn.executemany(
            "INSERT INTO media_file (id, artist, title) VALUES (?, ?, ?)",
            [
                ("id-one", "Artist One", "Title One"),
                ("id-two", "!!! • Meah Pace", "Panama Canal"),
                ("id-three", "Band Name", "AM/FM"),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_resolver_normalizes_names(tmp_path):
    db_path = _create_db(tmp_path)
    resolver = TrackNameResolver(db_path)

    id_to_name = resolver.ids_to_names(["id-one", "id-two", "missing"])
    assert id_to_name["id-one"] == "Artist One - Title One"
    assert id_to_name["id-two"] == "!!! & Meah Pace - Panama Canal"
    assert "missing" not in id_to_name

    names_to_ids = resolver.names_to_ids(
        ["Artist One - Title One", "Band Name - AM_FM", "Unknown"]
    )
    assert names_to_ids["Band Name - AM_FM"] == "id-three"
    assert "Unknown" not in names_to_ids

    assert resolver.name_to_id("!!! & Meah Pace - Panama Canal") == "id-two"
