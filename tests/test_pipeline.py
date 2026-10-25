"""The whole pipeline on a tiny made-up export: build_database.py, build_gold.py, then the recommender's view.

The export imitates Spotify's Extended Streaming History (the same fields, as JSON), without a technical log,
which a newcomer may not have. It holds two listening sessions on two days, an exact duplicate, a podcast
episode, two versions of one song, two artists whose names differ only in case, and two tracks that aren't
music: one by genre, one by name. The scripts run as you would run them, with a config.toml of their own.
"""

import json
import os
import subprocess
import sys

import duckdb
import pytest

from recommender.config import ROOT
from recommender.data import load_library
from recommender.profile import build_profile

TIMEZONE = "Asia/Tokyo"  # UTC+9 all year, so local hours are easy to check
CONFIG = f"""
[home]
timezone = "{TIMEZONE}"

[not_music]
genres = ["nature sounds"]
pattern = '\\bwhite noise\\b'
"""
GENRES = """track_id,track_name,artist_name,album_name,genre,genres,genre_source
0000000000000000000001,First Song,Test Band,First Album,rock,rock|pop,track
0000000000000000000002,First Song - Remastered 2011,Test Band,First Album,rock,rock|pop,album
0000000000000000000003,Second Song,Test Band,First Album,rock,rock,artist
0000000000000000000004,First Song,TEST BAND,Other Album,jazz,jazz,track
0000000000000000000005,White Noise for Sleep,Noise Machine Lab,Sleep,NA,NA,NA
0000000000000000000006,Forest Rain,Calm Artist,Nature,nature sounds,nature sounds|ambient,artist
"""


def record(ts, ms, artist=None, title=None, album=None, track=None, reason_end="trackdone", episode=None):
    """One event as Spotify exports it; the IP address is from a range reserved for documentation."""
    return {
        "ts": ts, "platform": "android", "ms_played": ms, "conn_country": "ZZ", "ip_addr": "192.0.2.1",
        "master_metadata_track_name": title, "master_metadata_album_artist_name": artist,
        "master_metadata_album_album_name": album, "spotify_track_uri": track and f"spotify:track:{track}",
        "episode_name": episode, "episode_show_name": episode and "A Show",
        "spotify_episode_uri": episode and "spotify:episode:0000000000000000000009",
        "audiobook_title": None, "audiobook_uri": None, "audiobook_chapter_uri": None, "audiobook_chapter_title": None,
        "reason_start": "clickrow", "reason_end": reason_end, "shuffle": False, "skipped": None,
        "offline": False, "offline_timestamp": None, "incognito_mode": False,
    }


def export() -> list[dict]:
    noise = record("2024-03-01T12:12:00Z", 230_000, "Noise Machine Lab", "White Noise for Sleep", "Sleep", "0000000000000000000005")
    return [
        # Friday 1 March 2024: one session.
        record("2024-03-01T12:00:00Z", 180_000, "Test Band", "First Song", "First Album", "0000000000000000000001"),
        record("2024-03-01T12:03:30Z", 200_000, "Test Band", "First Song - Remastered 2011", "First Album", "0000000000000000000002"),
        record("2024-03-01T12:04:00Z", 10_000, "Test Band", "Second Song", "First Album", "0000000000000000000003", "fwdbtn"),
        record("2024-03-01T12:08:00Z", 240_000, "TEST BAND", "First Song", "Other Album", "0000000000000000000004"),
        noise,
        noise,  # the export repeats some events word for word
        record("2024-03-01T12:16:00Z", 200_000, "Calm Artist", "Forest Rain", "Nature", "0000000000000000000006"),
        # Saturday 2 March: a second session, and a podcast.
        record("2024-03-02T09:00:00Z", 200_000, "Test Band", "Second Song", "First Album", "0000000000000000000003"),
        record("2024-03-02T09:04:00Z", 180_000, "Test Band", "First Song", "First Album", "0000000000000000000001"),
        record("2024-03-02T09:05:00Z", 60_000, episode="Episode 1"),
    ]


def run(*args: str, env: dict) -> str:
    done = subprocess.run([sys.executable, "-X", "utf8", *args], cwd=ROOT, env=env, capture_output=True,
                          text=True, encoding="utf-8")
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The database built from the made-up export, and what the builds printed."""
    folder = tmp_path_factory.mktemp("pipeline")
    (folder / "export").mkdir()
    (folder / "export" / "Streaming_History_Audio_2024.json").write_text(json.dumps(export()), encoding="utf-8")
    (folder / "config.toml").write_text(CONFIG, encoding="utf-8")
    (folder / "track_genres.csv").write_text(GENRES, encoding="utf-8")
    env = {**os.environ, "MUSIC_PREFERENCE_CONFIG": str(folder / "config.toml")}
    db = folder / "spotify.duckdb"
    printed = run("scripts/build_database.py", "--data-dir", str(folder / "export"),
                  "--techlog-dir", str(folder / "no technical log"), "--db", str(db), env=env)
    printed += run("scripts/build_gold.py", "--db", str(db), "--genres", str(folder / "track_genres.csv"), env=env)
    return db, printed


def query(db, sql: str) -> list[tuple]:
    with duckdb.connect(str(db), read_only=True) as con:
        return con.sql(sql).fetchall()


def test_the_builds_succeed_without_warnings_or_a_technical_log(built):
    _, printed = built
    assert "WARNING" not in printed
    assert "no technical log folder" in printed


def test_cleaning_keeps_one_copy_and_separates_podcasts(built):
    db, _ = built
    assert query(db, "SELECT count(*) FROM bronze.streams_raw") == [(10,)]  # as exported
    assert query(db, "SELECT count(*) FROM silver.streams") == [(9,)]  # the repeat removed
    assert query(db, "SELECT count(*) FROM silver.plays") == [(8,)]  # music only
    assert query(db, "SELECT count(*) FROM silver.episodes WHERE play_count > 0") == [(1,)]


def test_local_time_uses_the_configured_time_zone(built):
    db, _ = built
    # The first play stopped at 12:00 UTC after 3 minutes: it started at 11:57 UTC, 20:57 in Tokyo, on a Friday.
    first = query(db, "SELECT local_hour, local_dow FROM gold.plays WHERE track_id = '0000000000000000000001' ORDER BY started_at LIMIT 1")
    assert first == [(20, 5)]


def test_sessions_labels_and_artists(built):
    db, _ = built
    assert query(db, "SELECT count(*) FROM gold.sessions") == [(2,)]  # a night's pause splits them
    labels = dict(query(db, "SELECT reason_end, label FROM gold.plays WHERE local_date = '2024-03-01' AND track_id IN "
                            "('0000000000000000000001', '0000000000000000000003')"))
    assert labels == {"trackdone": 1, "fwdbtn": 0}  # finished, and skipped within 30 seconds
    assert sorted(query(db, "SELECT artist_name FROM gold.artists WHERE artist_name ILIKE 'test band'")) == [("TEST BAND",), ("Test Band",)]


def test_tracks_that_are_not_music_are_marked_by_genre_and_by_name(built):
    db, _ = built
    not_music = query(db, "SELECT track_name FROM gold.tracks WHERE NOT is_music ORDER BY 1")
    assert not_music == [("Forest Rain",), ("White Noise for Sleep",)]


def test_the_recommender_merges_versions_and_leaves_out_what_is_not_music(built, tmp_path):
    db, _ = built
    library = load_library(db=db, candidates=tmp_path / "no pool.csv", user_db=tmp_path / "recommender.duckdb")
    songs = library.songs.set_index(["artist_name", "track_name"])
    assert len(songs) == 5  # the remastered First Song is the same song; TEST BAND's is another
    assert len(songs.loc[("Test Band", "First Song"), "track_ids"]) == 2
    assert not songs.loc[("Noise Machine Lab", "White Noise for Sleep"), "is_music"]

    profile = build_profile(library=library, save=False)
    assert profile.genres.index[0] == "rock"
    assert profile.genres.get("nature sounds", 0) == 0 and profile.genres.get("ambient", 0) == 0
    assert profile.artists.artist_name.iloc[0] == "Test Band"
