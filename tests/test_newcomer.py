"""Someone else's history: every model trains, every check runs, and a short history is told so, not crashed on.

The history is made up: eight artists with ten songs each, one listening session a day, and favourite artists
that change every four months and come round again later, so that songs go quiet and some come back, as they
do in a real history. A newcomer with 4.5 years of it can train the rediscover model; one with 2.5 years is told
that it is too little. Neither needs a Last.fm pool here (songs new to you are left out).
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from recommender import evaluate, ranker, sessions
from recommender.config import ROOT
from recommender.data import cutoff_after, load_library
from recommender.feedback import feedback_model
from recommender.recommend import recommend_songs, similar_songs

ARTISTS = {
    "Synth Coast": "synth-pop|electronic", "Granite Hours": "rock|indie", "Velvet Tram": "jazz",
    "Neon Harbor": "k-pop|pop", "Paper Lanterns": "folk|indie", "Static Bloom": "electronic",
    "Copper Fields": "country", "Night Ferry": "hip hop|rap",
}
SONGS_PER_ARTIST = 10
PHASE_DAYS = 120  # how long a pair of artists stays the favourites


def songs() -> pd.DataFrame:
    rows = [(artist, f"{artist} Song {n}", genres) for artist, genres in ARTISTS.items() for n in range(1, SONGS_PER_ARTIST + 1)]
    frame = pd.DataFrame(rows, columns=["artist", "title", "genres"])
    frame["track_id"] = [f"{i + 1:022d}" for i in range(len(frame))]
    return frame


def export(years: float, seed: int = 0) -> list[dict]:
    """A daily session of six plays: the songs of the phase's two favourite artists, and now and then any song."""
    rng = np.random.default_rng(seed)
    catalogue, records = songs(), []
    artists = list(ARTISTS)
    for day in range(int(years * 365)):
        phase = day // PHASE_DAYS
        favourites = catalogue[catalogue.artist.isin([artists[phase % 8], artists[(phase + 3) % 8]])].index
        clock = pd.Timestamp("2020-03-02 11:00", tz="UTC") + pd.Timedelta(day, unit="D") + pd.Timedelta(int(rng.integers(0, 120)), unit="min")
        for _ in range(6):
            song = catalogue.loc[rng.choice(favourites) if rng.random() < 0.95 else rng.integers(0, len(catalogue))]
            skipped = rng.random() < 0.15
            ms = int(rng.integers(5_000, 25_000)) if skipped else int(rng.integers(170_000, 230_000))
            clock += pd.Timedelta(ms, unit="ms")
            records.append({
                "ts": clock.strftime("%Y-%m-%dT%H:%M:%SZ"), "platform": "ios", "ms_played": ms, "conn_country": "ZZ",
                "ip_addr": "192.0.2.1", "master_metadata_track_name": song.title,
                "master_metadata_album_artist_name": song.artist, "master_metadata_album_album_name": f"{song.artist} Album",
                "spotify_track_uri": f"spotify:track:{song.track_id}", "episode_name": None, "episode_show_name": None,
                "spotify_episode_uri": None, "audiobook_title": None, "audiobook_uri": None, "audiobook_chapter_uri": None,
                "audiobook_chapter_title": None, "reason_start": "trackdone", "reason_end": "fwdbtn" if skipped else "trackdone",
                "shuffle": False, "skipped": None, "offline": False, "offline_timestamp": None, "incognito_mode": False,
            })
    return records


def build(folder, years: float):
    """The database of a made-up history of this many years, built by the scripts as you would run them."""
    (folder / "export").mkdir()
    (folder / "export" / "Streaming_History_Audio.json").write_text(json.dumps(export(years)), encoding="utf-8")
    catalogue = songs()
    genres = catalogue.assign(track_name=catalogue.title, artist_name=catalogue.artist, album_name=catalogue.artist + " Album",
                              genre=catalogue.genres.str.split("|").str[0], genre_source="track")
    genres[["track_id", "track_name", "artist_name", "album_name", "genre", "genres", "genre_source"]].to_csv(
        folder / "track_genres.csv", index=False)
    (folder / "config.toml").write_text('[home]\ntimezone = "UTC"\n', encoding="utf-8")
    env = {**os.environ, "MUSIC_PREFERENCE_CONFIG": str(folder / "config.toml")}
    db = folder / "spotify.duckdb"
    for script, *options in (
        ("scripts/build_database.py", "--data-dir", folder / "export", "--techlog-dir", folder / "none"),
        ("scripts/build_gold.py", "--genres", folder / "track_genres.csv"),
    ):
        done = subprocess.run([sys.executable, "-X", "utf8", script, *map(str, options), "--db", str(db)], cwd=ROOT,
                              env=env, capture_output=True, text=True, encoding="utf-8", check=False)
        assert done.returncode == 0, done.stdout + done.stderr
    return load_library(db=db, candidates=folder / "no pool.csv", user_db=folder / "recommender.duckdb")


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return build(tmp_path_factory.mktemp("four_and_a_half_years"), 4.5)


def run_cli(module, library, monkeypatch, capsys, *args) -> str:
    monkeypatch.setattr(module, "load_library", lambda *_, **__: library)
    monkeypatch.setattr(sys, "argv", [module.__name__, *args])
    module.main()
    return capsys.readouterr().out


def test_the_checks_start_two_years_after_the_first_play(library):
    assert cutoff_after(library, 2) == pd.Timestamp("2022-04-01", tz=library.ends_at.tz)


def test_a_newcomer_trains_the_rediscover_model_and_gets_songs_from_it(library, monkeypatch, capsys):
    printed = run_cli(ranker, library, monkeypatch, capsys)
    assert "Saved to" in printed and "Month by month: nothing to judge yet" in printed
    assert ranker.load_ranker() is not None
    table = recommend_songs(top_n=5, mode="rediscover", library=library)
    assert len(table) == 5


def test_without_ratings_the_models_wait_for_them(library):
    assert "not in use yet: 0 likes and 0 dislikes" in feedback_model(library.ratings, library.candidates).message
    assert "not in use yet" in ranker.rediscover_feedback(library).message


def test_songs_like_one_you_name(library):
    table = similar_songs("Velvet Tram Song 1", top_n=5, library=library)
    assert len(table) == 5


def test_the_session_check_runs(library, monkeypatch, capsys):
    assert "quarter by quarter" in run_cli(sessions, library, monkeypatch, capsys)


def test_the_evaluation_runs_without_a_pool(library, monkeypatch, capsys):
    printed = run_cli(evaluate, library, monkeypatch, capsys, "--window", "test")
    assert "test window" in printed and "learned: logistic regression" in printed
    assert "no test pool in candidates.csv" in printed


def test_too_short_a_history_is_told_so(tmp_path, monkeypatch, capsys):
    short = build(tmp_path, 2.5)
    with pytest.raises(SystemExit, match="Too little history"):
        run_cli(ranker, short, monkeypatch, capsys)
    printed = run_cli(evaluate, short, monkeypatch, capsys, "--window", "test")
    assert "too little history before this window" in printed
