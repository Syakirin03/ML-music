"""Read-only access to spotify.duckdb and candidates.csv, and where the recommender keeps its files.

Everything the recommender needs is loaded at once into a Library: one row per
song, one row per play, your likes and playlist adds, and the pools of songs
from Last.fm that scripts/fetch_candidates.py writes. That takes about a
second, and the database is closed again right away, so the builds in scripts/
can run whenever no recommender call is in progress.

A song here is a title and an artist, the title without what Spotify adds to
it: " - Remastered 2011", " (feat. ...)", " - Japanese Ver.", " - Sped Up",
"(TV Size)" (plain_titles). Spotify often has one song under several track ids
(the single and the album version, a re-release, a language version, a
remix), and those are merged, so that a song you played under one id doesn't
come back as new, or as "not played lately", under another: in the author's
history, about one track id in eight merged into another. The notebook
notebooks/01_explore_your_taste.ipynb shows how this was decided. The same plain
titles find a Last.fm song in your history (song_key), and an artist whose
Last.fm name differs from Spotify's, as after a band renames itself, is
recognised by the letters of the names and a song title in common
(artist_aliases).

The recommender never writes spotify.duckdb: the builds own it. What it learns
and saves goes to RECOMMENDER_DB, a small file of your own (see profile.py),
which also holds your ratings of recommendations (feedback.py). The Library
reads those too, so that every part of the recommender sees one snapshot.
"""

import json
import os
import re
from pathlib import Path
from typing import NamedTuple

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPOTIFY_DB = ROOT / "data" / "processed" / "spotify.duckdb"
CANDIDATES_CSV = ROOT / "data" / "processed" / "candidates.csv"
# The folder of your own files. The environment variable MUSIC_RECOMMENDER_USER_DIR, if set before the
# recommender is imported, points it elsewhere: to a copy, to try things without touching your ratings.
USER_DIR = Path(os.environ.get("MUSIC_RECOMMENDER_USER_DIR") or ROOT / "data" / "user")
# Your own file: the saved taste profile and model, and your ratings of recommendations.
RECOMMENDER_DB = USER_DIR / "recommender.duckdb"
# A copy of the ratings that feedback.py rewrites after every rating, in case the database is lost.
RATINGS_CSV = USER_DIR / "ratings.csv"

# What Spotify adds to titles and Last.fm mostly leaves out: "Song - Remastered
# 2011", "Song (feat. X)", "Song [From "Film"]". The rule of plain_title() in
# scripts/fetch_genres.py.
BRACKETED = re.compile(r"\s*[(\[（【][^)\]）】]*[)\]）】]")
DASH_SUFFIX = re.compile(r"\s+[-–—]\s+.*")

CANDIDATE_COLUMNS = [
    "pool", "pool_date", "artist_name", "track_name", "artist_rank", "listeners", "playcount", "genres", "is_seed",
    "similar_to", "source", "via_genre", "favourite_link", "asian", "artist_key", "song_key", "first_played_at",
]

# Every played track, for merge_songs().
TRACKS_SQL = """
SELECT t.track_id, t.track_name, t.artist_name, t.album_name, t.genres, t.is_music, s.plays
FROM gold.tracks t
JOIN gold.track_stats s USING (track_id)
WHERE t.is_played
"""

PLAYS_SQL = """
SELECT track_id, started_at, listen, finished, skipped_early, session_id, shuffle
FROM gold.plays
ORDER BY started_at, stream_id
"""

FEEDBACK_SQL = """
SELECT track_id, ts, event
FROM gold.feedback_events
WHERE event IN ('like', 'unlike', 'playlist_add', 'playlist_remove')
ORDER BY ts, event_id
"""

SPLITS_SQL = "SELECT split, min(started_at) AS starts_at FROM gold.plays GROUP BY split ORDER BY starts_at"

# The ratings table of RECOMMENDER_DB (feedback.py creates it and explains each column), oldest first.
RATING_COLUMNS = [
    "rated_at", "rating", "mode", "song_key", "track_id", "artist_key", "artist_name", "track_name", "genres",
    "list_id", "position", "score", "parts",
]
RATINGS_SQL = f"SELECT {', '.join(RATING_COLUMNS)} FROM ratings ORDER BY rated_at, rowid"
# The values of the rating column: a like, a dislike, and a neutral rating (you heard the song and have no
# opinion: it leaves your lists but says nothing about your taste); 0 is a rating taken back.
LIKE, DISLIKE, NEUTRAL = 1, -1, 2
# The ratings that say something about your taste: only these count in the profile and the learned models.
OPINIONS = (LIKE, DISLIKE)

# For the website's "Your data" page (data_overview()): the size of each layer of the database...
LAYERS_SQL = """
SELECT schema_name AS layer, count(*) AS tables, sum(estimated_size) AS row_count
FROM duckdb_tables()
WHERE schema_name IN ('bronze', 'silver', 'techlog', 'gold')
GROUP BY ALL
"""
# ...the rows left after each cleaning step...
STEPS_SQL = """
SELECT
    (SELECT count(*) FROM bronze.streams_raw) AS raw_streams,
    (SELECT count(*) FROM silver.streams) AS streams,
    (SELECT count(*) FROM silver.streams WHERE content_type = 'track') AS song_streams,
    (SELECT count(*) FROM silver.streams WHERE content_type = 'episode') AS podcast_streams,
    (SELECT count(*) FROM gold.plays) AS plays,
    (SELECT count(*) FROM gold.plays WHERE NOT is_music) AS not_music_plays,
    (SELECT count(DISTINCT track_id) FROM gold.plays) AS track_ids,
    (SELECT count(DISTINCT artist_name) FROM gold.plays WHERE is_music) AS artists,
    (SELECT sum(ms_played) / 3.6e6 FROM gold.plays) AS hours,
    (SELECT min(started_at) FROM gold.plays) AS first_play,
    (SELECT max(started_at) FROM gold.plays) AS last_play,
    (SELECT count(*) FROM bronze.techlog_raw) AS log_records,
    (SELECT count(*) FROM silver.log_events) AS log_events,
    (SELECT count(*) FROM silver.log_event_types) AS log_event_types,
    (SELECT min(ts) FROM silver.log_events) AS log_first,
    (SELECT max(ts) FROM silver.log_events) AS log_last,
    (SELECT count(*) FROM gold.plays WHERE genre IS NOT NULL) AS plays_with_genre,
    (SELECT count(*) FROM gold.sessions) AS sessions
"""
# ...and what you listen to, music only (is_music: config.toml can set some tracks aside), by month, hour and year.
MONTHS_SQL = """
SELECT date_trunc('month', started_at_local)::DATE AS month, count(*) AS plays, sum(ms_played) / 3.6e6 AS hours,
       avg(finished::INT) AS finished, avg(skipped_early::INT) AS skipped_early, avg(shuffle::INT) AS shuffle
FROM gold.plays
WHERE is_music
GROUP BY ALL
ORDER BY month
"""
HOURS_SQL = """
SELECT local_dow AS weekday, local_hour AS hour, count(*) AS plays, sum(ms_played) / 3.6e6 AS hours
FROM gold.plays
WHERE is_music
GROUP BY ALL
"""
GENRE_YEARS_SQL = """
SELECT year(started_at_local) AS year, genre, sum(ms_played) / 3.6e6 AS hours
FROM gold.plays
WHERE is_music
GROUP BY ALL
"""
ARTIST_YEARS_SQL = """
SELECT year(started_at_local) AS year, artist_name AS artist, sum(ms_played) / 3.6e6 AS hours,
       sum(ms_played) / sum(sum(ms_played)) OVER (PARTITION BY year(started_at_local)) AS share
FROM gold.plays
WHERE is_music
GROUP BY year(started_at_local), artist_name
QUALIFY row_number() OVER (PARTITION BY year ORDER BY hours DESC, artist) <= 3
ORDER BY year, hours DESC
"""
END_YEARS_SQL = """
SELECT year(started_at_local) AS year, end_type, count(*) AS plays
FROM gold.plays
WHERE is_music
GROUP BY ALL
ORDER BY year
"""


class Library(NamedTuple):
    """Your listening history, as the recommender sees it."""

    songs: pd.DataFrame  # one row per song; its row number is the song's number in plays and feedback
    # One row per play, oldest first: song, started_at, listen, finished, skipped_early, session_id, shuffle.
    # session_id is a row number of gold.sessions that every build regenerates: group by it, never store it.
    plays: pd.DataFrame
    feedback: pd.DataFrame  # likes, unlikes, playlist adds and removes: song, ts, event
    ends_at: pd.Timestamp  # just after the last play started, so every play started before it: "now", for the history
    split_starts: dict[str, pd.Timestamp]  # first play of the train, valid and test splits of gold.plays
    # One row per pool and song of candidates.csv, with when you first played the song (NaT if never);
    # no rows if there is no candidates.csv yet.
    candidates: pd.DataFrame
    # Every rating you gave a recommendation, oldest first (RATING_COLUMNS); current_ratings() gives the ones that count.
    ratings: pd.DataFrame = pd.DataFrame(columns=RATING_COLUMNS)


def artist_key(names: pd.Series) -> pd.Series:
    """Artists are told apart by name, as Spotify writes it: two artists can have names that differ only in case.

    Spotify spells each artist one way, so case is part of the name; only surrounding spaces are dropped.
    A Last.fm name in other letter case is matched to yours by artist_aliases().
    """
    return names.fillna("").str.strip()


def plain_titles(titles: pd.Series) -> pd.Series:
    """Titles without case and without what Spotify adds to them."""
    titles = titles.fillna("")
    plain = titles.str.replace(BRACKETED, "", regex=True).str.replace(DASH_SUFFIX, "", regex=True).str.split().str.join(" ")
    return plain.where(plain != "", titles).str.casefold()  # a title that is all brackets, such as "(Intro)", stays


def song_key(artists: pd.Series, titles: pd.Series) -> pd.Series:
    """Artist and title, ignoring case and what Spotify adds to titles: the same song on Spotify and Last.fm."""
    return artist_key(artists) + "\n" + plain_titles(titles)


def merge_songs(tracks: pd.DataFrame) -> pd.DataFrame:
    """One row per song: the tracks with the same artist_key and plain title, merged, most played first.

    The most played version gives the song its track id and names, and genres come from the most played
    version that has any. track_ids lists every version.
    """
    tracks = tracks.assign(artist_key=artist_key(tracks.artist_name), title_key=plain_titles(tracks.track_name))
    tracks = tracks.sort_values(["plays", "track_id"], ascending=[False, True])
    keys = ["artist_key", "title_key"]
    songs = tracks.groupby(keys, sort=False).agg(
        track_id=("track_id", "first"),
        track_name=("track_name", "first"),
        artist_name=("artist_name", "first"),
        album_name=("album_name", "first"),
        is_music=("is_music", "all"),
        plays=("plays", "sum"),
        track_ids=("track_id", sorted),
    )
    has_genres = tracks.genres.map(lambda g: isinstance(g, (list, np.ndarray)) and len(g) > 0)
    songs["genres"] = tracks[has_genres].groupby(keys, sort=False).genres.first()
    songs = songs.reset_index().sort_values(["plays", "track_id"], ascending=[False, True], ignore_index=True)
    # plays covers the whole history, so it only orders the songs: never use it as a feature.
    return songs[["track_id", "track_name", "artist_name", "album_name", "genres", "is_music", "track_ids", "artist_key"]]


def letters(name: str) -> str:
    """The letters and digits of a name: '(The) Band-Name' -> 'thebandname'."""
    return "".join(c for c in name.casefold() if c.isalnum())


def artist_aliases(pools: pd.DataFrame, songs: pd.DataFrame) -> dict[str, str]:
    """Pool artists (artist_key) that are an artist of your history under another name.

    Either the letters of one name contain the other's and they have a song title in common, as for a band
    you played under its old name, or the names differ only in letter case and no other artist of yours has
    the same name in other case. When two of your artists share a name in different cases, Last.fm's spelling
    needs the song title too.
    """
    artists_by_title = pd.Series(songs.artist_key.to_numpy(), index=plain_titles(songs.track_name).to_numpy())
    artists_by_title = artists_by_title.groupby(level=0).agg(set)
    unknown = pools[~pools.artist_key.isin(set(songs.artist_key))]
    aliases: dict[str, str] = {}
    for artist, title in zip(unknown.artist_key, plain_titles(unknown.track_name)):
        if artist in aliases:
            continue
        for known in artists_by_title.get(title, ()):
            a, b = letters(artist), letters(known)
            if len(a) > 1 and len(b) > 1 and (a in b or b in a):
                aliases[artist] = known
                break
    names = pd.Series(songs.artist_key.unique())
    only_spelling = {folded: group.iloc[0] for folded, group in names.groupby(names.str.casefold()) if len(group) == 1}
    for artist in unknown.artist_key.unique():
        if artist not in aliases and artist.casefold() in only_spelling:
            aliases[artist] = only_spelling[artist.casefold()]
    return aliases


def load_candidates(path: Path, songs: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """The pools of candidates.csv, with when you first played each song. No rows if there is no such file."""
    if not path.exists():
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    pools = pd.read_csv(path, encoding="utf-8-sig", na_values=["NA"], keep_default_na=False)
    pools["pool_date"] = pd.to_datetime(pools.pool_date, utc=True)
    pools["genres"] = pools.genres.str.split("|")
    pools["similar_to"] = pools.similar_to.map(json.loads)
    if "source" not in pools:  # a candidates.csv from before the genre artists
        pools["source"] = np.where(pools.is_seed, "seed", "similar")
        pools["via_genre"] = None
    if "favourite_link" not in pools:  # a candidates.csv from before the favourite artists
        pools["favourite_link"] = 0.0
    if "asian" not in pools:  # a candidates.csv from before: no song counts as Western, so none is pushed down
        pools["asian"] = True
    aliases = artist_aliases(pools.assign(artist_key=artist_key(pools.artist_name)), songs)
    pools["artist_key"] = artist_key(pools.artist_name).map(lambda key: aliases.get(key, key))
    pools["song_key"] = pools.artist_key + "\n" + plain_titles(pools.track_name)
    first_play = plays.groupby("song").started_at.min()
    keys = song_key(songs.artist_name, songs.track_name).iloc[first_play.index]
    first_by_key = pd.Series(first_play.to_numpy(), index=keys.to_numpy()).groupby(level=0).min()
    pools["first_played_at"] = pools.song_key.map(first_by_key)
    return pools[CANDIDATE_COLUMNS]


def window_listens(library: Library, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Listens per song (index: song number) in plays started in [start, end)."""
    plays = library.plays[(library.plays.started_at >= start) & (library.plays.started_at < end)]
    return plays.groupby("song").listen.sum()


def cutoff_after(library: Library, years: int) -> pd.Timestamp:
    """Midnight on the first day of the month after the one that holds the day years after your first play.

    The checks that learn from the past start there, so that they have some history to learn from: for a
    first play on 6 March 2018 and years 2, on 1 April 2020.
    """
    later = library.plays.started_at.min() + pd.DateOffset(years=years)
    return later.normalize() + pd.offsets.MonthBegin(1)


def candidate_pool(library: Library, name: str) -> pd.DataFrame:
    """The songs of one pool of candidates.csv (now, valid or test), each once. Notes when the pool is out of date."""
    rows = library.candidates[library.candidates.pool == name]
    expected = library.ends_at if name == "now" else library.split_starts[name]
    if len(rows) and rows.pool_date.iloc[0] != expected:
        print(
            f"Note: the {name} pool in candidates.csv is for {rows.pool_date.iloc[0]:%d %b %Y}, but your history says "
            f"{expected:%d %b %Y}: run scripts/fetch_candidates.py again"
        )
    return rows.drop_duplicates("song_key").reset_index(drop=True)


def load_ratings(db: Path = RECOMMENDER_DB) -> pd.DataFrame:
    """Every rating in db, oldest first; no rows if there are none yet."""
    if not db.exists():
        return pd.DataFrame(columns=RATING_COLUMNS)
    try:
        con = duckdb.connect(str(db), read_only=True)
    except duckdb.IOException as error:
        raise RuntimeError(f"Could not open {db.name} to read your ratings: {error}. Close other programs using it.") from error
    try:
        ratings = con.sql(RATINGS_SQL).df()
    except duckdb.CatalogException:
        return pd.DataFrame(columns=RATING_COLUMNS)
    finally:
        con.close()
    ratings["genres"] = ratings.genres.map(lambda g: g.split("|") if isinstance(g, str) and g else [])
    return ratings


def current_ratings(ratings: pd.DataFrame, before: pd.Timestamp | None = None) -> pd.DataFrame:
    """The latest rating of each song (rated before `before`, if given), leaving out ratings taken back.

    Neutral ratings are kept, since they keep a song off your lists; filter on OPINIONS for what counts as taste.
    """
    if before is not None:
        ratings = ratings[ratings.rated_at < before]
    latest = ratings.drop_duplicates("song_key", keep="last")
    return latest[latest.rating != 0].reset_index(drop=True)


def load_library(db: Path = SPOTIFY_DB, candidates: Path = CANDIDATES_CSV, user_db: Path = RECOMMENDER_DB) -> Library:
    if not db.exists():
        raise FileNotFoundError(f"Database not found: {db} (run scripts/build_database.py, then scripts/build_gold.py)")
    try:
        con = duckdb.connect(str(db), read_only=True)
    except duckdb.IOException as error:
        raise RuntimeError(f"Could not open {db.name}: {error}. Is a build running?") from error
    try:
        songs = merge_songs(con.sql(TRACKS_SQL).df())
        plays = con.sql(PLAYS_SQL).df()
        feedback = con.sql(FEEDBACK_SQL).df()
        splits = con.sql(SPLITS_SQL).df()
    except (duckdb.CatalogException, duckdb.BinderException) as error:
        raise RuntimeError(f"{db.name} has no gold layer, or an old one: {error} (run scripts/build_gold.py)") from error
    finally:
        con.close()

    song_of_track = songs.track_ids.explode().reset_index().set_index("track_ids")["index"]
    plays.insert(0, "song", plays.pop("track_id").map(song_of_track).astype(int))
    # Likes of tracks you never played can't be tied to a song: they have no title in the export.
    feedback.insert(0, "song", feedback.pop("track_id").map(song_of_track))
    feedback = feedback.dropna(subset=["song"]).astype({"song": int}).reset_index(drop=True)
    split_starts = dict(zip(splits.split, splits.starts_at))
    ends_at = plays.started_at.max() + pd.Timedelta(1, unit="s")
    return Library(songs, plays, feedback, ends_at, split_starts, load_candidates(candidates, songs, plays), load_ratings(user_db))


class Overview(NamedTuple):
    """What the website's "Your data" page shows about the database (data_overview())."""

    layers: pd.DataFrame  # layer, tables, row_count: bronze, silver, techlog and gold
    steps: pd.Series  # rows left after each cleaning step, and totals (STEPS_SQL)
    months: pd.DataFrame  # month, plays, hours, finished, skipped_early, shuffle (shares of the month's plays)
    hours: pd.DataFrame  # weekday (1 Monday ... 7 Sunday, local time), hour, plays, hours
    genre_years: pd.DataFrame  # year, genre (main genre, NaN if Last.fm had none), hours
    artist_years: pd.DataFrame  # year, artist, hours, share: the three most played artists of each year
    end_years: pd.DataFrame  # year, end_type (finished, skipped, interrupted), plays
    file_bytes: int  # the size of the database file


def data_overview(db: Path = SPOTIFY_DB) -> Overview:
    """The sizes, cleaning steps and listening summaries of the database, for the website (read-only, a second)."""
    if not db.exists():
        raise FileNotFoundError(f"Database not found: {db} (run scripts/build_database.py, then scripts/build_gold.py)")
    try:
        con = duckdb.connect(str(db), read_only=True)
    except duckdb.IOException as error:
        raise RuntimeError(f"Could not open {db.name}: {error}. Is a build running?") from error
    try:
        layers = con.sql(LAYERS_SQL).df()
        steps = con.sql(STEPS_SQL).df().iloc[0]
        frames = [con.sql(sql).df() for sql in (MONTHS_SQL, HOURS_SQL, GENRE_YEARS_SQL, ARTIST_YEARS_SQL, END_YEARS_SQL)]
    except (duckdb.CatalogException, duckdb.BinderException) as error:
        raise RuntimeError(f"{db.name} has no gold layer, or an old one: {error} (run scripts/build_gold.py)") from error
    finally:
        con.close()
    order = {"bronze": 0, "silver": 1, "techlog": 2, "gold": 3}
    layers = layers.sort_values("layer", key=lambda names: names.map(order)).reset_index(drop=True)
    return Overview(layers, steps, *frames, db.stat().st_size)
