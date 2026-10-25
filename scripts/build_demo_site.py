"""Build the data of the demo website in site/: a made-up listener, run through the real pipeline and recommender.

    python scripts/build_demo_site.py

The demo (site/, published on GitHub Pages) shows what the website does without anyone's data. This script
makes up a listener: the artists of HISTORY_ARTISTS and about YEARS years of listening to them, with favourites
that come and go, a Last.fm pool of songs new to them, and ratings of RATED_LISTS lists by a rule of its own
(LOVED_GENRES, SHUNNED_GENRES). Then it runs what you would run on your own export:

    build_database.py, build_gold.py   on the made-up export and its genres
    python -m recommender.ranker       the rediscover model, learned from the made-up history
    the website's lists and ratings    lists of songs new to the listener, rated list after list
    the rest                           the model learned from those ratings, songs like each song of the
                                       history, and the numbers of the Your data and How it works pages

and keeps what the website shows in site/data/demo.js. The page ranks the pool again in the browser
(site/assets/engine.js, the formulas of recommender/scoring.py and recommend.py) whenever a visitor moves a
dial. With node installed, this script then checks that the browser's lists are the recommender's
(scripts/check_demo_engine.js).

Nothing of yours is read or written: the builds run in a scratch folder (SCRATCH) with the default settings and
UTC, and the recommender's own files go there too, never to data/user/. Every artist and song is made up. Run it
again after changing the recommender, so that the demo shows what the code does now. It takes a few minutes.
"""

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
DATA_JS = SITE / "data" / "demo.js"
# Everything made here goes to this folder, emptied at every run, and so do the recommender's own files (set
# before it is imported): data/user/, where your ratings are, is never touched.
SCRATCH = Path(tempfile.gettempdir()) / "music-preference-demo"
os.environ["MUSIC_PREFERENCE_CONFIG"] = str(SCRATCH / "config.toml")
os.environ["MUSIC_RECOMMENDER_USER_DIR"] = str(SCRATCH / "user")
sys.path.insert(0, str(ROOT))
_connect = duckdb.connect


def connect_in_utc(*args, **kwargs):
    """duckdb.connect with times in UTC, the made-up listener's time zone: DuckDB would give them in this
    computer's, which is yours and has no place in the demo."""
    con = _connect(*args, **kwargs)
    con.execute("SET TimeZone = 'UTC'")
    return con


duckdb.connect = connect_in_utc

from recommender import ranker as ranker_cli
from recommender import recommend
from recommender.data import NEUTRAL, OPINIONS, RECOMMENDER_DB, current_ratings, data_overview, load_library, load_ratings
from recommender.features import genre_list
from recommender.feedback import MIN_EACH, MODEL_MARGIN, feedback_model, model_weights, rate, remember_list
from recommender.profile import HALF_LIFE_DAYS, RATING_LISTENS
from recommender.ranker import FEATURES, HORIZON_DAYS, load_ranker, ranker_summary, rediscover_feedback
from recommender.recommend import (
    FOR_VARIETY, MAX_PER_ARTIST, SAME_COMPANY, SHORTLIST, STRIKES, VARIETY, find_song, profile_or_build, rated_keys,
    recommend_songs, similar_songs, struck_artists,
)
from recommender.scoring import DORMANT_DAYS, FAMILIARITY, GENRE_BOOST, POPULARITY_WEIGHT, WESTERN, discovery_candidates
from recommender.sessions import DIMENSIONS, WINDOW, shared_sessions, song_space

# Each stage draws from its own stream (SEED, SEED + 1, ...), so that changing one doesn't reshuffle the others.
SEED = 7
# The history: from START, about YEARS years, with SESSIONS_PER_DAY sessions on an average day, more on weekends
# and in later years, and BREAKS holidays without music.
START = "2021-01-04"
YEARS = 5.7
SESSIONS_PER_DAY = 1.0
BREAKS = 4
# Favourites change every PHASE_DAYS: FAVOURITES artists, chosen by how much the listener likes them, take
# FAVOURITE_SHARE of a session, and the last phase's favourites FADING_SHARE. The rest are the HITS most played
# songs of any artist (the more liked, the likelier), and shuffled sessions shuffle the current and the last
# favourites. So the other songs of
# an artist go quiet when its phase ends, and some come back when it is a favourite again: what the rediscover
# model learns from.
PHASE_DAYS = 120
FAVOURITES = 2
FAVOURITE_SHARE = 0.7
FADING_SHARE = 0.15
HITS = 3
# When in the day sessions start, by hour (UTC: the made-up listener lives there).
WEEKDAY_HOURS = [1, 0.5, 0.2, 0.1, 0.1, 0.2, 1, 4, 5, 3, 2, 2, 3, 3, 2, 2, 3, 5, 6, 6, 7, 6, 4, 2]
WEEKEND_HOURS = [2, 1, 0.5, 0.2, 0.1, 0.1, 0.2, 0.5, 1, 2, 3, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 5, 4, 3]

# The made-up listener's artists: genres (main first), how much they like them, and how often they skip them.
HISTORY_ARTISTS = {
    "Synth Coast": ("synth-pop|electronic|new wave", 1.0, 0.10),
    "Marble Arcade": ("dream pop|indie pop|shoegaze", 0.95, 0.07),
    "Granite Hours": ("indie rock|rock|alternative rock", 0.8, 0.12),
    "Paper Lanterns": ("indie folk|folk|singer-songwriter", 0.7, 0.10),
    "Velvet Tram": ("jazz|soul|bossa nova", 0.7, 0.08),
    "Neon Harbor": ("k-pop|pop|dance", 0.6, 0.15),
    "Saltwater Choir": ("indie pop|pop|indie folk", 0.6, 0.12),
    "Static Bloom": ("electronic|house|ambient", 0.55, 0.18),
    "Kite District": ("city pop|j-pop|funk", 0.5, 0.10),
    "Lumen Avenue": ("r&b|soul|funk", 0.5, 0.15),
    "Night Ferry": ("hip hop|rap|lo-fi", 0.45, 0.20),
    "The Quiet Engines": ("post-rock|math rock|ambient", 0.4, 0.12),
    "Ember Radio": ("punk|pop punk|rock", 0.2, 0.35),
    "Copper Fields": ("country|americana|folk", 0.15, 0.40),
}
SONGS_PER_ARTIST = 14
# Songs of the listener's artists that they never play (these numbers in each artist's chart), which the pool has.
NEVER_PLAYED = (4, 9, 13)
# Songs that also exist as another version, which the recommender merges into one song.
VERSIONS = 10
VERSION_SUFFIXES = (" - Live", " (Acoustic)", " - 2024 Remaster")

# The pool's other artists, as fetch_candidates.py finds them on Last.fm: similar to the listener's artists (with
# Last.fm's match), top artists of the listener's main genres, and of genres they never play (the genre menu).
SIMILAR_ARTISTS = {
    "Chrome Valentine": ("synth-pop|new wave|electronic", {"Synth Coast": 0.86, "Marble Arcade": 0.30}),
    "Polar Arcade": ("synth-pop|electronic|dance", {"Synth Coast": 0.71, "Static Bloom": 0.42}),
    "Glass Cathedral": ("shoegaze|dream pop|noise pop", {"Marble Arcade": 0.83}),
    "Soft Satellite": ("dream pop|indie pop|bedroom pop", {"Marble Arcade": 0.77, "Saltwater Choir": 0.40}),
    "Rust Belt Saints": ("indie rock|rock|garage rock", {"Granite Hours": 0.80}),
    "Hollow Lighthouse": ("alternative rock|indie rock|post-punk", {"Granite Hours": 0.66, "Ember Radio": 0.25}),
    "Birch and Bell": ("indie folk|folk|chamber pop", {"Paper Lanterns": 0.84, "Saltwater Choir": 0.35}),
    "Willow Post": ("folk|singer-songwriter|americana", {"Paper Lanterns": 0.69, "Copper Fields": 0.30}),
    "Blue Hour Quartet": ("jazz|bossa nova|cool jazz", {"Velvet Tram": 0.88}),
    "Saffron Keys": ("jazz|soul|jazz fusion", {"Velvet Tram": 0.72, "Lumen Avenue": 0.38}),
    "Starlit Mile": ("k-pop|pop|dance-pop", {"Neon Harbor": 0.81}),
    "Prism Avenue": ("k-pop|dance|electropop", {"Neon Harbor": 0.64, "Kite District": 0.22}),
    "Lemon Tree Parade": ("indie pop|pop|twee pop", {"Saltwater Choir": 0.78}),
    "Harbour Kids": ("indie pop|indie rock|surf rock", {"Saltwater Choir": 0.58, "Granite Hours": 0.33}),
    "Circuit Garden": ("house|electronic|deep house", {"Static Bloom": 0.79}),
    "Low Orbit Club": ("ambient|electronic|downtempo", {"Static Bloom": 0.70, "The Quiet Engines": 0.36}),
    "Sunset Transit": ("city pop|funk|j-pop", {"Kite District": 0.85}),
    "Moonlit Parlour": ("j-pop|city pop|pop", {"Kite District": 0.68, "Neon Harbor": 0.27}),
    "Honey Static": ("r&b|soul|neo-soul", {"Lumen Avenue": 0.76}),
    "Velour Club": ("funk|soul|disco", {"Lumen Avenue": 0.62, "Kite District": 0.30}),
    "Concrete Psalms": ("hip hop|rap|boom bap", {"Night Ferry": 0.74}),
    "Midnight Laundromat": ("lo-fi|hip hop|chillhop", {"Night Ferry": 0.69, "Static Bloom": 0.28}),
    "Northern Drift": ("post-rock|ambient|shoegaze", {"The Quiet Engines": 0.82, "Marble Arcade": 0.26}),
    "Tidal Archive": ("math rock|post-rock|indie rock", {"The Quiet Engines": 0.65}),
    "Static Youth": ("punk|pop punk|hardcore punk", {"Ember Radio": 0.77}),
    "Fuse Box Hearts": ("pop punk|rock|emo", {"Ember Radio": 0.70}),
    "Dust Road Revival": ("country|americana|bluegrass", {"Copper Fields": 0.80}),
    "Prairie Telegraph": ("americana|folk|country", {"Copper Fields": 0.66, "Paper Lanterns": 0.31}),
}
GENRE_ARTISTS = {
    "Velvet Meridian": ("dream pop|shoegaze|ethereal wave", "dream pop"),
    "Orchid Telephone": ("synth-pop|electropop|new wave", "synth-pop"),
    "Atlas Choir": ("indie rock|post-punk|rock", "indie rock"),
    "Lantern Society": ("indie folk|folk|chamber folk", "indie folk"),
    "Golden Ratio Club": ("electronic|idm|ambient", "electronic"),
    "Cobalt Parade": ("indie pop|jangle pop|pop", "indie pop"),
}
MENU_ARTISTS = {
    "Iron Cathedral": ("heavy metal|metal|hard rock", "metal"),
    "Bassline Theory": ("drum and bass|electronic|jungle", "drum and bass"),
    "Marigold Strings": ("classical|baroque|chamber music", "classical"),
    "Rumba Royale": ("salsa|latin|cumbia", "latin"),
    "Island Frequency": ("reggae|dub|ska", "reggae"),
    "Sunbeam Collective": ("afrobeats|afropop|dance", "afrobeats"),
    "Warehouse Theory": ("techno|electronic|minimal techno", "techno"),
    "Silver Monsoon": ("c-pop|mandopop|pop", "c-pop"),
}
# Artists who make Asian music: the pool's asian column, which the Less Western dial uses.
ASIAN = {"Neon Harbor", "Kite District", "Starlit Mile", "Prism Avenue", "Sunset Transit", "Moonlit Parlour", "Silver Monsoon"}
# The pool keeps this many top songs of each artist new to the listener.
TRACKS_PER_ARTIST = 10
# The listener's own artists have more songs on Last.fm than they ever played: this many more each, below the
# songs of the catalogue in their charts.
SEED_EXTRA = 16
# More artists for the pool, with names made of NAME_FIRST and NAME_SECOND: MORE_SIMILAR more similar to each of
# the listener's artists, GENRE_DEPTH top artists of each of their TOP_GENRES most played main genres, and
# MENU_DEPTH of each genre of MENU_GENRES (the genre menu of fetch_candidates.py). A real pool is big next to the
# SHORTLIST of songs that the model learned from ratings reorders (about 75,000 songs on the author's data), and
# this one has to be too, or the dials would hardly change a list.
MORE_SIMILAR = 9
TOP_GENRES = 6
GENRE_DEPTH = 14
MENU_DEPTH = 8
MENU_GENRES = {
    "metal": "metal|heavy metal|hard rock", "techno": "techno|electronic|minimal techno",
    "drum and bass": "drum and bass|jungle|electronic", "classical": "classical|baroque|chamber music",
    "latin": "latin|salsa|reggaeton", "reggae": "reggae|dub|ska", "afrobeats": "afrobeats|afropop|dance",
    "edm": "edm|electro house|dance", "trance": "trance|progressive trance|electronic",
    "blues": "blues|electric blues|soul", "gospel": "gospel|soul|r&b", "k-pop": "k-pop|pop|dance-pop",
    "j-rock": "j-rock|rock|alternative rock", "c-pop": "c-pop|mandopop|pop", "j-pop": "j-pop|pop|anime",
    "trap": "trap|hip hop|rap", "grunge": "grunge|alternative rock|rock", "emo": "emo|pop punk|rock",
    "disco": "disco|funk|dance", "opera": "opera|classical|vocal", "bluegrass": "bluegrass|country|folk",
    "ambient": "ambient|drone|electronic", "lo-fi": "lo-fi|chillhop|hip hop", "bossa nova": "bossa nova|jazz|mpb",
}
# Genres of Asian music: their artists count as Asian, as do those similar to the listener's Asian artists.
ASIAN_GENRES = {"k-pop", "j-pop", "j-rock", "c-pop", "mandopop", "city pop"}
NAME_FIRST = [
    "Apricot", "Arctic", "Atlas", "Autumn", "Azure", "Basalt", "Birch", "Bronze", "Canyon", "Cedar", "Cinder",
    "Citrus", "Clover", "Cobalt", "Comet", "Crystal", "Delta", "Driftwood", "Fable", "Fern", "Flint", "Glacier",
    "Hazel", "Indigo", "Jade", "Juniper", "Lagoon", "Lunar", "Magnolia", "Maple", "Meadow", "Mercury", "Mint",
    "Mosaic", "Oak", "Onyx", "Opal", "Orchid", "Pebble", "Pine", "Quartz", "Saffron", "Sable", "Sage", "Scarlet",
    "Sienna", "Solar", "Sparrow", "Tangerine", "Thistle", "Topaz", "Tundra", "Velour", "Willow", "Zephyr",
]
NAME_SECOND = [
    "Avenue", "Ballroom", "Bureau", "Cassette", "Choir", "Circuit", "Club", "Collective", "Company", "Cinema",
    "Coast", "District", "Drive", "Express", "Factory", "Fields", "Hotel", "Kids", "Lane", "League", "Machine",
    "Motel", "Orchestra", "Parade", "Parlour", "Quartet", "Radio", "Revival", "Riot", "Room", "Saints", "Signal",
    "Society", "Station", "Telephone", "Theory", "Transit", "Union", "Voyage", "Waves",
]

# How the made-up listener rates songs new to them. The recommender doesn't know this rule: the model learned
# from the ratings has to find it. A like is likelier for a song whose main genre is in LOVED_GENRES, less likely
# for SHUNNED_GENRES, and a little likelier for songs few people know.
RATED_LISTS = 14
LOVED_GENRES = {"dream pop", "shoegaze", "jazz", "bossa nova", "city pop", "post-rock"}
SHUNNED_GENRES = {"country", "americana", "punk", "pop punk", "k-pop", "house"}
NEUTRAL_SHARE = 0.05

# A time with an offset other than UTC's: the demo's data must have none, or it would tell your time zone.
NOT_UTC = re.compile(r"\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-](?!00:00)\d{2}:\d{2}")
# Songs like each song of the history are worked out for this many places, enough for a few Next songs.
LIKE_PLACES = 30
# The numbers of the Your data page, as the website works them out (app_pages/your_data.py).
TOP_GENRE_COUNT = 5
MIN_YEAR_HOURS = 20
MIN_GENRE_RATINGS = 8

ADJECTIVES = [
    "Amber", "Silver", "Quiet", "Electric", "Hollow", "Golden", "Velvet", "Midnight", "Northern", "Paper", "Glass",
    "Neon", "Lucid", "Wild", "Distant", "Burning", "Frozen", "Secret", "Faded", "Bright", "Gentle", "Restless",
    "Crimson", "Lonely", "Silent", "Endless", "Sunday", "Broken", "Easy", "Little", "Summer", "Winter", "Last",
    "First", "Slow", "Blue", "Green", "Ivory", "Salt", "Coral",
]
NOUNS = [
    "Harbor", "Signal", "Garden", "Avenue", "River", "Window", "Engine", "Letters", "Orbit", "Echo", "Tide", "Lantern",
    "Meadow", "Skyline", "Compass", "Mirror", "Season", "Station", "Horizon", "Carousel", "Postcard", "Thunder",
    "Morning", "Satellite", "Highway", "Island", "Fever", "Weather", "Machine", "Holiday", "Daydream", "Paradise",
    "Shadows", "Waves", "Lights", "Streets", "Hearts", "Stars", "Rooms", "Roads",
]


@contextlib.contextmanager
def caught():
    """What the recommender prints, kept instead (its CLIs reconfigure stdout, which StringIO can't do)."""
    out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    with contextlib.redirect_stdout(out):
        yield out


def text_of(out: io.TextIOWrapper) -> str:
    out.flush()
    return out.buffer.getvalue().decode("utf-8")


def titles(rng: np.random.Generator, count: int, taken: set[str]) -> list[str]:
    """count made-up song titles, none of them inside another (so that "artist - title" finds one song). taken
    holds titles used already; once every pair of words is, other artists' titles may be used again."""
    chosen = []
    while len(chosen) < count:
        title = f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}"
        fresh = title not in taken or len(taken) >= len(ADJECTIVES) * len(NOUNS)
        if fresh and not any(title in other or other in title for other in chosen):
            chosen.append(title)
            taken.add(title)
    return chosen


def catalogue(rng: np.random.Generator) -> pd.DataFrame:
    """Every song of the listener's artists: title, track id, genres, length, chart place, when it came out, and
    how skippable it is. A few songs get a second version (another track id), to be merged."""
    days = int(YEARS * 365.25)
    rows, taken = [], set()
    for artist, (genres, _, _) in HISTORY_ARTISTS.items():
        names = titles(rng, SONGS_PER_ARTIST, taken)
        released = np.sort(rng.integers(0, int(days * 0.9), SONGS_PER_ARTIST))
        released[:6] = -1  # out before the history starts
        rng.shuffle(released)
        for place, title in enumerate(names, 1):
            rows.append({
                "artist": artist, "title": title, "genres": genres, "place": place,
                "ms": int(rng.integers(150_000, 290_000)),
                "released": 10**9 if place in NEVER_PLAYED else int(released[place - 1]),
                "skippable": float(np.clip(rng.lognormal(0, 0.5), 0.3, 3.0)),
                "version": None,
            })
    songs = pd.DataFrame(rows)
    versions = songs[songs.released < 10**9].sample(VERSIONS, random_state=SEED)
    extra = versions.assign(
        version=[str(title) for title in versions.title],
        title=[title + VERSION_SUFFIXES[number % len(VERSION_SUFFIXES)] for number, title in enumerate(versions.title)],
    )
    songs = pd.concat([songs, extra], ignore_index=True)
    songs["track_id"] = [f"{number + 1:022d}" for number in range(len(songs))]
    return songs


def history(songs: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    """The streaming history, as Spotify exports it: one record per play, written when the play stopped."""
    days = int(YEARS * 365.25)
    start = int(datetime.fromisoformat(START).replace(tzinfo=timezone.utc).timestamp() * 1000)
    artists = list(HISTORY_ARTISTS)
    liking = np.array([HISTORY_ARTISTS[artist][1] for artist in artists])
    breaks = set()
    for _ in range(BREAKS):
        first = int(rng.integers(30, days - 30))
        breaks |= set(range(first, first + int(rng.integers(8, 22))))
    phases = {}
    originals = songs[songs.version.isna()]
    versions = {row.version: row for row in songs[songs.version.notna()].itertuples()}
    weekday, weekend = (np.array(hours) / np.sum(hours) for hours in (WEEKDAY_HOURS, WEEKEND_HOURS))
    by_artist = {artist: originals[originals.artist == artist] for artist in artists}
    records, busy_until = [], 0

    def pick(artist: str, day: int, way: str = "chart"):
        """One of the artist's songs out by then: by chart place (new songs more), any (shuffle), or a hit."""
        out = by_artist[artist][by_artist[artist].released <= day]
        if way == "hit":
            out = out.nsmallest(HITS, "place")
        weights = 1 / out.place.to_numpy() ** 0.9 * np.where(day - out.released.to_numpy() < 45, 3.0, 1.0)
        if way == "any":
            weights = np.ones(len(out))
        return out.iloc[rng.choice(len(out), p=weights / weights.sum())]

    for day in range(days):
        if day in breaks:
            continue
        phase = day // PHASE_DAYS
        if phase not in phases:
            phases[phase] = rng.choice(artists, FAVOURITES, replace=False, p=liking**2 / (liking**2).sum())
        date = start + day * 86_400_000
        is_weekend = (day % 7) >= 5  # START is a Monday
        growth = 0.7 + 0.6 * day / days
        shuffle_share = 0.05 + 0.55 * max(0.0, (day - 0.7 * days) / (0.3 * days))
        for _ in range(rng.poisson(SESSIONS_PER_DAY * growth * (1.25 if is_weekend else 1.0))):
            hour = int(rng.choice(24, p=weekend if is_weekend else weekday))
            clock = max(date + (hour * 60 + int(rng.integers(0, 60))) * 60_000, busy_until + 31 * 60_000)
            shuffle = bool(rng.random() < shuffle_share)
            previous_end = "clickrow"
            current, last = phases[phase], phases.get(phase - 1, phases[phase])
            for _ in range(3 + rng.poisson(6)):
                share = rng.random()
                if shuffle:
                    artist = str(rng.choice(np.concatenate([current, last])))
                    song = pick(artist, day, "any")
                elif share < FAVOURITE_SHARE + FADING_SHARE:
                    artist = str(rng.choice(current if share < FAVOURITE_SHARE else last))
                    song = pick(artist, day)
                else:
                    artist = str(rng.choice(artists, p=liking**3 / (liking**3).sum()))
                    song = pick(artist, day, "hit")
                if song.title in versions and rng.random() < 0.25:
                    song = versions[song.title]
                skip = HISTORY_ARTISTS[artist][2] * song.skippable * (0.6 if artist in phases[phase] else 1.0)
                chance = rng.random()
                if chance < min(skip, 0.8):
                    ms, reason = int(rng.integers(3_000, 25_000)), "fwdbtn"
                elif chance < min(skip, 0.8) + 0.06:
                    ms, reason = int(song.ms * rng.uniform(0.4, 0.85)), "logout"
                elif chance < min(skip, 0.8) + 0.11:
                    ms, reason = int(song.ms * rng.uniform(0.35, 0.7)), "fwdbtn"
                else:
                    ms, reason = int(song.ms), "trackdone"
                clock += ms
                records.append({
                    "ts": datetime.fromtimestamp(clock / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "platform": "android", "ms_played": ms, "conn_country": "ZZ", "ip_addr": "192.0.2.1",
                    "master_metadata_track_name": song.title, "master_metadata_album_artist_name": artist,
                    "master_metadata_album_album_name": f"{artist} Album", "spotify_track_uri": f"spotify:track:{song.track_id}",
                    "episode_name": None, "episode_show_name": None, "spotify_episode_uri": None, "audiobook_title": None,
                    "audiobook_uri": None, "audiobook_chapter_uri": None, "audiobook_chapter_title": None,
                    "reason_start": previous_end, "reason_end": reason, "shuffle": shuffle, "skipped": None,
                    "offline": False, "offline_timestamp": None, "incognito_mode": False,
                })
                previous_end = "trackdone" if reason == "trackdone" else reason
                clock += int(rng.integers(0, 3_000))
            busy_until = clock
    return records


def build_database(records: list[dict], songs: pd.DataFrame) -> Path:
    """The made-up export and its genres through build_database.py and build_gold.py, as you would run them."""
    export = SCRATCH / "export"
    export.mkdir(parents=True)
    years = pd.Series([record["ts"][:4] for record in records])
    for year in sorted(years.unique()):
        chosen = [record for record, in_year in zip(records, years == year) if in_year]
        (export / f"Streaming_History_Audio_{year}_0.json").write_text(json.dumps(chosen), encoding="utf-8")
    genres = songs.assign(
        track_name=songs.title, artist_name=songs.artist, album_name=songs.artist + " Album",
        genre=songs.genres.str.split("|").str[0], genre_source="artist",
    )
    genres[["track_id", "track_name", "artist_name", "album_name", "genre", "genres", "genre_source"]].to_csv(
        SCRATCH / "track_genres.csv", index=False
    )
    db = SCRATCH / "spotify.duckdb"
    for script, *options in (
        ("scripts/build_database.py", "--data-dir", export, "--techlog-dir", SCRATCH / "no technical log"),
        ("scripts/build_gold.py", "--genres", SCRATCH / "track_genres.csv"),
    ):
        done = subprocess.run([sys.executable, "-X", "utf8", script, *map(str, options), "--db", str(db)], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8", check=False)
        if done.returncode != 0:
            sys.exit(f"{script} failed on the made-up export:\n{done.stdout}{done.stderr}")
        for line in done.stdout.splitlines():
            if line.startswith("WARNING:"):
                print(f"  {script}: {line}")
    return db


def pool_artists(top_genres: list[str], rng: np.random.Generator) -> list[tuple]:
    """The pool's artists, as fetch_candidates.py finds them: (name, genres, Last.fm's similar_to links, source,
    via_genre, asian). The named ones first, then made-up names for the rest."""
    taken = set(HISTORY_ARTISTS) | set(SIMILAR_ARTISTS) | set(GENRE_ARTISTS) | set(MENU_ARTISTS)

    def name() -> str:
        while True:
            made = f"{rng.choice(NAME_FIRST)} {rng.choice(NAME_SECOND)}"
            if made not in taken:
                taken.add(made)
                return made

    artists = [(artist, genres, {}, "seed", None, artist in ASIAN) for artist, (genres, _, _) in HISTORY_ARTISTS.items()]
    artists += [(artist, genres, links, "similar", None, artist in ASIAN) for artist, (genres, links) in SIMILAR_ARTISTS.items()]
    for seed, (genres, _, _) in HISTORY_ARTISTS.items():
        first, second, third = genres.split("|")
        for _ in range(MORE_SIMILAR):
            links = {seed: round(float(rng.uniform(0.2, 0.8)), 3)}
            if rng.random() < 0.3:
                links[str(rng.choice([other for other in HISTORY_ARTISTS if other != seed]))] = round(float(rng.uniform(0.1, 0.4)), 3)
            mixed = [first, second, third] if rng.random() < 0.6 else [second, first, third]
            artists.append((name(), "|".join(mixed), links, "similar", None, seed in ASIAN))
    artists += [(artist, genres, {}, "genre", via, artist in ASIAN) for artist, (genres, via) in GENRE_ARTISTS.items()]
    by_main = {genres.split("|")[0]: genres for genres, _, _ in HISTORY_ARTISTS.values()}
    for genre in top_genres:
        for _ in range(GENRE_DEPTH):
            artists.append((name(), by_main.get(genre, genre), {}, "genre", genre, genre in ASIAN_GENRES))
    artists += [(artist, genres, {}, "menu", via, artist in ASIAN) for artist, (genres, via) in MENU_ARTISTS.items()]
    for genre, genres in MENU_GENRES.items():
        for _ in range(MENU_DEPTH):
            artists.append((name(), genres, {}, "menu", genre, genre in ASIAN_GENRES))
    return artists


def write_pool(path: Path, songs: pd.DataFrame, pool_date: pd.Timestamp, top_genres: list[str], rng: np.random.Generator) -> None:
    """The now pool of candidates.csv, as fetch_candidates.py would write it from Last.fm."""
    taken = set(songs.title)
    originals = songs[songs.version.isna()]
    rows = []
    for artist, genres, links, source, via, asian in pool_artists(top_genres, rng):
        if source == "seed":
            names = originals[originals.artist == artist].sort_values("place").title.tolist()
            names += titles(rng, SEED_EXTRA, taken | set(names))
        else:
            names = titles(rng, TRACKS_PER_ARTIST, taken)
        median = {"seed": 400_000, "similar": 150_000, "genre": 900_000, "menu": 1_500_000}[source]
        audience = rng.lognormal(np.log(median), 0.9)
        for place, title in enumerate(names, 1):
            listeners = int(audience / place**0.85 * rng.lognormal(0, 0.25))
            rows.append({
                "pool": "now", "pool_date": pool_date.isoformat(), "artist_name": artist, "track_name": title,
                "artist_rank": place, "listeners": listeners, "playcount": int(listeners * rng.uniform(3, 12)),
                "genres": genres, "is_seed": source == "seed", "similar_to": json.dumps(links), "source": source,
                "via_genre": via or "NA", "favourite_link": 0.0, "asian": asian,
            })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def like_chance(row: pd.Series, median_listeners: float) -> float:
    """The made-up listener's chance of liking a song new to them (see LOVED_GENRES)."""
    genres = row.genres.split(", ") if row.genres else []
    if genres and genres[0] in LOVED_GENRES:
        chance = 0.85
    elif LOVED_GENRES & set(genres):
        chance = 0.45
    elif genres and genres[0] in SHUNNED_GENRES:
        chance = 0.05
    else:
        chance = 0.2
    return min(chance + (0.1 if row.listeners < median_listeners else 0.0), 0.95)


def rate_lists(library, rng: np.random.Generator):
    """RATED_LISTS lists of songs new to the listener, each rated before the next is made, as on the website:
    the next list leaves out the songs shown and gives other artists a turn. Returns the library with the ratings."""
    shown, artist_songs, liked_artists = set(), {}, set()
    median_listeners = library.candidates.listeners.median()
    for _ in range(RATED_LISTS):
        library = library._replace(ratings=load_ratings())
        resting = {artist for artist, keys in artist_songs.items() if len(keys) >= MAX_PER_ARTIST} - liked_artists
        with caught():
            table = recommend_songs(top_n=10, library=library, exclude=frozenset(shown), exclude_artists=frozenset(resting))
        list_id = remember_list(table, "discover")
        actions = {"like": [], "neutral": [], "dislike": []}
        for position, row in enumerate(table.itertuples(), 1):
            if rng.random() < NEUTRAL_SHARE:
                actions["neutral"].append(str(position))
            elif rng.random() < like_chance(pd.Series(row._asdict()), median_listeners):
                actions["like"].append(str(position))
                liked_artists.add(row.artist_key)
            else:
                actions["dislike"].append(str(position))
        for action, positions in actions.items():
            if positions:
                rate(positions, action, list_id=list_id)
        shown |= set(table.song_key)
        for artist, key in zip(table.artist_key, table.song_key):
            artist_songs.setdefault(artist, set()).add(key)
    return library._replace(ratings=load_ratings())


def every_song(mode: str, library, **settings) -> pd.DataFrame:
    """A recommend_songs() table of every candidate in ranked order, with no cap per artist and no variety: the
    page applies both (and the dials) itself."""
    cap = recommend.MAX_PER_ARTIST
    recommend.MAX_PER_ARTIST = 10**9
    try:
        with caught():
            return recommend_songs(top_n=10**9, mode=mode, variety=0.0, library=library, **settings)
    finally:
        recommend.MAX_PER_ARTIST = cap


def shown_why(why: str) -> list[str]:
    """The reasons a card shows: the chances are in its meter, so the reasons that say them are left out
    (app_pages/recommendations.py info_html)."""
    return [part for part in why.split("; ") if part and "chance you" not in part]


def rounded(value, places: int | None = 6):
    """value rounded to places (None: in full), or None for a missing value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value) if places is None else round(float(value), places)


def discover_data(library, profile) -> dict:
    """The pool as the page ranks it: every song new to the listener, in the order discover() takes them, with the
    parts of the score that the dials don't change, and the model learned from ratings as plain numbers."""
    songs = discovery_candidates(library, profile.as_of)
    songs = songs[~songs.song_key.isin(rated_keys(library))].reset_index(drop=True)
    struck = {artist: name for artist, name in struck_artists(library).items() if artist in set(songs.artist_key)}
    songs = songs[~songs.artist_key.isin(struck.keys())].reset_index(drop=True)
    table = every_song("discover", library, genres=(), genre_boost=0.0, western=0.0, mainstream=0.0).set_index("song_key")
    liked = profile.artists.weight.clip(lower=0)
    sums = songs.similar_to.map(lambda links: sum(liked.get(seed.strip(), 0.0) * match for seed, match in links.items()))
    table = table.loc[songs.song_key]
    if not np.allclose(sums / sums.max(), table.similar_match.to_numpy()):
        sys.exit("The pool's similar_match differs from scoring.py's: update discover_data()")
    played = set(library.songs.artist_key.iloc[library.plays.song[library.plays.started_at < profile.as_of].unique()])
    artists, rows = {}, []
    # The parts in full: the page adds them up again, and a rounded part could swap two songs. Everything but
    # the title, listeners and chart place is the artist's, so it is kept once per artist.
    for song, (key, item), similar in zip(songs.itertuples(), table.iterrows(), sums):
        why = shown_why(item.why)
        listeners = recommend.count_text(int(song.listeners))
        if why[-1] != f"#{song.artist_rank} of their songs on Last.fm, {listeners} listeners":
            sys.exit(f"The reasons of {item.title} end unexpectedly ({why[-1]}): update discover_data()")
        artist = {
            "name": item.artist, "key": song.artist_key, "genres": genre_list(song.genres),
            "genre_match": float(item.genre_match), "artist_match": float(item.artist_match), "similar_sum": float(similar),
            "asian": bool(song.asian), "played": song.artist_key in played, "why_head": why[:-2], "genre_phrase": why[-2],
        }
        if artists.setdefault(song.artist_key, artist) != artist:
            sys.exit(f"Songs of {item.artist} differ in more than title, listeners and place: update discover_data()")
        rows.append([item.title, list(artists).index(song.artist_key), key, int(song.listeners), int(song.artist_rank), listeners])
    learned = feedback_model(library.ratings, library.candidates)
    model = None
    if learned.model is not None:
        maxabs, regression = learned.model[0], learned.model[-1]
        model = {
            "numbers": learned.numbers, "mean": learned.scaler.mean_.tolist(), "scale": learned.scaler.scale_.tolist(),
            "genres": sorted(learned.genres, key=learned.genres.get), "maxabs": maxabs.scale_.tolist(),
            "coef": regression.coef_[0].tolist(), "intercept": float(regression.intercept_[0]),
        }
    rated = current_ratings(library.ratings)
    rated = rated[rated["mode"] == "discover"]
    pool = library.candidates[library.candidates.pool == "now"].drop_duplicates("song_key")
    return {
        # genre_match divides by the length of your profile's genre weights over the genres of the history and
        # the songs being ranked, so the page needs them all to redo it when songs are left out.
        "profile_genres": [[genre, float(weight)] for genre, weight in profile.genres.items()],
        "artists": list(artists.values()), "songs": rows, "model": model, "model_ratings": learned.ratings,
        "struck": sorted(struck.values()),
        "pool_songs": len(pool), "pool_artists": int(pool.artist_key.nunique()),
        "genres": pool.genres.dropna().explode().value_counts().index.tolist(),
        "likes": int((rated.rating == 1).sum()), "dislikes": int((rated.rating == -1).sum()),
        "neutral": int((rated.rating == NEUTRAL).sum()),
    }


def rediscover_data(library) -> list[dict]:
    """Every song to rediscover, ranked by the model learned from the history, with its reasons."""
    table = every_song("rediscover", library)
    return [{
        "title": row.title, "artist": row.artist, "artist_key": row.artist_key, "song_key": row.song_key,
        "genres": row.genres.split(", ") if row.genres else [], "chance": float(row.chance),
        "why": shown_why(row.why),
    } for row in table.itertuples()]


def like_data(library) -> tuple[list[dict], dict]:
    """The songs of the history, most played first (as find_song() searches them), and for each the LIKE_PLACES
    songs most like it, other artists only and with the same artist: [song, similarity, session, genre,
    sessions together, 1 if played with the same songs (SAME_COMPANY)] each."""
    space = song_space(library, library.ends_at)
    original = recommend.song_space
    recommend.song_space = lambda *_: space  # the same for every song: worked out once
    plays = recommend.song_evidence(library, library.ends_at).plays
    songs = library.songs
    keys = recommend.song_key(songs.artist_name, songs.track_name)
    number_of = {key: number for number, key in enumerate(keys)}
    lists = {}
    try:
        for number, song in enumerate(songs.itertuples()):
            query = f"{song.artist_name} - {song.track_name}"
            if not song.is_music or find_song(library, query) != number:
                continue
            together = shared_sessions(space, number)
            lists[number] = {}
            for name, same in (("other", False), ("same", True)):
                with caught():
                    table = similar_songs(query, LIKE_PLACES, same, library=library)
                lists[number][name] = [
                    [number_of[row.song_key], rounded(row.similarity, 4), rounded(row.session_similarity, 4),
                     rounded(row.genre_similarity, 4), int(together[number_of[row.song_key]]),
                     int(row.session_similarity >= SAME_COMPANY)]
                    for row in table.itertuples()
                ]
    finally:
        recommend.song_space = original
    history_songs = [{
        "title": song.track_name, "artist": song.artist_name, "artist_key": song.artist_key, "song_key": keys.iloc[number],
        "genres": genre_list(song.genres), "plays": int(plays.iloc[number]),
    } for number, song in enumerate(songs.itertuples())]
    return history_songs, lists


def data_page(db: Path, library) -> dict:
    """The numbers and charts of the Your data page, worked out as app_pages/your_data.py does."""
    overview = data_overview(db)
    steps = overview.steps
    months = overview.months.assign(month=lambda frame: pd.to_datetime(frame.month)).set_index("month")
    every_month = pd.date_range(months.index.min(), months.index.max(), freq="MS")
    months = months.reindex(every_month).fillna({"plays": 0, "hours": 0.0}).rename_axis("month").reset_index()

    genre_years = overview.genre_years
    top = genre_years.dropna(subset=["genre"]).groupby("genre").hours.sum().nlargest(TOP_GENRE_COUNT).index.tolist()
    genres = genre_years.assign(label=genre_years.genre.map(lambda genre: genre if genre in top else "Other"))
    genres = genres.groupby(["year", "label"], as_index=False).hours.sum()
    totals = genres.groupby("year").hours.transform("sum")
    genres = genres[totals >= MIN_YEAR_HOURS].assign(share=lambda frame: frame.hours / frame.groupby("year").hours.transform("sum"))
    years = sorted(genres.year.unique())

    artists = overview.artist_years.assign(place=lambda frame: frame.groupby("year").cumcount() + 1)
    artists = artists[artists.year.isin(years)]
    ends = overview.end_years[overview.end_years.year.isin(years)]
    ends = ends.assign(share=lambda frame: frame.plays / frame.groupby("year").plays.transform("sum"))

    rated = current_ratings(library.ratings)
    opinions = rated[rated.rating.isin(OPINIONS)].assign(genre=lambda frame: frame.genres.map(lambda g: g[0] if len(g) else None))
    by_genre = opinions.dropna(subset=["genre"]).groupby("genre").rating.agg(
        likes=lambda r: int((r == 1).sum()), dislikes=lambda r: int((r == -1).sum())
    )
    by_genre = by_genre[by_genre.likes + by_genre.dislikes >= MIN_GENRE_RATINGS].reset_index()
    return {
        "steps": {name: (value.isoformat() if isinstance(value, pd.Timestamp) else rounded(value, 1) if isinstance(value, float)
                         else None if pd.isna(value) else int(value)) for name, value in steps.items()},
        "songs": len(library.songs),
        "layers": [[row.layer, int(row.tables), int(row.row_count)] for row in overview.layers.itertuples()],
        "file_mb": round(overview.file_bytes / 1e6, 1),
        "months": [[f"{row.month:%Y-%m}", rounded(row.hours, 1), int(row.plays)] for row in months.itertuples()],
        "hours": [[int(row.weekday), int(row.hour), rounded(row.hours, 1), int(row.plays)] for row in overview.hours.itertuples()],
        "genre_labels": top + ["Other"],
        "genre_years": [[int(row.year), row.label, rounded(row.hours, 1), rounded(row.share, 4)] for row in genres.itertuples()],
        "artist_years": [[int(row.year), int(row.place), row.artist, rounded(row.share, 4)] for row in artists.itertuples()],
        "end_years": [[int(row.year), row.end_type, int(row.plays), rounded(row.share, 4)] for row in ends.itertuples()],
        "ratings_by_genre": [[row.genre, row.likes, row.dislikes] for row in by_genre.itertuples()],
        "ratings_overall": rounded((opinions.rating == 1).mean(), 4) if len(opinions) else None,
    }


def models_page(library, profile) -> dict:
    """What the How it works page shows of the models and the profile."""
    model = load_ranker()
    learned = feedback_model(library.ratings, library.candidates)
    return {
        "ranker": {feature: rounded(weight, 4) for feature, weight in model.weights.weight.items()} if model else None,
        "trained": {name: int(value) for name, value in (ranker_summary() or {}).items() if name in ("cutoffs", "examples", "positives")},
        "discover": {"message": learned.message, "in_use": learned.model is not None,
                     "weights": {name: rounded(value, 4) for name, value in model_weights(learned).head(12).items()}
                     if learned.model is not None else {}},
        "rediscover": {"message": rediscover_feedback(library).message},
        "profile_genres": [[genre, rounded(weight, 4)] for genre, weight in profile.genres.head(10).items()],
        "profile_artists": [[row.artist_name, rounded(row.weight, 4)] for row in profile.artists.head(10).itertuples()],
    }


def references(library) -> dict:
    """Lists the recommender makes for some settings, to check the page's against (check_demo_engine.js)."""
    cases = {"discover": [], "rediscover": [], "like": []}
    for settings in ({}, {"familiarity": 0.2, "variety": 0.0, "mainstream": 0.3},
                     {"genres": ["jazz", "city pop"], "genre_boost": 0.6, "western": 0.4},
                     {"new_artists": True, "variety": 1.0, "top": 20}, {"familiarity": 1.0, "variety": 0.3, "top": 30}):
        cases["discover"].append(discover_case(library, settings, set(), set()))
    first = cases["discover"][0]
    cases["discover"].append(discover_case(library, {}, set(first["keys"]), resting_after(first)))
    for settings in ({}, {"variety": 0.5, "top": 15}):
        with caught():
            table = recommend_songs(top_n=settings.get("top", 10), mode="rediscover", variety=settings.get("variety", 0.0), library=library)
        cases["rediscover"].append({"settings": settings, "exclude": [], "rest": [], "keys": table.song_key.tolist(),
                                    "why": [shown_why(why) for why in table.why], "artists": table.artist_key.tolist()})
    first = cases["rediscover"][0]
    with caught():
        table = recommend_songs(top_n=10, mode="rediscover", variety=0.0, library=library,
                                exclude=frozenset(first["keys"]), exclude_artists=frozenset(resting_after(first)))
    cases["rediscover"].append({"settings": {}, "exclude": first["keys"], "rest": sorted(resting_after(first)),
                                "keys": table.song_key.tolist(), "why": [shown_why(why) for why in table.why]})
    songs = library.songs
    for number, same in ((0, False), (3, True), (17, False)):
        query = f"{songs.artist_name.iloc[number]} - {songs.track_name.iloc[number]}"
        with caught():
            table = similar_songs(query, 10, same, library=library)
        cases["like"].append({"query": query, "same": same, "keys": table.song_key.tolist(), "why": [why.split("; ") for why in table.why]})
    with caught():
        table = similar_songs("harbor", 10, False, library=library)
    cases["like"].append({"query": "harbor", "same": False, "keys": table.song_key.tolist(), "why": [why.split("; ") for why in table.why]})
    return cases


def resting_after(case: dict) -> set[str]:
    """Artists with MAX_PER_ARTIST songs in a list: on the demo, they rest on the next one (no like keeps them)."""
    counts = pd.Series(case["artists"]).value_counts()
    return set(counts.index[counts >= MAX_PER_ARTIST])


def discover_case(library, settings: dict, exclude: set[str], rest: set[str]) -> dict:
    with caught():
        table = recommend_songs(
            top_n=settings.get("top", 10), familiarity=settings.get("familiarity", FAMILIARITY),
            new_artists_only=settings.get("new_artists", False), variety=settings.get("variety", VARIETY["discover"]),
            mainstream=settings.get("mainstream", POPULARITY_WEIGHT), genres=tuple(settings.get("genres", ())),
            genre_boost=settings.get("genre_boost", GENRE_BOOST), western=settings.get("western", WESTERN),
            library=library, exclude=frozenset(exclude), exclude_artists=frozenset(rest),
        )
    return {"settings": settings, "exclude": sorted(exclude), "rest": sorted(rest), "keys": table.song_key.tolist(),
            "artists": table.artist_key.tolist(), "why": [shown_why(why) for why in table.why],
            "liked_chance": [rounded(value, None) for value in table.liked_chance]}


def constants() -> dict:
    """The recommender's settings that the page uses or names, read from the code so the page stays in step."""
    return {
        "MAX_PER_ARTIST": MAX_PER_ARTIST, "SHORTLIST": SHORTLIST, "VARIETY": VARIETY, "FAMILIARITY": FAMILIARITY,
        "POPULARITY_WEIGHT": POPULARITY_WEIGHT, "GENRE_BOOST": GENRE_BOOST, "WESTERN": WESTERN, "STRIKES": STRIKES,
        "MIN_EACH": MIN_EACH, "MODEL_MARGIN": MODEL_MARGIN, "HORIZON_DAYS": HORIZON_DAYS, "DORMANT_DAYS": DORMANT_DAYS,
        "HALF_LIFE_DAYS": HALF_LIFE_DAYS, "RATING_LISTENS": RATING_LISTENS, "WINDOW": WINDOW, "DIMENSIONS": DIMENSIONS,
        "SAME_COMPANY": SAME_COMPANY, "FOR_VARIETY": FOR_VARIETY, "FEATURES": len(FEATURES),
        "TOP_GENRE_COUNT": TOP_GENRE_COUNT, "MIN_YEAR_HOURS": MIN_YEAR_HOURS, "MIN_GENRE_RATINGS": MIN_GENRE_RATINGS,
    }


def copy_static() -> None:
    """The font (with its licence) and the logo, from static/, so that the site fetches nothing elsewhere."""
    (SITE / "fonts").mkdir(parents=True, exist_ok=True)
    for source, target in (("fonts/InterVariable.woff2", "fonts/InterVariable.woff2"),
                           ("fonts/Inter-LICENSE.txt", "fonts/Inter-LICENSE.txt"), ("logo.svg", "logo.svg")):
        shutil.copyfile(ROOT / "static" / source, SITE / target)


def check_engine(cases: dict) -> None:
    """Run the page's ranking (site/assets/engine.js) under node on the same settings, and compare."""
    node = shutil.which("node")
    if node is None:
        print("Note: node isn't installed, so the page's ranking wasn't checked against the recommender's")
        return
    path = SCRATCH / "references.json"
    path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    done = subprocess.run([node, str(ROOT / "scripts" / "check_demo_engine.js"), str(DATA_JS), str(path)],
                          capture_output=True, text=True, encoding="utf-8", check=False)
    print(done.stdout.rstrip())
    if done.returncode != 0:
        sys.exit(f"The page's ranking differs from the recommender's:\n{done.stderr}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if RECOMMENDER_DB.parent != SCRATCH / "user":
        sys.exit("The recommender's files would go elsewhere than the scratch folder: nothing was done")
    shutil.rmtree(SCRATCH, ignore_errors=True)
    (SCRATCH / "user").mkdir(parents=True)
    (SCRATCH / "config.toml").write_text('[home]\ntimezone = "UTC"\n', encoding="utf-8")
    rng = np.random.default_rng(SEED)
    songs = catalogue(rng)
    records = history(songs, rng)
    print(f"Made up a listener: {len(HISTORY_ARTISTS)} artists, {len(songs)} track ids, {len(records):,} plays "
          f"({records[0]['ts'][:10]} to {records[-1]['ts'][:10]}), in {SCRATCH}")
    db = build_database(records, songs)
    pool = SCRATCH / "candidates.csv"
    first = load_library(db=db, candidates=SCRATCH / "no pool.csv")
    listens = first.plays.groupby("song").listen.sum()
    main_genres = first.songs.genres.map(lambda genres: (genre_list(genres) or [None])[0]).iloc[listens.index].to_numpy()
    top_genres = listens.groupby(main_genres).sum().nlargest(TOP_GENRES).index.tolist()
    write_pool(pool, songs, first.ends_at, top_genres, np.random.default_rng(SEED + 1))
    library = load_library(db=db, candidates=pool)
    print(f"Built its database and a pool of {len(library.candidates):,} songs; {len(library.songs):,} songs played")

    ranker_cli.load_library = lambda *_, **__: library
    sys.argv = ["ranker"]
    with caught() as out:
        ranker_cli.main()
    print("Trained the rediscover model: " + next(line for line in text_of(out).splitlines() if line.startswith("Examples")))

    library = rate_lists(library, np.random.default_rng(SEED + 2))
    with caught():
        profile = profile_or_build(library)
    discover = discover_data(library, profile)
    print(f"Rated {RATED_LISTS} lists: {discover['likes']} likes and {discover['dislikes']} dislikes of songs new to it. "
          f"The model learned from them is {feedback_model(library.ratings, library.candidates).message}")
    history_songs, lists = like_data(library)
    plays = library.plays.started_at
    data = {
        "constants": constants(),
        "listener": {"first_play": f"{plays.min():%Y-%m-%d}", "last_play": f"{plays.max():%Y-%m-%d}"},
        "discover": discover,
        "rediscover": rediscover_data(library),
        "history": history_songs,
        "like": lists,
        "data": data_page(db, library),
        "models": models_page(library, profile),
    }
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if NOT_UTC.search(text):
        sys.exit(f"A time in the demo's data isn't in UTC ({NOT_UTC.search(text).group()}): nothing was written")
    DATA_JS.parent.mkdir(parents=True, exist_ok=True)
    DATA_JS.write_text(
        "// Made by scripts/build_demo_site.py from a made-up listener: every artist, song and rating is made up.\n"
        f"window.DEMO = {text};\n", encoding="utf-8",
    )
    copy_static()
    print(f"Wrote {DATA_JS.relative_to(ROOT)} ({DATA_JS.stat().st_size / 1e3:,.0f} kB): {len(discover['songs'])} songs new "
          f"to the listener, {len(data['rediscover'])} to rediscover, songs like each of {len(lists)}")
    check_engine(references(library))


if __name__ == "__main__":
    main()
