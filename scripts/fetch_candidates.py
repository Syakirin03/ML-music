"""Build pools of songs for the recommender to choose new music from, with Last.fm.

    python scripts/fetch_candidates.py [--db FILE] [--out FILE]

Writes candidates.csv with three pools of songs, each built from your taste at
one moment:

    now    the end of your listening history: python -m recommender.recommend
           recommends from this one
    valid  the start of the valid split of gold.plays
    test   the start of the test split

python -m recommender.evaluate uses the valid and test pools to replay the
past: it ranks a pool with your taste profile as of the pool's date, and checks
where the songs you went on to play ended up. A pool is built in six steps:

1. Seeds: your SEED_ARTISTS artists with the most listens before the pool's
   date, each listen counting half as much for every AFFINITY_HALF_LIFE_DAYS
   of age (gold.artists.affinity as of that date). Tracks that aren't music
   to you (gold.plays.is_music, from config.toml) are left out.
2. Artists: the seeds, and the SIMILAR_ARTISTS other artists that Last.fm's
   similar-artist lists (artist.getSimilar) point to most strongly from the
   seeds, by the sum over the seeds of the seed's weight times Last.fm's
   match, which runs from 0 to 1.
3. Genre artists: for each of your TOP_GENRES main genres with the most
   listens before the pool's date (counted the same way), the
   ARTISTS_PER_GENRE top artists of that genre on Last.fm (tag.getTopArtists)
   that aren't in the pool yet. They reach artists your top artists don't
   lead to.
4. Favourite artists, for the now pool only: Last.fm's similar artists of
   each of FAVOURITE_ARTISTS ([pool] in config.toml), and theirs in turn (two steps out), up to
   FAVOURITE_REACH artists not in the pool yet, the most strongly linked
   first. They reach the smaller bands of a scene you asked for more of. The
   valid and test pools leave them out: they mustn't know whom you came to
   like later, or the tests would flatter the recommender.
5. Genre menu artists, for the now pool only: for each of GENRE_MENU, a
   broad list of well-known genres, Last.fm's ARTISTS_PER_GENRE top artists
   of it that aren't in the pool yet. They let the website's "More of these
   genres" find songs in genres you barely play, such as techno, which steps
   1 to 4 hardly reach. The valid and test pools leave them out, so that
   python -m recommender.evaluate measures the pools your taste leads to.
6. Songs: each artist's TRACKS_PER_ARTIST most played songs on Last.fm
   (artist.getTopTracks), with the artist's genres (artist.getTopTags, read
   with the genre rules of fetch_genres.py).

The pools keep the songs you have played: the recommender leaves those out
itself, because it knows when you first played each one. Columns, one row per
pool and song:

    pool         now, valid or test
    pool_date    the moment the pool is for, UTC
    artist_name  as Last.fm writes it; for a seed, as Spotify does
    track_name   as Last.fm writes it
    artist_rank  1 for the artist's most played song on Last.fm, 2 for the next, ...
    listeners    Last.fm listeners of the song
    playcount    Last.fm plays of the song
    genres       the artist's genres, main one first, separated by |
    is_seed      the artist is one of the pool's seeds
    source       how the artist got in: seed, similar, genre, favourite or menu
    via_genre    for source genre or menu, the genre whose top artists it is among
    favourite_link  how strongly Last.fm links the artist to one of
                 FAVOURITE_ARTISTS, 0 to 1: its match, or for two steps out
                 the product of both matches; 0 in the valid and test pools.
                 The recommender doesn't score it
    asian        the artist makes Asian music: one of its TAGS_READ top
                 Last.fm tags names an Asian country or genre (ASIAN_TAGS), or
                 its name is in Japanese or Korean script. The recommender
                 can push the others down ("less Western")
    similar_to   JSON: the seeds whose Last.fm similar artists include this
                 artist, with Last.fm's match for each, e.g. {"Seed Artist": 0.85}

Missing values are written as NA. Last.fm's lists and numbers are today's, also
in the valid and test pools, so those pools know a little of what came after
their date: songs released since, and the listeners since.

Every answer is kept in lastfm_cache.jsonl, the cache fetch_genres.py uses, so
a stopped run resumes where it stopped, and reruns only ask Last.fm for what is
not cached yet. The first run makes about 4,000 requests, about 18 minutes.
Last.fm only gets artist names. The API key is read as in fetch_genres.py.

Without step 3, a pool could only offer artists that your top artists lead
to, and a taste you have only just started, a new favourite with a handful of
plays, leads nowhere yet.
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from build_gold import AFFINITY_HALF_LIFE_DAYS, CONFIG
from fetch_genres import CACHE_FILE, GENRE_LIST, MISSING, LastFm, LastFmError, genres_in, load_genres, read_api_key

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "processed" / "spotify.duckdb"
DEFAULT_OUT = ROOT / "data" / "processed" / "candidates.csv"

# Your top artists that each pool starts from.
SEED_ARTISTS = 100
# Similar artists asked for per seed. Changing it asks Last.fm again.
SIMILAR_PER_SEED = 30
# Similar artists that go into a pool, the most strongly pointed to first.
SIMILAR_ARTISTS = 400
# Songs per artist: its most played on Last.fm. 10 missed most later finds by
# artists already in the history, since fans go deep into albums; 50 costs no
# more requests. Changing it asks Last.fm again.
TRACKS_PER_ARTIST = 50
# Genre artists: your main genres with the most listens, and Last.fm's top artists asked for per genre.
TOP_GENRES = 20
ARTISTS_PER_GENRE = 20
# Artists whose Last.fm network joins the now pool, as Last.fm writes them ([pool] favourite_artists in
# config.toml), and how many new artists it may add.
FAVOURITE_ARTISTS = tuple(CONFIG["pool"]["favourite_artists"])
FAVOURITE_REACH = 150
# Genres whose top artists join the now pool (step 5), as Last.fm tags them, so that the website's
# "More of these genres" has songs in each. ARTISTS_PER_GENRE are asked for per genre.
GENRE_MENU = (
    "j-pop", "j-rock", "k-pop", "c-pop", "mandopop", "cantopop", "city pop", "anime", "vocaloid", "visual kei",
    "opm", "dangdut", "techno", "house", "deep house", "edm", "trance", "dubstep", "drum and bass", "electronic",
    "ambient", "lo-fi", "synth-pop", "hyperpop", "hip hop", "trap", "phonk", "r&b", "soul", "funk", "disco", "jazz",
    "blues", "classical", "metal", "metalcore", "punk", "pop punk", "emo", "shoegaze", "dream pop", "folk",
    "country", "reggae", "reggaeton", "latin", "afrobeats",
)
# Asian music, for the asian column: words of Asian countries and genres in an artist's top TAGS_READ Last.fm
# tags. "asian" alone is left out, and the short codes (k-pop, t-pop, ...) must be whole words: "alt-pop" isn't Thai.
ASIAN_TAGS = re.compile(
    r"japan|korea|chinese|china|taiwan|mandarin|canton|hong kong|thai|indonesia|malay|filipin|pinoy|\bopm\b|"
    r"vietnam|singapor|\b[jkct]-?pop\b|\b[jk]-?rock\b|\bk-?indie\b|\bk-?rap\b|anime|visual kei|vocaloid|city pop|"
    r"mandopop|cantopop|dangdut|kayokyoku|\benka\b"
)
TAGS_READ = 10
# Japanese kana, CJK ideographs and Korean hangul.
ASIAN_SCRIPT = re.compile(r"[぀-ヿ㐀-鿿가-힯]")

COLUMNS = (
    "pool", "pool_date", "artist_name", "track_name", "artist_rank", "listeners", "playcount", "genres", "is_seed", "similar_to",
    "source", "via_genre", "favourite_link", "asian",
)

# The moment of each pool: just after the last play for now, as in the
# recommender's Library.ends_at, and the first play of the valid and test splits.
POOLS_SQL = """
SELECT 'now' AS pool, max(started_at) + INTERVAL 1 SECOND AS pool_date FROM gold.plays
UNION ALL
SELECT split, min(started_at) FROM gold.plays WHERE split IN ('valid', 'test') GROUP BY split
ORDER BY pool_date
"""

SEEDS_SQL = """
SELECT artist_name, sum(listen * pow(0.5, epoch($pool_date - started_at) / 86400 / $half_life_days)) AS weight
FROM gold.plays
WHERE started_at < $pool_date AND is_music
GROUP BY artist_name
ORDER BY weight DESC, artist_name
LIMIT $seeds
"""

GENRES_SQL = """
SELECT genre, sum(listen * pow(0.5, epoch($pool_date - started_at) / 86400 / $half_life_days)) AS weight
FROM gold.plays
WHERE started_at < $pool_date AND is_music AND genre IS NOT NULL
GROUP BY genre
ORDER BY weight DESC, genre
LIMIT $genres
"""


def parse_similar(data: dict) -> list[list]:
    """[[artist, match], ...] from an artist.getSimilar answer."""
    artists = data.get("similarartists", {}).get("artist", [])
    if isinstance(artists, dict):  # a lone artist can come as an object instead of a list
        artists = [artists]
    return [[artist["name"], float(artist.get("match") or 0)] for artist in artists]


def parse_top_tracks(data: dict) -> list[list]:
    """[[title, listeners, playcount], ...], most played first, from an artist.getTopTracks answer."""
    tracks = data.get("toptracks", {}).get("track", [])
    if isinstance(tracks, dict):
        tracks = [tracks]
    return [[track["name"], int(track.get("listeners") or 0), int(track.get("playcount") or 0)] for track in tracks]


def parse_top_artists(data: dict) -> list[str]:
    """Artist names, top first, from a tag.getTopArtists answer."""
    artists = data.get("topartists", {}).get("artist", [])
    if isinstance(artists, dict):
        artists = [artists]
    return [artist["name"] for artist in artists]


def is_asian(name: str, tags: list[list]) -> bool:
    """Whether an artist makes Asian music, by its name's script or its top Last.fm tags (see ASIAN_TAGS)."""
    top = sorted(tags, key=lambda tag: -tag[1])[:TAGS_READ]
    return bool(ASIAN_SCRIPT.search(name)) or any(ASIAN_TAGS.search(tag.casefold()) for tag, _ in top)


def favourite_links(lastfm: LastFm, favourites: tuple[str, ...]) -> dict[str, tuple[str, float]]:
    """casefolded artist -> (name, link) for the artists Last.fm links to the favourites, one or two steps out."""
    links: dict[str, tuple[str, float]] = {}

    def add(name: str, strength: float) -> None:
        key = name.casefold()
        if key not in {f.casefold() for f in favourites} and strength > links.get(key, ("", 0.0))[1]:
            links[key] = (name, strength)

    for favourite in favourites:
        first = lastfm.answer("artist.getSimilar", {"artist": favourite, "limit": SIMILAR_PER_SEED}, "artists", parse_similar) or []
        for name, match in first:
            add(name, match)
            for name2, match2 in lastfm.answer("artist.getSimilar", {"artist": name, "limit": SIMILAR_PER_SEED}, "artists", parse_similar) or []:
                add(name2, match * match2)
    return links


def build_pool(
    lastfm: LastFm, genres: dict[str, str], pool: str, pool_date: datetime, seeds: list[tuple], top_genres: list[str],
    start: float, favourites: tuple[str, ...] = (), menu: tuple[str, ...] = (),
) -> tuple[list[dict], int, int]:
    """The pool's rows, how many artists are in it, and how many artists' requests failed."""
    weights = dict(seeds)
    seed_names = {name.casefold(): name for name in weights}
    links: dict[str, dict[str, float]] = defaultdict(dict)  # casefolded artist -> {seed: match}
    names: dict[str, str] = {}  # casefolded artist -> the name as Last.fm writes it
    failed = 0
    for seed in weights:
        similar = lastfm.answer("artist.getSimilar", {"artist": seed, "limit": SIMILAR_PER_SEED}, "artists", parse_similar)
        if similar is None:
            failed += 1
            continue
        for name, match in similar:
            key = name.casefold()
            if key != seed.casefold():
                links[key][seed] = match
                names.setdefault(key, name)
    strength = {key: sum(weights[seed] * match for seed, match in found.items()) for key, found in links.items()}
    others = sorted((key for key in strength if key not in seed_names), key=lambda key: (-strength[key], key))
    artists = [(name, "seed", None) for name in seed_names.values()] + [(names[key], "similar", None) for key in others[:SIMILAR_ARTISTS]]
    in_pool = {name.casefold() for name, _, _ in artists}

    def add_genre_artists(genre_names, source: str) -> None:
        nonlocal failed
        for genre in genre_names:
            top = lastfm.answer("tag.getTopArtists", {"tag": genre, "limit": ARTISTS_PER_GENRE}, "artists", parse_top_artists)
            if top is None:
                failed += 1
                continue
            for name in top:
                if name.casefold() not in in_pool:
                    in_pool.add(name.casefold())
                    artists.append((name, source, genre))

    add_genre_artists(top_genres, "genre")
    favoured = favourite_links(lastfm, favourites) if favourites else {}
    reach = sorted((key for key in favoured if key not in in_pool), key=lambda key: (-favoured[key][1], key))
    artists += [(favoured[key][0], "favourite", None) for key in reach[:FAVOURITE_REACH]]
    in_pool |= set(reach[:FAVOURITE_REACH])
    add_genre_artists(menu, "menu")

    rows = []
    for done, (artist, source, via_genre) in enumerate(artists, 1):
        tracks = lastfm.answer("artist.getTopTracks", {"artist": artist, "limit": TRACKS_PER_ARTIST}, "tracks", parse_top_tracks)
        tags = lastfm.top_tags("artist.getTopTags", {"artist": artist})
        if done % 100 == 0 or done == len(artists):
            minutes = (time.monotonic() - start) / 60
            print(f"  {pool} pool: {done:,} of {len(artists):,} artists, {lastfm.requests:,} Last.fm requests, {minutes:.0f} min", flush=True)
        if tracks is None or tags is None:
            failed += 1
            continue
        artist_genres = genres_in(tags, genres)
        asian = is_asian(artist, tags)
        similar_to = links.get(artist.casefold(), {})
        for rank, (title, listeners, playcount) in enumerate(tracks[:TRACKS_PER_ARTIST], 1):
            rows.append({
                "pool": pool,
                "pool_date": pool_date.astimezone(timezone.utc).isoformat(),
                "artist_name": artist,
                "track_name": title,
                "artist_rank": rank,
                "listeners": listeners,
                "playcount": playcount,
                "genres": "|".join(artist_genres) or MISSING,
                "is_seed": source == "seed",
                "similar_to": json.dumps(similar_to, ensure_ascii=False),
                "source": source,
                "via_genre": via_genre or MISSING,
                "favourite_link": round(favoured.get(artist.casefold(), ("", 0.0))[1], 4),
                "asian": asian,
            })
    return rows, len(artists), failed


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # the BOM lets Excel see UTF-8
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="DuckDB file built by build_gold.py")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV file to write")
    args = parser.parse_args()
    if not args.db.exists():
        sys.exit(f"Database not found: {args.db} (run build_database.py, then build_gold.py)")

    genres = load_genres(GENRE_LIST)
    api_key = read_api_key()
    con = duckdb.connect(str(args.db), read_only=True)
    try:
        pools = con.execute(POOLS_SQL).fetchall()
        seeds = {
            pool: con.execute(
                SEEDS_SQL, {"pool_date": pool_date, "half_life_days": AFFINITY_HALF_LIFE_DAYS, "seeds": SEED_ARTISTS}
            ).fetchall()
            for pool, pool_date in pools
        }
        top_genres = {
            pool: [genre for genre, _ in con.execute(
                GENRES_SQL, {"pool_date": pool_date, "half_life_days": AFFINITY_HALF_LIFE_DAYS, "genres": TOP_GENRES}
            ).fetchall()]
            for pool, pool_date in pools
        }
    except (duckdb.CatalogException, duckdb.BinderException) as error:
        sys.exit(f"{args.db.name} has no gold layer, or an old one: {error} (run build_gold.py)")
    finally:
        con.close()

    lastfm = LastFm(api_key, CACHE_FILE)
    rows: list[dict] = []
    summary = []
    failed = 0
    start = time.monotonic()
    try:
        for pool, pool_date in pools:
            pool_rows, artists, pool_failed = build_pool(
                lastfm, genres, pool, pool_date, seeds[pool], top_genres[pool], start,
                FAVOURITE_ARTISTS if pool == "now" else (), GENRE_MENU if pool == "now" else (),
            )
            rows += pool_rows
            failed += pool_failed
            summary.append(f"  {pool:<5} pool, as of {pool_date:%d %b %Y}: {artists:,} artists, {len(pool_rows):,} songs")
    except LastFmError as error:
        sys.exit(f"Stopped: {error}. The answers so far are saved in {CACHE_FILE.name}; run again to continue.")
    except KeyboardInterrupt:
        sys.exit(f"Interrupted. The answers so far are saved in {CACHE_FILE.name}; run again to continue.")
    finally:
        lastfm.close()

    try:
        write_csv(args.out, rows)
    except PermissionError:
        sys.exit(f"Could not write {args.out}: close it if another program has it open, then run again")
    print(f"Pools of songs ({lastfm.requests:,} Last.fm requests this run):")
    print("\n".join(summary))
    if failed:
        print(f"WARNING: requests for {failed:,} artists failed, so they are missing; run again to retry")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
