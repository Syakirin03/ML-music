"""Look up the genre of every track in silver.tracks with the Last.fm API.

    python scripts/fetch_genres.py [--db FILE] [--out FILE] [--limit N]

Writes track_genres.csv, one row per track in silver.tracks, most played first:

    track_id                             Spotify track id, as in silver.tracks
    track_name, artist_name, album_name  as in silver.tracks
    genre                                its main genre: the first of genres
    genres                               all its genres, most tagged first, separated by |
    genre_source                         where they come from: track, album, artist_tracks or artist (see below)

Missing values are written as NA.

Last.fm has no genre field, only tags that listeners give to tracks, albums
and artists: "k-pop" and "pop", but also "seen live", "2019" or "Korean". A tag
counts as a genre if it is in musicbrainz_genres.txt (the MusicBrainz genre
list), EXTRA_GENRES or GENRE_ALIASES below, or in [genres] of config.toml,
ignoring case, accents, spaces and punctuation, so "Hip-Hop", "hiphop" and
"hip hop" are all "hip hop". Each track gets the genres of the first of these
that has any:

    track          the track's tags, under its Spotify title, then without
                   what Spotify adds to titles, such as " - Remastered 2011"
                   or " (feat. ...)"
    album          its album's tags, the same way
    artist_tracks  the genres most common among the artist's other tracks here
                   that have track or album genres
    artist         the artist's tags

Tracks without genre tags at any level get NA, and so do the tracks only seen
in the technical log, which have no title or artist to look up.

Every Last.fm answer is kept in lastfm_cache.jsonl. A run that stops part way
resumes where it stopped, and later runs (after a new export, or after changing
the genre rules below) only ask Last.fm for what is not cached yet. Delete the
cache to ask for everything again.

The API key is read from the LASTFM_API_KEY environment variable, or else from
secrets/lastfm_api_key.txt.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import duckdb
import requests

ROOT = Path(__file__).resolve().parent.parent
# The project folder, for recommender.config: your settings in config.toml.
sys.path.insert(0, str(ROOT))
from recommender.config import CONFIG

DEFAULT_DB = ROOT / "data" / "processed" / "spotify.duckdb"
DEFAULT_OUT = ROOT / "data" / "processed" / "track_genres.csv"
CACHE_FILE = ROOT / "data" / "interim" / "lastfm_cache.jsonl"
GENRE_LIST = ROOT / "data" / "external" / "musicbrainz_genres.txt"
API_KEY_FILE = ROOT / "secrets" / "lastfm_api_key.txt"

API_URL = "https://ws.audioscrobbler.com/2.0/"
USER_AGENT = "MusicPreference/1.0 (genres for a personal listening history)"

# Last.fm allows 5 requests per second, averaged over five minutes.
SECONDS_BETWEEN_REQUESTS = 0.25
ATTEMPTS = 5
# After this many failed requests in a row the run stops: Last.fm is probably down.
MAX_FAILURES_IN_A_ROW = 5

# Last.fm error codes. 6 is its answer for an unknown track, album or artist.
# These are temporary: operation failed, service offline, temporary error,
# rate limit exceeded.
NOT_FOUND = 6
TEMPORARY_ERRORS = {8, 11, 16, 29}

# Tag weights are relative: an item's most used tag has 100. Weaker tags are
# often one listener's idea.
MIN_TAG_WEIGHT = 10

# Genre tags common on Last.fm that the MusicBrainz list does not have, and those of [genres] extra in
# config.toml.
EXTRA_GENRES = (
    "a cappella",
    "alternative",
    "anime",
    "indie",
    "rap",
    "soundtrack",
    "video game music",
    "vocaloid",
    "world music",
    *CONFIG["genres"]["extra"],
)

# Other names for a genre, and the genre they mean, then those of [genres] aliases in config.toml.
GENRE_ALIASES = {
    "a capella": "a cappella",
    "alt rock": "alternative rock",
    "alt-pop": "alternative pop",
    "alternative rap": "alternative hip hop",
    "alternative rnb": "alternative r&b",
    "anisong": "anime",
    "anison": "anime",
    "bollywood": "filmi",
    "ccm": "contemporary christian",
    "christian": "contemporary christian",
    "christmas": "christmas music",
    "contemporary rnb": "contemporary r&b",
    "dnb": "drum and bass",
    "electrorock": "electronic rock",
    "game music": "video game music",
    "hardcore rap": "hardcore hip hop",
    "j-indie": "indie",
    "japanese pop": "j-pop",
    "japanese rock": "j-rock",
    "k-indie": "indie",
    "k-rnb": "r&b",
    "k-rock": "rock",
    "kids": "children's music",
    "korean pop": "k-pop",
    "musicals": "musical",
    "nature": "nature sounds",
    "original soundtrack": "soundtrack",
    "ost": "soundtrack",
    "punk pop": "pop punk",
    "rhythm and blues": "r&b",
    "rnb": "r&b",
    "soundtracks": "soundtrack",
    "techno pop": "synth-pop",
    "vgm": "video game music",
    "video game": "video game music",
    "video games": "video game music",
    "world": "world music",
    "worship": "praise & worship",
    **CONFIG["genres"]["aliases"],
}

MISSING = "NA"
COLUMNS = ("track_id", "track_name", "artist_name", "album_name", "genre", "genres", "genre_source")

# What Spotify adds to titles, while Last.fm keeps most tags under the plain
# title: "Song - Remastered 2011", "Song (feat. X)", "Song [From "Film"]", "曲（OP）".
BRACKETED = re.compile(r"\s*[(\[（【][^)\]）】]*[)\]）】]")
DASH_SUFFIX = re.compile(r"\s+[-–—]\s+.*")

# What write_csv counts, in the order the summary lists them.
OUTCOMES = {
    "track": "from the track's tags",
    "album": "from its album's tags",
    "artist_tracks": "from the artist's other tracks",
    "artist": "from the artist's tags",
    "not_tagged": "NA: not on Last.fm, or no genre tags",
    "failed": "NA: Last.fm request failed (run again to retry)",
    "no_name": "NA: no title or artist (only seen in the technical log)",
}


class LastFmError(Exception):
    pass


class LastFm:
    """Requests to Last.fm: throttled, retried, and cached in a JSON lines file. fetch_candidates.py uses it too."""

    def __init__(self, api_key: str, cache_path: Path):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.cache: dict[tuple, dict] = {}
        text = cache_path.read_text(encoding="utf-8") if cache_path.exists() else ""
        for line in text.splitlines():
            try:
                entry = json.loads(line)
                self.cache[cache_key(entry.pop("method"), entry.pop("params"))] = entry
            except (json.JSONDecodeError, KeyError):
                pass  # a line cut short when a run was killed
        self.cache_file = cache_path.open("a", encoding="utf-8")
        if text and not text.endswith("\n"):
            self.cache_file.write("\n")
        self.next_request_at = 0.0
        self.requests = 0
        self.failures_in_a_row = 0

    def close(self) -> None:
        self.cache_file.close()
        self.session.close()

    def top_tags(self, method: str, params: dict) -> list[list] | None:
        """[[tag, weight], ...]; [] if Last.fm has no such item; None if the request failed."""
        return self.answer(method, params, "tags", parse_tags)

    def answer(self, method: str, params: dict, field: str, parse: Callable[[dict], list]) -> list | None:
        """parse(Last.fm's answer), cached under field; [] if Last.fm has no such item; None if the request failed."""
        key = cache_key(method, params)
        if key not in self.cache:
            answer = self.fetch(method, params, field, parse)
            if answer is None:
                return None
            self.cache[key] = answer
            self.cache_file.write(json.dumps({"method": method, "params": params, **answer}, ensure_ascii=False) + "\n")
            self.cache_file.flush()
        return self.cache[key].get(field, [])

    def fetch(self, method: str, params: dict, field: str, parse: Callable[[dict], list]) -> dict | None:
        """{field: parse(answer)}, or {"error": 6} if not found; None if every attempt failed."""
        for attempt in range(ATTEMPTS):
            if attempt:
                time.sleep(2**attempt)
            time.sleep(max(0.0, self.next_request_at - time.monotonic()))
            self.next_request_at = time.monotonic() + SECONDS_BETWEEN_REQUESTS
            self.requests += 1
            try:
                data = self.session.get(
                    API_URL,
                    params={"method": method, **params, "autocorrect": 1, "api_key": self.api_key, "format": "json"},
                    timeout=30,
                ).json()
            except (requests.RequestException, ValueError):
                continue  # no connection, or an error page instead of JSON
            error = data.get("error")
            if error in TEMPORARY_ERRORS:
                continue
            if error not in (None, NOT_FOUND):
                raise LastFmError(f"Last.fm error {error}: {data.get('message')}")
            self.failures_in_a_row = 0
            return {"error": error} if error else {field: parse(data)}
        self.failures_in_a_row += 1
        if self.failures_in_a_row >= MAX_FAILURES_IN_A_ROW:
            raise LastFmError(f"{self.failures_in_a_row} requests in a row failed, so Last.fm seems to be down")
        return None


def cache_key(method: str, params: dict) -> tuple:
    return (method, *sorted(params.items()))


def parse_tags(data: dict) -> list[list]:
    tags = data.get("toptags", {}).get("tag", [])
    if isinstance(tags, dict):  # a lone tag can come as an object instead of a list
        tags = [tags]
    return [[tag["name"], int(tag.get("count") or 0)] for tag in tags]


def plain_title(title: str) -> str:
    """'Song Title (with Another Artist) - Remix' -> 'Song Title'."""
    return " ".join(DASH_SUFFIX.sub("", BRACKETED.sub("", title)).split())


def genre_key(tag: str) -> str:
    """'Hip-Hop', 'hiphop' and 'hip hop' all give 'hiphop'; 'kayōkyoku' gives 'kayokyoku'."""
    tag = unicodedata.normalize("NFKD", tag.casefold()).replace("&", "and")
    return "".join(c for c in tag if c.isalnum())


def load_genres(path: Path) -> dict[str, str]:
    """Genre names by genre_key."""
    if not path.exists():
        sys.exit(f"Genre list not found: {path}")
    genres: dict[str, str] = {}
    for name in [*path.read_text(encoding="utf-8").splitlines(), *EXTRA_GENRES]:
        name = name.strip()
        key = genre_key(name)
        if key and not name.startswith("#"):
            genres.setdefault(key, name)
    for alias, name in GENRE_ALIASES.items():
        genres[genre_key(alias)] = name
    return genres


def genres_in(tags: list[list], genres: dict[str, str]) -> list[str]:
    """The genres among an item's tags, most used first, each once."""
    found: list[str] = []
    for tag, weight in sorted(tags, key=lambda tag: -tag[1]):
        genre = genres.get(genre_key(tag))
        if genre and weight >= MIN_TAG_WEIGHT and genre not in found:
            found.append(genre)
    return found


def common_genres(genre_lists: list[list[str]]) -> list[str]:
    """The genres in at least half as many of the lists as the most common one, most common first."""
    counts = Counter(genre for genres in genre_lists for genre in genres)
    top = max(counts.values())
    return [genre for genre, n in counts.most_common() if 2 * n >= top]


def queries(artist: str, title: str, album: str | None) -> list[tuple[str, str, dict]]:
    """The Last.fm queries for a track's tags and then its album's, in the order to try them: (level, method, params)."""
    result = []
    for level, name in (("track", title), ("album", album)):
        if not name:
            continue
        names = [name]
        if (plain := plain_title(name)) and plain != name:
            names.append(plain)
        result += [(level, f"{level}.getTopTags", {"artist": artist, level: n}) for n in names]
    return result


def read_api_key() -> str:
    key = os.environ.get("LASTFM_API_KEY", "").strip()
    if not key and API_KEY_FILE.exists():
        key = API_KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        sys.exit(f"No Last.fm API key: set LASTFM_API_KEY or save the key in {API_KEY_FILE}")
    return key


def read_tracks(db: Path) -> list[tuple]:
    """(track_id, track_name, artist_name, album_name) of every track, most played first."""
    if not db.exists():
        sys.exit(f"Database not found: {db} (run build_database.py)")
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute(
            "SELECT track_id, track_name, artist_name, album_name FROM silver.tracks ORDER BY play_count DESC, track_id"
        ).fetchall()
    finally:
        con.close()


def find_genres(lastfm: LastFm, genres: dict[str, str], songs: list[tuple]) -> tuple[dict, set]:
    """Genres and their source for each (artist, title, album) that has any, and the songs whose lookup failed."""
    found: dict[tuple, tuple[list[str], str]] = {}
    failed: set[tuple] = set()
    start = time.monotonic()

    def look_up(song: tuple, song_queries: list[tuple[str, str, dict]]) -> None:
        for level, method, params in song_queries:
            tags = lastfm.top_tags(method, params)
            if tags is None:
                failed.add(song)
                return
            if song_genres := genres_in(tags, genres):
                found[song] = (song_genres, level)
                return

    def progress(done: int, total: int, what: str) -> None:
        if done % 250 == 0 or done == total:
            minutes = (time.monotonic() - start) / 60
            print(f"  {what}: {done:,} of {total:,} songs, {lastfm.requests:,} Last.fm requests, {minutes:.0f} min", flush=True)

    for done, song in enumerate(songs, 1):
        look_up(song, queries(*song))
        progress(done, len(songs), "track and album tags")

    # Last.fm has one page per artist name, so artists with the same name share
    # its tags (a singer can get a namesake DJ's techno). The artist's own tracks
    # are a better guide where some have genres.
    by_artist: dict[str, list[list[str]]] = defaultdict(list)
    for (artist, _, _), (song_genres, _) in found.items():
        by_artist[artist].append(song_genres)
    rest = [song for song in songs if song not in found and song not in failed]
    for done, song in enumerate(rest, 1):
        artist = song[0]
        if artist in by_artist:
            found[song] = (common_genres(by_artist[artist]), "artist_tracks")
        else:
            look_up(song, [("artist", "artist.getTopTags", {"artist": artist})])
        progress(done, len(rest), "artist tags")
    return found, failed


def write_csv(path: Path, tracks: list[tuple], found: dict, failed: set) -> Counter:
    outcomes: Counter = Counter()
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # the BOM lets Excel see UTF-8
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for track_id, title, artist, album in tracks:
            song = (artist, title, album)
            song_genres, level = found.get(song, ([], None))
            outcomes[level or ("no_name" if not (title and artist) else "failed" if song in failed else "not_tagged")] += 1
            writer.writerow([
                track_id,
                title or MISSING,
                artist or MISSING,
                album or MISSING,
                song_genres[0] if song_genres else MISSING,
                "|".join(song_genres) or MISSING,
                level or MISSING,
            ])
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="DuckDB file with silver.tracks")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV file to write")
    parser.add_argument("--limit", type=int, help="only the N most played tracks, to try things out")
    args = parser.parse_args()

    genres = load_genres(GENRE_LIST)
    api_key = read_api_key()
    tracks = read_tracks(args.db)[: args.limit]
    # Spotify often has one song under several track ids; each title, artist and album is looked up once.
    songs = list(dict.fromkeys((artist, title, album) for _, title, artist, album in tracks if title and artist))
    named = sum(1 for _, title, artist, _ in tracks if title and artist)
    print(
        f"{len(tracks):,} tracks: {named:,} with a title and artist ({len(songs):,} different songs), "
        f"{len(tracks) - named:,} without"
    )

    lastfm = LastFm(api_key, CACHE_FILE)
    try:
        found, failed = find_genres(lastfm, genres, songs)
    except LastFmError as error:
        sys.exit(f"Stopped: {error}. The answers so far are saved in {CACHE_FILE.name}; run again to continue.")
    except KeyboardInterrupt:
        sys.exit(f"Interrupted. The answers so far are saved in {CACHE_FILE.name}; run again to continue.")
    finally:
        lastfm.close()

    try:
        outcomes = write_csv(args.out, tracks, found, failed)
    except PermissionError:
        sys.exit(f"Could not write {args.out}: close it if another program has it open, then run again")
    with_genre = sum(n for outcome, n in outcomes.items() if not OUTCOMES[outcome].startswith("NA"))
    print(
        f"Genres for {with_genre:,} of the {named:,} tracks with a title and artist "
        f"({lastfm.requests:,} Last.fm requests this run)"
    )
    for outcome, label in OUTCOMES.items():
        if outcomes[outcome]:
            print(f"  {outcomes[outcome]:>7,}  {label}")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
