"""Find songs on Spotify, so that the website can show Spotify's player for songs new to you.

    python -m recommender.spotify ARTIST TITLE      check the setup by looking up one song

Songs of your history have Spotify track ids already, but songs from the
Last.fm pool (candidates.csv) only have names. find_track_ids() searches
Spotify for each by title and artist, with a Spotify developer app of your
own, and keeps every answer, "not found" too, in CACHE, so that each song is
looked up once.

Setting up the app, once (since February 2026, Spotify only runs a developer
app while its owner has Premium):

1. Log in at https://developer.spotify.com/dashboard and choose Create app.
2. Give it any name and description, enter http://127.0.0.1:8501 as Redirect
   URI (the form asks for one; it isn't used), tick Web API, and save.
3. In the app's Settings, copy the Client ID, and the Client secret under
   View client secret.
4. Put them in secrets/spotify_client.txt, the ID on the first line and the
   secret on the second, or in the environment variables SPOTIFY_CLIENT_ID
   and SPOTIFY_CLIENT_SECRET.

What is sent: the title and artist of each song the website shows you, with
the app's credentials (the client credentials flow, which logs in as the app,
not as you, so nothing of your account is read). Spotify's terms forbid using
its content to train machine-learning models, so the ids found are only used
to show the player, never as features.

A song counts as found only when a result has the same plain title (data.py
plain_titles) and an artist with the same letters (data.py letters), so that a
cover or a remix by someone else isn't played instead.

Your own account, optional. Logging in with Connect Spotify on the website's
Settings page lets it save songs to your Liked Songs and add them to your
playlists (Account). Spotify asks you to agree, then sends you back to
REDIRECT_URI, which must be one of the app's Redirect URIs: it is the one step 2
enters. The website gets tokens for the SCOPES below, nothing more: it can't
read your listening, email or payment details. They are kept on this computer
only, in ACCOUNT_FILE; Disconnect on the Settings page deletes it, and
https://www.spotify.com/account/apps/ takes the access back on Spotify's side.
A development-mode app only lets in Spotify users on its list: if Spotify says
you aren't registered, add your account under User Management in the app's
settings on the dashboard.

    python -m recommender.spotify --account        whether the website is connected, and as whom
"""

import json
import os
import re
import secrets
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlencode

import pandas as pd
import requests

if not __package__:
    # Started as a file (VS Code's Run button, or python recommender/spotify.py) instead of with
    # python -m: Python then looks for imports in recommender/, so add the project folder first.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recommender.config import CONFIG
from recommender.data import ROOT, letters, plain_titles

CREDENTIALS = ROOT / "secrets" / "spotify_client.txt"
CACHE = ROOT / "data" / "interim" / "spotify_search_cache.jsonl"
TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"
# Search results for this country ([home] market in config.toml): songs you can play there. Empty: no country.
MARKET = CONFIG["home"]["market"]
# Results per search: the most a development-mode app gets since February 2026.
SEARCH_LIMIT = 10
# Seconds to wait for Spotify, and the longest wait it may ask for when it is busy (429) before giving up.
TIMEOUT = 20
MAX_RETRY_AFTER = 30

# Your account (see the docstring). REDIRECT_URI must match one of the app's Redirect URIs exactly; Spotify
# allows http only for 127.0.0.1, never for localhost.
REDIRECT_URI = os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501")
AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
API_URL = "https://api.spotify.com/v1"
# What the website may do with your account: check and add to your Liked Songs, see your playlists and add to them.
SCOPES = (
    "user-library-read user-library-modify playlist-read-private playlist-read-collaborative "
    "playlist-modify-private playlist-modify-public"
)
# Your login: the tokens Spotify gave the website, on this computer only. Deleting the file logs the website out.
ACCOUNT_FILE = ROOT / "secrets" / "spotify_account.json"
# A login in progress: the random state sent to Spotify, checked when it sends you back, so that a link someone
# else made can't log the website in to their account. It is good for LOGIN_MINUTES.
LOGIN_FILE = ROOT / "secrets" / "spotify_login.json"
LOGIN_MINUTES = 15
# Spotify takes at most this many URIs per request to your library (PUT /me/library, GET /me/library/contains).
LIBRARY_BATCH = 40
# A Spotify track id: 22 letters and digits.
TRACK_ID = re.compile(r"[A-Za-z0-9]{22}")


class NoSpotifyApp(Exception):
    pass


class NotConnected(Exception):
    """The website isn't logged in to your Spotify account, or Spotify no longer accepts the login."""


def credentials() -> tuple[str, str]:
    """The app's Client ID and secret: from SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET, else CREDENTIALS."""
    client_id, secret = os.environ.get("SPOTIFY_CLIENT_ID"), os.environ.get("SPOTIFY_CLIENT_SECRET")
    if client_id and secret:
        return client_id.strip(), secret.strip()
    if CREDENTIALS.exists():
        lines = [line.strip() for line in CREDENTIALS.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) >= 2:
            return lines[0], lines[1]
        raise NoSpotifyApp(f"{CREDENTIALS} should have the Client ID on the first line and the secret on the second")
    raise NoSpotifyApp(f"No Spotify app set up yet: see recommender/spotify.py, then put its Client ID and secret in {CREDENTIALS}")


class Spotify:
    """Searches with an app token, fetched when first needed and again when it expires (after about an hour)."""

    def __init__(self) -> None:
        self.client_id, self.secret = credentials()
        self.token, self.expires = None, 0.0

    def headers(self) -> dict[str, str]:
        if self.token is None or time.monotonic() > self.expires - 60:
            reply = requests.post(
                TOKEN_URL, data={"grant_type": "client_credentials"}, auth=(self.client_id, self.secret), timeout=TIMEOUT
            )
            if reply.status_code in (400, 401):
                raise NoSpotifyApp(f"Spotify didn't accept the app's Client ID and secret ({reply.status_code}: {reply.text[:200]})")
            reply.raise_for_status()
            answer = reply.json()
            self.token, self.expires = answer["access_token"], time.monotonic() + answer.get("expires_in", 3600)
        return {"Authorization": f"Bearer {self.token}"}

    def search(self, query: str) -> list[dict]:
        """The tracks Spotify finds for a query, best first."""
        params = {"q": query, "type": "track", "limit": SEARCH_LIMIT, **({"market": MARKET} if MARKET else {})}
        for _ in range(3):
            reply = requests.get(SEARCH_URL, params=params, headers=self.headers(), timeout=TIMEOUT)
            if reply.status_code == 429:
                wait = float(reply.headers.get("Retry-After", 5))
                if wait > MAX_RETRY_AFTER:
                    raise RuntimeError(f"Spotify asks to wait {wait:.0f} s before searching again")
                time.sleep(wait)
                continue
            if reply.status_code == 403:
                raise NoSpotifyApp(f"Spotify refused the search (403: {reply.text[:200]}); the app needs its owner on Premium")
            reply.raise_for_status()
            return reply.json().get("tracks", {}).get("items", [])
        raise RuntimeError("Spotify stayed busy (429) after three tries")


def latin(text: str) -> bool:
    """Whether text has Latin letters (a-z), as an English or romanised title does."""
    return any("a" <= c <= "z" for c in text.casefold())


def same_song(item: dict, artist: str, title: str, field_search: bool = False) -> bool:
    """Whether a Spotify track is the song: one of its artists with the same letters, and the same plain title.

    In a field search (track and artist named), a title in another script also counts: Spotify may write an
    English title in Japanese script, say, and the track field itself made the match.
    """
    found, wanted = plain_titles(pd.Series([item.get("name", "")])).iloc[0], plain_titles(pd.Series([title])).iloc[0]
    same_title = found == wanted or (field_search and latin(found) != latin(wanted))
    return same_title and any(artist_letters(a.get("name", "")) == artist_letters(artist) for a in item.get("artists", []))


def artist_letters(name: str) -> str:
    """letters() of an artist's name, with $ read as s, as one site may stylise a name the other spells out."""
    return letters(name.replace("$", "s"))


def queries(artist: str, title: str) -> list[str]:
    """Field search first, then plain words: the field search misses some titles with symbols."""
    plain = plain_titles(pd.Series([title])).iloc[0].replace('"', "")
    return [f'track:"{plain}" artist:"{artist.replace(chr(34), "")}"', f"{artist} {plain}"]


def load_cache() -> dict[str, str | None]:
    """song_key -> Spotify track id, or None for a song looked up and not found."""
    found = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                answer = json.loads(line)
                found[answer["song_key"]] = answer["track_id"]
    return found


def find_track_ids(songs: pd.DataFrame, client: Spotify | None = None) -> dict[str, str | None]:
    """For rows with song_key, artist and title (a table of recommend.py): song_key -> Spotify track id, or None.

    Only songs not in CACHE are searched; raises NoSpotifyApp if one must be and there is no app.
    """
    known = load_cache()
    missing = songs[~songs.song_key.isin(known.keys())]
    if len(missing):
        client = client or Spotify()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a", encoding="utf-8") as cache:
            for row in missing.drop_duplicates("song_key").itertuples():
                match = None
                for number, query in enumerate(queries(row.artist, row.title)):
                    items = client.search(query)
                    match = next((item for item in items if same_song(item, row.artist, row.title)), None)
                    if match is None and number == 0:  # the field search: a title in another script counts too
                        match = next((item for item in items if same_song(item, row.artist, row.title, field_search=True)), None)
                    if match:
                        break
                known[row.song_key] = match["id"] if match else None
                cache.write(json.dumps({
                    "song_key": row.song_key, "artist": row.artist, "title": row.title,
                    "track_id": known[row.song_key], "found": f"{match['name']} - {match['artists'][0]['name']}" if match else None,
                    "searched_at": pd.Timestamp.now(tz="UTC").isoformat(),
                }, ensure_ascii=False) + "\n")
    return {key: known[key] for key in songs.song_key}


def write_private(path: Path, data: dict) -> None:
    """Write data as JSON that only you may read, via a temporary file, so that a crash never leaves half a file.

    On macOS and Linux the file gets permissions 600. Windows ignores those, and the file gets its folder's
    permissions: inside your user folder, only you (and administrators) may read it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as file:
        json.dump(data, file)
    temporary.replace(path)


def login_url() -> str:
    """The Spotify page that asks you to let the website use your account (SCOPES); it sends you back to REDIRECT_URI.

    The same link is given until LOGIN_MINUTES have passed, so that a page shown again keeps a link that works.
    """
    client_id, _ = credentials()
    try:
        pending = json.loads(LOGIN_FILE.read_text(encoding="utf-8"))
        fresh = time.time() - pending["created"] < LOGIN_MINUTES * 60 - 60
    except (FileNotFoundError, ValueError, KeyError):
        fresh = False
    if not fresh:
        pending = {"state": secrets.token_urlsafe(24), "created": time.time()}
        write_private(LOGIN_FILE, pending)
    return AUTHORIZE_URL + "?" + urlencode({
        "client_id": client_id, "response_type": "code", "redirect_uri": REDIRECT_URI, "scope": SCOPES,
        "state": pending["state"],
    })


def finish_login(code: str, state: str) -> str:
    """Trade the code Spotify sent you back with for tokens and keep them in ACCOUNT_FILE; returns your Spotify name."""
    try:
        pending = json.loads(LOGIN_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        raise NotConnected("This login wasn't started on this computer, or it is done already: press Connect Spotify again")
    if not secrets.compare_digest(str(pending.get("state", "")), state):
        raise NotConnected("The login didn't come back with the code this computer sent, so it was refused: try again")
    if time.time() - pending.get("created", 0) > LOGIN_MINUTES * 60:
        raise NotConnected(f"The login took longer than {LOGIN_MINUTES} minutes: press Connect Spotify again")
    client_id, secret = credentials()
    reply = requests.post(
        TOKEN_URL, data={"grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT_URI},
        auth=(client_id, secret), timeout=TIMEOUT,
    )
    if reply.status_code != 200:
        raise NotConnected(f"Spotify didn't accept the login ({reply.status_code}: {reply.text[:200]})")
    answer = reply.json()
    write_private(ACCOUNT_FILE, {
        "access_token": answer["access_token"], "refresh_token": answer["refresh_token"],
        "expires_at": time.time() + answer.get("expires_in", 3600), "scope": answer.get("scope", ""),
    })
    LOGIN_FILE.unlink(missing_ok=True)
    return Account().name


def disconnect() -> None:
    """Log the website out of your account: delete ACCOUNT_FILE (and any login in progress)."""
    ACCOUNT_FILE.unlink(missing_ok=True)
    LOGIN_FILE.unlink(missing_ok=True)


def track_uris(track_ids) -> list[str]:
    """spotify:track: URIs of the ids that are Spotify track ids, in order."""
    return [f"spotify:track:{track}" for track in track_ids if isinstance(track, str) and TRACK_ID.fullmatch(track)]


class Account:
    """Your Spotify account, with the tokens of ACCOUNT_FILE, refreshed when they expire (after about an hour)."""

    def __init__(self, path: Path | None = None) -> None:
        path = path or ACCOUNT_FILE
        if not path.exists():
            raise NotConnected("The website isn't connected to your Spotify account yet")
        self.path = path
        self.tokens = json.loads(path.read_text(encoding="utf-8"))
        self.client_id, self.secret = credentials()
        self._me: dict | None = None

    def refresh(self) -> None:
        reply = requests.post(
            TOKEN_URL, data={"grant_type": "refresh_token", "refresh_token": self.tokens["refresh_token"]},
            auth=(self.client_id, self.secret), timeout=TIMEOUT,
        )
        if reply.status_code in (400, 401):
            raise NotConnected("Spotify no longer accepts the website's login: connect again on the Settings page")
        reply.raise_for_status()
        answer = reply.json()
        self.tokens.update(
            access_token=answer["access_token"], expires_at=time.time() + answer.get("expires_in", 3600),
            refresh_token=answer.get("refresh_token", self.tokens["refresh_token"]),
        )
        write_private(self.path, self.tokens)

    def call(self, method: str, path: str, **options):
        """Spotify's answer to one request of the Web API (path after /v1), or None when it has no body."""
        refreshed = False
        for _ in range(3):
            if time.time() > self.tokens.get("expires_at", 0) - 60:
                self.refresh()
            headers = {"Authorization": f"Bearer {self.tokens['access_token']}"}
            reply = requests.request(method, API_URL + path, headers=headers, timeout=TIMEOUT, **options)
            if reply.status_code == 429:
                wait = float(reply.headers.get("Retry-After", 5))
                if wait > MAX_RETRY_AFTER:
                    raise RuntimeError(f"Spotify asks to wait {wait:.0f} s")
                time.sleep(wait)
                continue
            if reply.status_code == 401 and not refreshed:  # the token ran out early: refresh once and try again
                self.refresh()
                refreshed = True
                continue
            if reply.status_code == 401:
                raise NotConnected("Spotify no longer accepts the website's login: connect again on the Settings page")
            if reply.status_code == 403:
                raise RuntimeError(
                    f"Spotify refused ({reply.text[:200]}). If it says you aren't registered, add your account under "
                    "User Management in your app's settings on the Spotify dashboard"
                )
            reply.raise_for_status()
            return reply.json() if reply.content else None
        raise RuntimeError("Spotify stayed busy (429) after three tries")

    @property
    def name(self) -> str:
        """Your Spotify display name (your user id if you have none)."""
        if self._me is None:
            self._me = self.call("GET", "/me")
        return self._me.get("display_name") or self._me["id"]

    def playlists(self, most: int = 200) -> list[dict]:
        """Your playlists you can add songs to (yours and collaborative ones), as {id, name, songs}, in Spotify's order."""
        if self._me is None:
            self._me = self.call("GET", "/me")
        found = []
        for offset in range(0, most, 50):
            page = self.call("GET", "/me/playlists", params={"limit": 50, "offset": offset})
            for playlist in page.get("items", []):
                if playlist and (playlist["owner"]["id"] == self._me["id"] or playlist.get("collaborative")):
                    count = (playlist.get("items") or playlist.get("tracks") or {}).get("total", 0)
                    found.append({"id": playlist["id"], "name": playlist["name"], "songs": count})
            if not page.get("next"):
                break
        return found

    def library_query(self, uris: list[str]) -> str:
        # Spotify reads the commas between URIs as separators, so only the URIs themselves are encoded.
        return "?uris=" + ",".join(quote(uri, safe="") for uri in uris)

    def saved(self, track_ids) -> set[str]:
        """Which of the track ids are in your Liked Songs."""
        uris = track_uris(track_ids)
        found = set()
        for start in range(0, len(uris), LIBRARY_BATCH):
            batch = uris[start:start + LIBRARY_BATCH]
            answers = self.call("GET", "/me/library/contains" + self.library_query(batch)) or []
            found |= {uri.rsplit(":", 1)[1] for uri, saved in zip(batch, answers) if saved}
        return found

    def save(self, track_ids) -> int:
        """Add the songs to your Liked Songs; returns how many were sent."""
        uris = track_uris(track_ids)
        for start in range(0, len(uris), LIBRARY_BATCH):
            self.call("PUT", "/me/library" + self.library_query(uris[start:start + LIBRARY_BATCH]))
        return len(uris)

    def add_to_playlist(self, playlist_id: str, track_ids) -> int:
        """Add the songs at the end of one of your playlists; returns how many were sent."""
        uris = track_uris(track_ids)
        if uris:
            self.call("POST", f"/playlists/{quote(playlist_id, safe='')}/items", json={"uris": uris[:100]})
        return len(uris[:100])

    def create_playlist(self, name: str, description: str = "") -> dict:
        """A new private playlist of yours, as {id, name, songs}."""
        made = self.call("POST", "/me/playlists", json={"name": name, "public": False, "description": description})
        return {"id": made["id"], "name": made["name"], "songs": 0}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:] == ["--account"]:
        try:
            print(f"The website is connected to the Spotify account of {Account().name}")
        except (NotConnected, NoSpotifyApp, RuntimeError, requests.RequestException) as error:
            sys.exit(str(error))
        return
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    artist, title = sys.argv[1], sys.argv[2]
    try:
        client = Spotify()
        for query in queries(artist, title):
            items = client.search(query)
            match = next((item for item in items if same_song(item, artist, title)), None)
            print(f'Searched {query!r}: {len(items)} results{"" if match else ", none the same song"}')
            if match:
                print(f"Found: {match['name']} - {', '.join(a['name'] for a in match['artists'])}")
                print(f"https://open.spotify.com/track/{match['id']}")
                return
    except (NoSpotifyApp, RuntimeError, requests.RequestException) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
