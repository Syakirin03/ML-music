"""What the website's pages share: your history (loaded once), your ratings, the choices you make on the
Settings page, settings kept while you switch pages, and your Spotify account."""

import contextlib
import io
import json

import pandas as pd
import requests
import streamlit as st

from recommender import spotify
from recommender.data import NEUTRAL, SPOTIFY_DB, USER_DIR, candidate_pool, current_ratings, data_overview, load_library, load_ratings
from recommender.feedback import feedback_model, model_weights
from recommender.ranker import rediscover_feedback

# Your choices on the Settings page (wallpaper, what a like also does on Spotify), and your own wallpaper picture.
# They sit with your ratings, so that MUSIC_RECOMMENDER_USER_DIR moves them to a test copy too.
SETTINGS_FILE = USER_DIR / "website.json"
WALLPAPER_FILE = USER_DIR / "wallpaper.jpg"
DEFAULT_SETTINGS = {"wallpaper": "aurora", "dim": 0.35, "on_like": "rate", "playlist_id": None, "playlist_name": None}
# What a rating is called on the website, by its value in the ratings table.
RATING_WORDS = {1: "Liked", -1: "Disliked", NEUTRAL: "Neutral"}
MODE_WORDS = {"discover": "new to you", "rediscover": "to rediscover", "like": "like one you named"}

state = st.session_state


@st.cache_resource(show_spinner="Loading your listening history...")
def history():
    """Your history, loaded once: songs, plays and the pool don't change while the website runs."""
    return load_library()


def library():
    """history() with your ratings as they are now."""
    return history()._replace(ratings=load_ratings())


def ratings_stamp(ratings: pd.DataFrame) -> tuple[int, str]:
    """Changes whenever a rating is added: the key of the caches below."""
    return len(ratings), str(ratings.rated_at.max()) if len(ratings) else ""


@st.cache_resource
def pool_genres() -> list[str]:
    """The genres of the songs new to you, the most common first, for "More of these genres"."""
    with contextlib.redirect_stdout(io.StringIO()):  # candidate_pool's note about an old pool shows with the list
        songs = candidate_pool(history(), "now")
    return songs.genres.dropna().explode().value_counts().index.tolist()


@st.cache_data(show_spinner=False, max_entries=8)
def learned_models(stamp: tuple[int, str]) -> dict:
    """Both models learned from your ratings: whether each is in use, and what it learned (worked out again only
    when your ratings change)."""
    current = library()
    models = {"discover": feedback_model(current.ratings, current.candidates), "rediscover": rediscover_feedback(current)}
    return {
        mode: {"message": model.message, "in_use": model.model is not None, "ratings": model.ratings,
               "weights": model_weights(model) if model.model is not None else pd.Series(dtype=float)}
        for mode, model in models.items()
    }


@st.cache_data(show_spinner="Reading the database...")
def overview(modified: float):
    """data_overview(), read again only when the database file changes (modified is its time)."""
    return data_overview()


def database_overview():
    return overview(SPOTIFY_DB.stat().st_mtime)


def reload_all() -> None:
    """Forget everything loaded, after a rebuild of the database or the pool of new songs."""
    history.clear()
    pool_genres.clear()
    learned_models.clear()
    overview.clear()
    account_info.clear()
    state.page = None


def current() -> pd.DataFrame:
    """Your current ratings (the newest of each song), oldest first."""
    return current_ratings(load_ratings())


# Your choices on the Settings page, kept on disk.

def site_settings() -> dict:
    try:
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        saved = {}
    return {**DEFAULT_SETTINGS, **{key: value for key, value in saved.items() if key in DEFAULT_SETTINGS}}


def save_site_settings(**changes) -> None:
    settings = {**site_settings(), **changes}
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = SETTINGS_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(SETTINGS_FILE)


# Settings of the sidebar, kept while you visit other pages (Streamlit forgets a widget's value when its page
# isn't shown): each widget has the key kept(name, default) returns, and remember(name) after it.

def kept(name: str, default):
    key = f"keep_{name}"
    prefs = state.setdefault("prefs", {})
    if key not in state:
        state[key] = prefs.get(name, default)
    return key


def remember(name: str):
    state.setdefault("prefs", {})[name] = state[f"keep_{name}"]
    return state[f"keep_{name}"]


# Your Spotify account.

def account() -> spotify.Account | None:
    """Your account, or None if the website isn't connected (or there is no Spotify app)."""
    try:
        return spotify.Account()
    except (spotify.NotConnected, spotify.NoSpotifyApp, ValueError, KeyError):
        return None


@st.cache_data(ttl=600, show_spinner=False)
def account_info(stamp: float) -> dict:
    """Your Spotify name and the playlists you can add to, or the error that stopped Spotify from saying; read
    again after 10 minutes or a new login (stamp is the login file's time)."""
    try:
        you = spotify.Account()
        return {"name": you.name, "playlists": you.playlists(), "error": None}
    except (spotify.NotConnected, spotify.NoSpotifyApp, RuntimeError, requests.RequestException, KeyError) as error:
        return {"name": None, "playlists": [], "error": str(error)}


def connection() -> dict | None:
    """account_info() if the website is connected to your Spotify account, else None."""
    if not spotify.ACCOUNT_FILE.exists():
        return None
    return account_info(spotify.ACCOUNT_FILE.stat().st_mtime)


def message(text: str) -> None:
    """A message shown once, as a toast, on the next run of the page."""
    state["message"] = text
