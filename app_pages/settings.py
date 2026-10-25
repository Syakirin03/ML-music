"""The Settings page: the wallpaper, your Spotify account, and reloading the data after a rebuild.

The wallpaper and what a like also does are kept in data/user/website.json, and your own picture in
data/user/wallpaper.jpg, so they are the same every time you open the website.
"""

import html
import io

import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from app_pages import common, look
from app_pages.common import WALLPAPER_FILE, state
from recommender import spotify

# Your picture is shrunk to at most this size and saved as JPEG: big enough for a screen, small enough to load fast.
WALLPAPER_SIZE = (2560, 1600)
WALLPAPER_QUALITY = 85
ON_LIKE = {"rate": "Only rate it", "liked": "Also save it to my Liked Songs", "playlist": "Also add it to a playlist"}


def on_wallpaper() -> None:
    common.save_site_settings(wallpaper=state.wallpaper_choice or "plain")


def on_dim() -> None:
    common.save_site_settings(dim=state.wallpaper_dim)


def on_picture() -> None:
    upload = state.get("wallpaper_upload")
    if upload is None:
        return
    try:
        picture = ImageOps.exif_transpose(Image.open(io.BytesIO(upload.getvalue()))).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        common.message(f"That file isn't a picture this website can read: {error}")
        return
    picture.thumbnail(WALLPAPER_SIZE)
    WALLPAPER_FILE.parent.mkdir(parents=True, exist_ok=True)
    picture.save(WALLPAPER_FILE, "JPEG", quality=WALLPAPER_QUALITY, optimize=True)
    common.save_site_settings(wallpaper="picture")
    state.wallpaper_choice = "picture"
    common.message("Your picture is the wallpaper now")


def on_like_choice() -> None:
    common.save_site_settings(on_like=state.on_like_choice)


def on_playlist(names: dict[str, str]) -> None:
    chosen = state.get("like_playlist")
    common.save_site_settings(playlist_id=chosen, playlist_name=names.get(chosen))


def on_create() -> None:
    name = (state.get("new_playlist") or "").strip()
    if not name:
        common.message("Give the playlist a name first")
        return
    try:
        you = common.account()
        made = you.create_playlist(name, "Songs I liked on my own music recommender")
    except (AttributeError, spotify.NotConnected, RuntimeError, requests.RequestException) as error:
        common.message(f"Spotify didn't make the playlist: {error}")
        return
    common.save_site_settings(playlist_id=made["id"], playlist_name=made["name"], on_like="playlist")
    common.account_info.clear()
    state.pop("like_playlist", None)
    state.pop("on_like_choice", None)
    common.message(f"Made the private playlist {made['name']}: liked songs go there now")


def on_disconnect() -> None:
    spotify.disconnect()
    common.account_info.clear()
    common.message("Disconnected from Spotify. To take the access back on Spotify too, see spotify.com/account/apps")


def on_reload() -> None:
    common.reload_all()
    common.message("Reloaded your history, the pool of new songs and the models")


settings = common.site_settings()
look.hero("Settings", "Make it yours", "The wallpaper, your Spotify account, and the data behind the recommendations.")

st.header("Wallpaper")
kind = look.theme_kind() or "light"
names = {key: name for key, (name, _) in look.WALLPAPERS.items()}
swatches = st.columns(len(names), gap="small")
for column, key in zip(swatches, names):
    background = look.wallpaper_css(key, kind) or ("#000000" if kind == "dark" else "#ffffff")
    column.html(f'<div class="mp-swatch" style="background: {html.escape(background.replace(" fixed", ""), quote=True)};"></div>')
    column.caption(names[key] if key != "picture" or WALLPAPER_FILE.exists() else f"{names[key]}: upload one below")
state.setdefault("wallpaper_choice", settings["wallpaper"])
st.pills("Choose one", list(names), format_func=names.get, key="wallpaper_choice", on_change=on_wallpaper, required=True)
st.file_uploader(
    "Or use a picture of your own", type=["jpg", "jpeg", "png", "webp"], key="wallpaper_upload", on_change=on_picture,
    help="It is kept in data/user/wallpaper.jpg, shrunk to fit a screen",
)
state.setdefault("wallpaper_dim", float(settings["dim"]))
st.slider("Dim the wallpaper", 0.0, 0.8, step=0.05, key="wallpaper_dim", on_change=on_dim,
          help="More dimming keeps text easy to read over a busy picture")

st.header("Spotify")
connected = common.connection()
try:
    spotify.credentials()
    has_app = True
except spotify.NoSpotifyApp as error:
    has_app = False
    st.info(f"{error}. The website needs your Spotify developer app for its players and for your account.")
if has_app and connected is None:
    st.markdown(
        "Connect your Spotify account to **save songs to your Liked Songs** and **add them to your playlists** from "
        "the cards, or every time you like one. The website asks only for that: it can't read your listening, email "
        "or payment details, and the login stays on this computer (secrets/spotify_account.json)."
    )
    st.link_button("Connect Spotify", spotify.login_url(), icon=":material/link:", type="primary")
    st.caption(
        f"Spotify sends you back to {spotify.REDIRECT_URI}, which must be among your app's Redirect URIs on the "
        "Spotify dashboard (the setup in recommender/spotify.py enters it). If Spotify says you aren't registered, "
        "add your account under User Management in the app's settings."
    )
elif connected is not None:
    if connected["error"]:
        st.warning(f"The website is connected, but Spotify refused: {connected['error']}")
    else:
        count = len(connected["playlists"])
        st.success(f"Connected as {connected['name']}, with {count} playlist{'' if count == 1 else 's'} you can add to.")
    with st.container(horizontal=True, gap="small"):
        st.button("Disconnect", icon=":material/link_off:", on_click=on_disconnect)
        st.button("Read my playlists again", icon=":material/refresh:", on_click=common.account_info.clear)
    if not connected["error"]:
        state.setdefault("on_like_choice", settings["on_like"])
        st.radio("When I like a song", list(ON_LIKE), format_func=ON_LIKE.get, key="on_like_choice", on_change=on_like_choice)
        if state.on_like_choice == "playlist":
            playlists = {playlist["id"]: playlist["name"] for playlist in connected["playlists"]}
            if settings["playlist_id"] in playlists:
                state.setdefault("like_playlist", settings["playlist_id"])
            st.selectbox("The playlist", list(playlists), format_func=playlists.get, key="like_playlist", index=None,
                         placeholder="Choose one of your playlists", on_change=on_playlist, args=(playlists,))
            with st.container(horizontal=True, gap="small", vertical_alignment="bottom"):
                st.text_input("Or make a new private playlist", key="new_playlist", placeholder="My recommender finds")
                st.button("Make it", on_click=on_create)

st.header("Your data")
st.markdown(
    "After rebuilding the database or the pool of new songs (the scripts in scripts/), reload them here; the "
    "website otherwise reads them once."
)
st.button("Reload history and pool", icon=":material/sync:", on_click=on_reload)
st.caption(
    "Your ratings are in data/user/recommender.duckdb, with a copy in data/user/ratings.csv after every rating. "
    "Nothing can rebuild them, so back that folder up now and then."
)
look.footer()
