"""Your music recommender as a local website: songs to like, rate neutral or dislike, lists that learn from it,
your data explored, and how it all works.

    streamlit run app.py

Run it from the project folder, then open http://127.0.0.1:8501 (http://localhost:8501 works too, but Spotify
sends you back to 127.0.0.1 after you connect your account). Only this computer can open it
(.streamlit/config.toml), and it sends nothing anywhere but the Spotify and Last.fm requests the pages explain.

This file is the frame: the look (font, colours and your wallpaper), the logo, the navigation at the top, and the
end of a Spotify login. The pages are in app_pages/:

    Recommendations  recommendations.py  song cards with Spotify's player; choices are sent, and the next list
                                         built, once the list is all chosen or on Next songs
    Your ratings     ratings.py          every rating, with Undo, and the backup as a file
    Your data        your_data.py        where the data comes from, how it is cleaned and stored, and charts
    How it works     how_it_works.py     the logic, how the machine learns, how well, and how it could do better
    Settings         settings.py         wallpaper, your Spotify account, reloading after a rebuild

It sits in the project folder because streamlit run puts the script's folder on sys.path: inside recommender/,
profile.py would shadow the standard library's profile module.
"""

import mimetypes
from pathlib import Path

import requests
import streamlit as st

from app_pages import common, look
from recommender import spotify

# Static files are served with the type Python's mimetypes guesses; Windows doesn't know web fonts.
mimetypes.add_type("font/woff2", ".woff2")
LOGO = Path(__file__).resolve().parent / "static" / "logo.svg"

st.set_page_config(page_title="Music Preference", page_icon=str(LOGO), layout="wide")
state = common.state

# Spotify sends you back here after Connect Spotify, with a code to trade for your login (or an error if you said no).
params = st.query_params
if "code" in params and "state" in params:
    try:
        name = spotify.finish_login(params["code"], params["state"])
        common.message(f"Connected to Spotify as {name}")
    except (spotify.NotConnected, spotify.NoSpotifyApp, RuntimeError, requests.RequestException) as error:
        common.message(f"Spotify login didn't work: {error}")
    common.account_info.clear()
    st.query_params.clear()
elif "error" in params:
    common.message(f"Spotify login stopped: {params['error']}")
    st.query_params.clear()

look.apply_look(common.site_settings())
st.logo(str(LOGO), size="large")
page = st.navigation(
    [
        st.Page("app_pages/recommendations.py", title="Recommendations", icon=":material/library_music:", default=True),
        st.Page("app_pages/ratings.py", title="Your ratings", icon=":material/thumbs_up_down:", url_path="ratings"),
        st.Page("app_pages/your_data.py", title="Your data", icon=":material/insights:", url_path="data"),
        st.Page("app_pages/how_it_works.py", title="How it works", icon=":material/school:", url_path="how-it-works"),
        st.Page("app_pages/settings.py", title="Settings", icon=":material/tune:", url_path="settings"),
    ],
    position="top",
)
if state.get("message"):
    st.toast(state.pop("message"))
page.run()
