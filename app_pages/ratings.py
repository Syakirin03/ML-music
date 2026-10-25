"""The Your ratings page: every song you rated, newest first, with Undo, and your ratings as a file."""

import pandas as pd
import streamlit as st

from app_pages import common, look
from app_pages.common import MODE_WORDS, RATING_WORDS
from recommender.data import NEUTRAL, RATINGS_CSV
from recommender.feedback import rate_rows

# Ratings listed at once; the filters narrow them down.
SHOWN = 100
KINDS = {"All": None, "Liked": 1, "Neutral": NEUTRAL, "Disliked": -1}
LISTS = {"All lists": None, "New to you": "discover", "To rediscover": "rediscover", "Like one you named": "like"}


def md(text) -> str:
    return str(text).translate(str.maketrans({c: "\\" + c for c in "\\`*_{}[]()#+-.!|<>~$"}))


def on_undo(row: pd.Series) -> None:
    try:
        rate_rows(row.to_frame().T, "clear")
    except (FileNotFoundError, RuntimeError, LookupError, ValueError) as error:
        common.message(f"Not taken back: {error}")
        return
    common.message(f"Took back your rating of {row.track_name}: it can come back in your lists")


rated = common.current()
look.hero(
    "Your ratings", "What you told the recommender",
    "Every song you liked, rated neutral or disliked, newest first. A like or a dislike counts in your taste profile "
    "and teaches the two models learned from your ratings; a neutral rating only keeps the song off your lists. "
    "Undo takes a rating back: the song can come back in your lists.",
    [(f"{(rated.rating == value).sum():,}", word.lower()) for value, word in RATING_WORDS.items()]
    + [(f"{len(rated):,}", "songs in all")],
)

pending = len(common.state.get("choices", {}))
if pending:
    st.info(f"{pending} choice{'s' if pending != 1 else ''} on the Recommendations page {'are' if pending != 1 else 'is'} "
            "not sent yet: choices are sent once you've chosen the whole list or press Next songs.")
if rated.empty:
    st.info("Nothing rated yet: like, rate neutral or dislike songs on the Recommendations page.")
else:
    with st.container(horizontal=True, gap="medium", vertical_alignment="bottom"):
        kind = st.segmented_control("Rating", list(KINDS), default="All", key="ratings_kind")
        where = st.segmented_control("From", list(LISTS), default="All lists", key="ratings_list")
        st.download_button(
            "Download all ratings (CSV)", RATINGS_CSV.read_bytes() if RATINGS_CSV.exists() else b"",
            file_name="ratings.csv", mime="text/csv", icon=":material/download:", disabled=not RATINGS_CSV.exists(),
        )
    chosen = rated.iloc[::-1]
    if KINDS.get(kind) is not None:
        chosen = chosen[chosen.rating == KINDS[kind]]
    if LISTS.get(where) is not None:
        chosen = chosen[chosen["mode"] == LISTS[where]]
    st.caption(f"{len(chosen):,} ratings{f', the newest {SHOWN} shown' if len(chosen) > SHOWN else ''}. "
               f"A copy of all of them is kept in {RATINGS_CSV}.")
    tz = common.history().ends_at.tz
    with st.container(key="panel-ratings"):
        for _, row in chosen.head(SHOWN).iterrows():
            words, undo = st.columns([6, 1], vertical_alignment="center")
            when = pd.Timestamp(row.rated_at).tz_convert(tz)
            genres = ", ".join(look.genre_label(genre) for genre in row.genres[:3]) if len(row.genres) else ""
            words.markdown(
                f"**{RATING_WORDS[row.rating]}** · **{md(row.track_name)}** · {md(row.artist_name)}  \n"
                f":gray[{md(MODE_WORDS[row['mode']])}{', ' + md(genres) if genres else ''}, {when:%d %b %Y %H:%M}]"
            )
            undo.button("Undo", key=f"clear-{row.song_key}-{row.rated_at}", on_click=on_undo, args=(row,), width="stretch")
look.footer()
