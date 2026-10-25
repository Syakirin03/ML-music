"""The Recommendations page: songs to like, rate neutral or dislike, ten at a time, with Spotify's player.

Each card is one song: Spotify's player (a 30-second preview, or the whole song if you're logged in to Spotify in
the browser), its genres with the main one highlighted, your chance of liking it, why it is there, and links to
open it in Spotify or, with your account connected (Settings), to save it to your Liked Songs or a playlist.

Like, Neutral and Dislike only mark your choice on the card (click it again to take it back): nothing is saved
and the list stays as it is, so a song playing keeps playing. Once you have chosen every song of the list, or
press Next songs, your choices are saved together (exactly as python -m recommender.feedback saves them), the next
list is built with them, and the page goes back to its top. Changing a setting sends the choices made so far
first. Neutral means you heard it and have no opinion: the song leaves your lists, and nothing else changes.
The next list shows songs not shown yet since you last changed a setting, and gives other artists a turn: an
artist with MAX_PER_ARTIST songs shown since then comes back only if you liked one of them.
"""

import contextlib
import html
import io
from urllib.parse import quote

import requests
import streamlit as st

from app_pages import common, look
from app_pages.common import state
from recommender import spotify
from recommender.data import candidate_pool
from recommender.feedback import MIN_EACH, rate, remember_list
from recommender.profile import NoProfile
from recommender.ranker import HORIZON_DAYS
from recommender.recommend import MAX_PER_ARTIST, VARIETY, recommend_songs, similar_songs
from recommender.scoring import DORMANT_DAYS, FAMILIARITY, GENRE_BOOST, MORE_GENRES, POPULARITY_WEIGHT, WESTERN

MODES = {"Songs new to you": "discover", "Songs to rediscover": "rediscover", "Songs like one you name": "like"}
# The rating buttons of a card: action, label, icon. The one you chose is filled and checked.
RATE_BUTTONS = [
    ("like", "Like", ":material/thumb_up:"),
    ("neutral", "Neutral", ":material/remove:"),
    ("dislike", "Dislike", ":material/thumb_down:"),
]
SENT_WORDS = {"like": "liked", "neutral": "neutral", "dislike": "disliked"}
FAILURES = (FileNotFoundError, RuntimeError, LookupError, ValueError, NoProfile)
# The compact Spotify player, in pixels.
SPOTIFY_PLAYER_HEIGHT = 80
# Each card's player is a small page of its own that loads Spotify's player through Spotify's iFrame API
# (developer.spotify.com/documentation/embeds). When one starts playing, it says so on a channel that the
# other cards' pages listen to, and they pause: one song at a time. The pages share the app's origin, so
# BroadcastChannel reaches them all.
PLAYER_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><style>html, body {{ margin: 0; background: transparent; overflow: hidden; }}</style></head>
<body><div id="player"></div>
<script>
  const me = "{track}-" + Math.random();
  const channel = new BroadcastChannel("music-recommender-player");
  window.onSpotifyIframeApiReady = (api) => {{
    api.createController(document.getElementById("player"), {{ uri: "spotify:track:{track}", width: "100%", height: {height} }}, (player) => {{
      player.addListener("playback_started", () => channel.postMessage(me));
      channel.onmessage = (event) => {{ if (event.data !== me) player.pause(); }};
    }});
  }};
</script>
<script src="https://open.spotify.com/embed/iframe-api/v1" async></script>
</body></html>"""


def player_html(track: str) -> str:
    return PLAYER_PAGE.format(track=track, height=SPOTIFY_PLAYER_HEIGHT)


# Characters Streamlit's Markdown would read as formatting (and $ as maths) in song names.
MARKDOWN = str.maketrans({c: "\\" + c for c in "\\`*_{}[]()#+-.!|<>~$"})


def md(text) -> str:
    return str(text).translate(MARKDOWN)


def track_of(mode: str, spotify_ids: dict, row) -> str | None:
    """The song's Spotify track id: its own for songs of your history, the one found by search for new songs."""
    track = spotify_ids.get(row.song_key) if mode == "discover" else getattr(row, "track_id", None)
    return track if isinstance(track, str) and spotify.TRACK_ID.fullmatch(track) else None


def spotify_url(track: str | None, row) -> str:
    if track:
        return f"https://open.spotify.com/track/{track}"
    return getattr(row, "link", None) or "https://open.spotify.com/search/" + quote(f"{row.artist} {row.title}")


def sidebar() -> tuple[str, str, dict]:
    """The kind of list and its settings, kept while you visit other pages; and how far your ratings are."""
    with st.sidebar:
        st.subheader("Your list")
        label = st.radio("Show me", list(MODES), key=common.kept("mode", "Songs new to you"))
        common.remember("mode")
        mode = MODES[label]
        settings = {"top": st.slider("Songs per list", 5, 30, key=common.kept("top", 10))}
        common.remember("top")
        if mode == "discover":
            options = common.pool_genres()
            key = common.kept("genres", [genre for genre in MORE_GENRES if genre in options])
            state[key] = [genre for genre in state[key] if genre in options]
            settings["genres"] = tuple(st.multiselect(
                "More of these genres", options, key=key, format_func=look.genre_label,
                placeholder="Any genre: K-pop, Techno, EDM...",
                help="Songs whose main genre is one of these get the full boost, those where it comes second half of it, and so on",
            ))
            common.remember("genres")
            settings["genre_boost"] = st.slider(
                "How much more", 0.0, 1.0, step=0.05, key=common.kept("genre_boost", GENRE_BOOST),
                help="0: the genres above change nothing. Genres far from your taste need 0.5 or more to show",
            )
            settings["familiarity"] = st.slider(
                "Familiarity", 0.0, 1.0, step=0.05, key=common.kept("familiarity", FAMILIARITY),
                help="0: judge songs by their genres; 1: by whether you play the artist",
            )
            settings["variety"] = st.slider(
                "Variety", 0.0, 1.0, step=0.1, key=common.kept("variety", VARIETY["discover"]),
                help="Trade score for genres the list doesn't have yet",
            )
            settings["mainstream"] = st.slider(
                "Mainstream", 0.0, 0.5, step=0.05, key=common.kept("mainstream", POPULARITY_WEIGHT),
                help="How much songs with many Last.fm listeners are favoured",
            )
            settings["western"] = st.slider(
                "Less Western", 0.0, 1.0, step=0.05, key=common.kept("western", WESTERN),
                help="How far songs by Western artists are pushed down, so Japanese, Korean and Southeast Asian music comes first",
            )
            settings["new_artists"] = st.checkbox("Only artists I have never played", key=common.kept("new_artists", False))
            for name in ("genre_boost", "familiarity", "variety", "mainstream", "western", "new_artists"):
                common.remember(name)
        elif mode == "rediscover":
            settings["dormant_days"] = st.number_input(
                "Unplayed for at least this many days", 7, 3650, key=common.kept("dormant_days", DORMANT_DAYS),
            )
            settings["variety"] = st.slider(
                "Variety", 0.0, 1.0, step=0.1, key=common.kept("variety_rediscover", VARIETY["rediscover"]),
                help="Trade chance for genres the list doesn't have yet",
            )
            common.remember("dormant_days")
            common.remember("variety_rediscover")
        else:
            settings["song"] = st.text_input(
                "A song you know", key=common.kept("song", ""), placeholder='A title, or "artist - title"',
            ).strip()
            settings["same_artist"] = st.checkbox("Include the same artist's songs", key=common.kept("same_artist", False))
            common.remember("song")
            common.remember("same_artist")

        rated = common.current()
        new = rated[rated["mode"] == "discover"]
        st.divider()
        st.subheader("Your ratings")
        pending = len(state.get("choices", {}))
        st.caption(
            f"{len(rated):,} songs rated: {(rated.rating == 1).sum():,} liked, {(rated.rating == 2).sum():,} neutral, "
            f"{(rated.rating == -1).sum():,} disliked"
            + (f". {pending} more chosen on this list, sent with the rest." if pending else "")
        )
        for kind, value in (("Likes", 1), ("Dislikes", -1)):
            count = int((new.rating == value).sum())
            st.progress(min(count / MIN_EACH, 1.0), text=f"{kind} of songs new to you: {count} of the {MIN_EACH} a model needs")
    return label, mode, settings


def resting_artists() -> set[str]:
    """Artists with MAX_PER_ARTIST songs shown since the settings last changed, unless you liked one of their songs."""
    return {artist for artist, songs in state.artist_songs.items() if len(songs) >= MAX_PER_ARTIST} - state.liked_artists


def build(mode: str, settings: dict, key: tuple, exclude: frozenset, exclude_artists: frozenset, words: str) -> None:
    """A new list for the mode and settings, kept in the shown table so that it can be rated, as state.page."""
    library = common.library()
    printed = io.StringIO()
    try:
        with st.spinner(words), contextlib.redirect_stdout(printed):
            if mode == "like":
                table = similar_songs(
                    settings["song"], settings["top"], settings["same_artist"], library=library,
                    exclude=exclude | frozenset(state.rated_here),
                )
            else:
                table = recommend_songs(
                    top_n=settings["top"], mode=mode, familiarity=settings.get("familiarity", FAMILIARITY),
                    new_artists_only=settings.get("new_artists", False),
                    dormant_days=settings.get("dormant_days", DORMANT_DAYS), variety=settings["variety"],
                    mainstream=settings.get("mainstream", POPULARITY_WEIGHT), genres=settings.get("genres", MORE_GENRES),
                    genre_boost=settings.get("genre_boost", GENRE_BOOST), western=settings.get("western", WESTERN),
                    library=library, exclude=exclude, exclude_artists=exclude_artists,
                )
        list_id = remember_list(table, mode) if len(table) else None
    except FAILURES as error:
        state.page = {"key": key, "error": str(error)}
        state.choices = {}
        return
    spotify_ids, no_app = {}, None
    if mode == "discover" and len(table):
        try:
            with st.spinner("Finding the songs on Spotify..."):
                spotify_ids = spotify.find_track_ids(table)
        except spotify.NoSpotifyApp as error:
            no_app = str(error)
        except (RuntimeError, requests.RequestException) as error:
            printed.write(f"Note: Spotify search failed, so some songs have only a link: {error}\n")
    lines = printed.getvalue().splitlines()
    # The profile is rebuilt after every rating: expected here, so not worth a note.
    notes = [line.removeprefix("Note: ") for line in lines if line.startswith("Note:") and "newer than your profile" not in line]
    notes = [note[:1].upper() + note[1:] for note in notes]
    header = next((line for line in lines if not line.startswith("Note:")), "")
    saved = set()
    if common.connection() and len(table):
        tracks = [track_of(mode, spotify_ids, row) for row in table.itertuples()]
        try:
            you = common.account()
            saved = you.saved([track for track in tracks if track]) if you else set()
        except (spotify.NotConnected, RuntimeError, requests.RequestException) as error:
            notes.append(f"Spotify didn't say which of these are in your Liked Songs: {error}")
    state.shown_keys |= set(table.song_key)
    for artist, song in zip(table.artist_key, table.song_key):
        state.artist_songs.setdefault(artist, set()).add(song)
    state.page = {
        "key": key, "table": table, "list_id": list_id, "notes": notes, "header": header, "mode": mode,
        "spotify_ids": spotify_ids, "no_app": no_app, "saved": saved,
    }
    state.choices = {}


def songs_text(count: int) -> str:
    return f"{count} song{'' if count == 1 else 's'}"


def on_choose(position: int, song: dict, action: str) -> None:
    """Mark like, neutral or dislike on a card, or take the mark back by clicking it again. Nothing is sent until
    every song of the list has a mark (then it is sent at once) or you press Next songs."""
    choices = state.setdefault("choices", {})
    if choices.get(song["song_key"], {}).get("action") == action:
        del choices[song["song_key"]]
        return
    choices[song["song_key"]] = {"position": position, "action": action, **song}
    if len(choices) >= len(state.page["table"]):
        send_choices()
        state.action = "next"
        state.scroll = True


def on_next() -> None:
    send_choices()
    state.action = "next"
    state.scroll = True


def send_choices() -> None:
    """Save the choices marked on the list as ratings, do what Settings says on Spotify with the liked songs, and
    keep a summary for the bar at the top (state.last)."""
    choices = list(state.get("choices", {}).values())
    state.choices = {}
    state.last = None
    if not choices:
        return
    sent: dict[str, int] = {}
    for action, _, _ in RATE_BUTTONS:
        chosen = [choice for choice in choices if choice["action"] == action]
        for list_id in sorted({choice["list_id"] for choice in chosen}):
            positions = [str(choice["position"]) for choice in chosen if choice["list_id"] == list_id]
            try:
                rate(positions, action, list_id=list_id)
            except FAILURES as error:
                common.message(f"Some choices weren't saved: {error}")
                continue
            sent[action] = sent.get(action, 0) + len(positions)
    saved = [choice for choice in choices if choice["action"] in sent]
    liked = [choice for choice in saved if choice["action"] == "like"]
    state.rated_here |= {choice["song_key"] for choice in saved}
    state.liked_artists |= {choice["artist_key"] for choice in liked}
    state.last = {"sent": sent, "spotify_note": also_on_spotify([choice["track"] for choice in liked])}


def also_on_spotify(tracks: list[str | None]) -> str:
    """Save the songs you liked to your Liked Songs or a playlist, if Settings says so; returns what happened."""
    settings = common.site_settings()
    if settings["on_like"] == "rate" or not tracks or not common.connection():
        return ""
    found = [track for track in tracks if track]
    missing = len(tracks) - len(found)
    gap = f" ({songs_text(missing)} you liked {'isn' if missing == 1 else 'aren'}'t on Spotify under that title and artist)" if missing else ""
    if not found:
        return "None of the songs you liked is on Spotify under its title and artist, so none was saved there."
    try:
        you = common.account()
        if you is None:
            return ""
        if settings["on_like"] == "liked":
            you.save(found)
            return f"Also saved {songs_text(len(found))} to your Liked Songs on Spotify{gap}."
        if settings["playlist_id"]:
            you.add_to_playlist(settings["playlist_id"], found)
            return f"Also added {songs_text(len(found))} to your playlist {settings['playlist_name']} on Spotify{gap}."
        return "Choose a playlist on the Settings page to add liked songs to it."
    except (spotify.NotConnected, RuntimeError, requests.RequestException) as error:
        return f"Spotify didn't take them: {error}"


def on_save(track: str, title: str) -> None:
    try:
        you = common.account()
        if you is None:
            raise spotify.NotConnected("The website isn't connected to your Spotify account")
        you.save([track])
        state.page["saved"].add(track)
        common.message(f"Saved {title} to your Liked Songs")
    except (spotify.NotConnected, RuntimeError, requests.RequestException) as error:
        common.message(f"Not saved to Spotify: {error}")


def on_add(track: str, title: str, chooser: str, names: dict[str, str]) -> None:
    playlist = state.get(chooser)
    if not playlist:
        common.message("Choose a playlist first")
        return
    try:
        you = common.account()
        if you is None:
            raise spotify.NotConnected("The website isn't connected to your Spotify account")
        you.add_to_playlist(playlist, [track])
        common.message(f"Added {title} to {names.get(playlist, 'your playlist')}")
    except (spotify.NotConnected, RuntimeError, requests.RequestException) as error:
        common.message(f"Not added: {error}")


def value_html(mode: str, row) -> str:
    """Your chance of liking the song, highlighted, and what else ranks it."""
    liked = getattr(row, "liked_chance", float("nan"))
    if mode == "like":
        return look.meter(min(row.similarity, 1.0), "similarity to the song you named", f"{row.similarity:.2f}")
    if liked == liked:  # not NaN: the model learned from your ratings ranks the list
        second = ""
        if mode == "rediscover" and row.chance == row.chance:
            second = f'<div class="mp-second">{row.chance:.0%} chance you play it again within {HORIZON_DAYS} days, by your history</div>'
        return look.meter(liked, "chance you'll like it") + second
    if mode == "rediscover" and row.chance == row.chance:
        return look.meter(row.chance, f"chance you play it again within {HORIZON_DAYS} days")
    return f'<div class="mp-second">Score {row.score:.2f}: the model learned from your ratings isn\'t in use yet</div>'


def info_html(mode: str, position: int, row, picked) -> str:
    genres = [genre for genre in str(row.genres).split(", ") if genre] if row.genres else []
    # The chances are in the meter above the reasons, so they aren't repeated in them.
    why = "; ".join(part for part in str(row.why).split("; ") if "chance you" not in part)
    return (
        f'<div class="mp-song-head"><span class="mp-pos">{position}</span><div>'
        f'<div class="mp-song-title">{html.escape(str(row.title))}</div>'
        f'<div class="mp-song-artist">{html.escape(str(row.artist))}</div></div></div>'
        + (f'<div class="mp-chips">{look.chips(genres, picked)}</div>' if genres else "")
        + value_html(mode, row)
        + f'<div class="mp-why">{html.escape(why[:1].upper() + why[1:])}</div>'
    )


def card(page: dict, position: int, row, picked, account: dict | None) -> None:
    mode = page["mode"]
    ids = f"{page['list_id']}-{position}"
    track = track_of(mode, page["spotify_ids"], row)
    song = {"song_key": row.song_key, "artist_key": row.artist_key, "title": str(row.title), "artist": str(row.artist),
            "track": track, "list_id": page["list_id"]}
    chosen = state.get("choices", {}).get(row.song_key, {}).get("action")
    with st.container(key=f"card-{ids}"):
        player, info, actions = st.columns([2.4, 4.4, 1.25], gap="medium", vertical_alignment="center")
        with player:
            if track:
                st.iframe(player_html(track), height=SPOTIFY_PLAYER_HEIGHT)
            elif mode == "discover" and row.song_key in page["spotify_ids"]:
                st.caption("Spotify doesn't have it under this title and artist.")
            else:
                st.caption("No Spotify player for this song.")
            with st.container(horizontal=True, gap="small"):
                st.link_button("Open in Spotify", spotify_url(track, row), icon=":material/open_in_new:", type="tertiary")
                if account and track and not account["error"]:
                    if track in page["saved"]:
                        st.button("In Liked Songs", key=f"saved-{ids}", icon=":material/favorite:", type="tertiary", disabled=True)
                    else:
                        st.button("Save", key=f"save-{ids}", icon=":material/favorite_border:", type="tertiary",
                                  on_click=on_save, args=(track, song["title"]), help="Save to your Liked Songs on Spotify")
                    with st.popover("Playlist", icon=":material/playlist_add:", type="tertiary"):
                        chooser = f"playlist-{ids}"
                        names = {playlist["id"]: playlist["name"] for playlist in account["playlists"]}
                        st.selectbox("Add to", list(names), key=chooser, format_func=names.get, index=None,
                                     placeholder="Choose one of your playlists")
                        st.button("Add", key=f"add-{ids}", type="primary", on_click=on_add,
                                  args=(track, song["title"], chooser, names))
        with info:
            st.html(info_html(mode, position, row, picked))
        with actions:
            for action, label, icon in RATE_BUTTONS:
                st.button(label, key=f"rate-{action}-{ids}", width="stretch",
                          icon=":material/check_circle:" if chosen == action else icon,
                          type="primary" if chosen == action else "secondary",
                          help="Your choice: click again to take it back" if chosen == action else None,
                          on_click=on_choose, args=(position, song, action))


def intro(mode: str, page: dict | None) -> tuple[str, str]:
    """The page's title and sentence for a kind of list."""
    if mode == "discover":
        pool = candidate_pool(common.history(), "now") if len(common.history().candidates) else None
        size = f" of {len(pool):,} songs by {pool.artist_key.nunique():,} artists" if pool is not None else ""
        return "Songs new to you", (
            f"Songs you have never played, picked from a pool{size} that Last.fm's charts and similar-artist lists "
            "give for your taste. Choose Like, Neutral or Dislike for each: once you've chosen them all, or press "
            "Next songs, your choices are sent and the next songs are picked with them."
        )
    if mode == "rediscover":
        return "Songs to rediscover", (
            "Songs you loved that have gone quiet: at least three full listens, none lately. A model trained on your "
            "history guesses which you would go back to, and one trained on your ratings which you want back now."
        )
    return "Songs like one you name", (
        "Songs from your history that you play in the same sessions as the one you name, or that share its genres. "
        "Type a title in the sidebar."
    )


def ordering(page: dict) -> list[tuple[str, str]]:
    """The figures under the title: how the list is ranked."""
    table = page["table"]
    if not len(table):
        return []
    if "liked_chance" in table and table.liked_chance.notna().any():
        return [("Ranked by", "a model learned from your ratings"), (f"{table.liked_chance.mean():.0%}", "average chance you'll like these")]
    if page["mode"] == "rediscover" and table.chance.notna().any():
        return [("Ranked by", "a model learned from your history")]
    if page["mode"] == "like":
        return [("Ranked by", "how alike you play them, and their genres")]
    return [("Ranked by", "the hand-set score")]


# The page.
label, mode, settings = sidebar()
for name, empty in (("shown_keys", set), ("artist_songs", dict), ("liked_artists", set), ("rated_here", set)):
    state.setdefault(name, empty())
key = (mode, tuple(sorted(settings.items())))
page = state.get("page")
action = state.pop("action", None)
if page is None or page.get("key") != key:
    # A setting changed, or the list was let go (Reload on Settings): choices marked are sent before it changes.
    if state.get("choices"):
        send_choices()
    else:
        state.last = None
if mode == "like" and not settings["song"]:
    page = None
elif page is None or page.get("key") != key:
    state.shown_keys, state.artist_songs, state.liked_artists, state.rated_here = set(), {}, set(), set()
    build(mode, settings, key, frozenset(), frozenset(), "Picking songs...")
    page = state.page
elif action == "next":
    build(mode, settings, key, frozenset(state.shown_keys), frozenset(resting_artists()), "Picking your next songs...")
    page = state.page

title, sentence = intro(mode, page)
look.hero("Recommendations", title, html.escape(sentence), ordering(page) if page and "table" in page else None)

last = state.get("last")
if last and last["sent"]:
    total = sum(last["sent"].values())
    parts = ", ".join(f"{count} {SENT_WORDS[kind]}" for kind, count in last["sent"].items())
    with st.container(key="panel-last", horizontal=True, vertical_alignment="center", gap="medium"):
        st.markdown(f"**Sent {songs_text(total)}**: {parts}. The songs below were picked with them. {md(last['spotify_note'])}")
        st.page_link("app_pages/ratings.py", label="Change them", icon=":material/edit:")

if page is not None and "table" in page and len(page["table"]):
    marked, total = len(state.get("choices", {})), len(page["table"])
    with st.container(key="panel-progress"):
        st.progress(marked / total, text=(
            f"Chosen {marked} of {total}. Your choices are sent, and the next songs picked, once you've chosen all "
            f"{total} or press Next songs. Click a choice again to take it back."
        ))

if page is None:
    st.info("Type a song you know in the sidebar, and songs like it appear here.")
elif "error" in page:
    st.error(page["error"])
else:
    if mode == "like" and page["header"]:
        st.caption(md(page["header"].rstrip(":")))
    for note in page["notes"]:
        st.info(note)
    if page.get("no_app"):
        st.caption(md(f"To play samples of songs new to you here: {page['no_app']}"))
    account = common.connection()
    picked = set(settings.get("genres", ()))
    for position, row in enumerate(page["table"].itertuples(), 1):
        card(page, position, row, picked, account)
    if len(page["table"]):
        st.space("small")
        marked = len(state.get("choices", {}))
        with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
            st.button(f"Send {songs_text(marked)} and show the next ones" if marked else "Next songs",
                      icon=":material/arrow_forward:", on_click=on_next, type="primary")
            st.caption(
                ("The songs you didn't choose are skipped. " if marked else "None chosen: these songs are skipped. ")
                + f"The next list shows songs not shown yet since you last changed a setting, and gives other artists "
                f"a turn: those with {MAX_PER_ARTIST} songs shown come back only if you liked one of them."
            )
look.footer()
if state.pop("scroll", False):
    look.scroll_to_top()
