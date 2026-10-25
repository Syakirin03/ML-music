"""Recommend songs from your taste profile, and find songs like one you name.

    python -m recommender.recommend [--top N] [--familiarity F] [--new-artists] [--variety V]
    python -m recommender.recommend --rediscover [--top N] [--dormant-days D] [--formula [--familiarity F]] [--variety V]
    python -m recommender.recommend --like SONG [--top N] [--same-artist]

recommend_songs() has two modes:

    discover    (the default) songs you have never played, from the "now" pool
                of candidates.csv, which scripts/fetch_candidates.py builds:
                the most played songs on Last.fm of your top artists and of
                the artists Last.fm finds similar to them
    rediscover  songs from your history that you liked (at least LIKED_LISTENS
                listens) but haven't played for dormant_days

scoring.py explains the score that ranks them and its parts (song_match,
genre_match, artist_match, similar_match, popularity, genre_pick), and familiarity, which
moves the weight from "sounds like what you play" to "by an artist you play".
In rediscover, the model that python -m recommender.ranker learns from your
history ranks the songs instead, by your chance of playing each again within
90 days, and says which features raised it most; the hand-set rediscover score
is still there with ranker="formula" (--formula), and familiarity only
changes that one. Without a trained model, rediscover uses the formula.

In discover, songs with the same score come in order of Last.fm listeners; in
rediscover, the one you played more comes first. At most MAX_PER_ARTIST songs
per artist make the list, and variety (0 to 1, default VARIETY for the mode)
trades score for genres the list doesn't have yet: each next song is the one
with the best (1 - variety) * score - variety * its genre similarity to the
most similar song already listed (maximal marginal relevance). At 0 the list
is simply the best songs. evaluate.py shows how well each ranking predicts
what you went on to play, and what variety costs.

Songs you have rated (feedback.py) aren't recommended again, and songs new to
you leave out an artist once you disliked STRIKES of their songs there and
liked none. Once a model
learned from your ratings of songs new to you is in use, it orders the best
SHORTLIST songs of discover by your chance of liking each (liked_chance).
The CLI keeps every list it prints, so that you can rate its songs by number.

similar_songs() is song-to-song similarity: the songs in your history most
like the one you name, by other artists unless you ask for the same one. They
are ranked by how alike you play them (the session vectors of sessions.py),
how alike their genres are, and a little by how much you play them lately,
with the weights sessions.py explains.

Run it from the project folder with python -m recommender.recommend, or as a
file, for example with VS Code's Run button (which passes no options).
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

if not __package__:
    # Started as a file (VS Code's Run button, or python recommender/recommend.py) instead of with
    # python -m: Python then looks for imports in recommender/, so add the project folder first.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recommender.data import NEUTRAL, Library, current_ratings, load_library, song_key
from recommender.features import genre_list, genre_matrix, genre_vocabulary
from recommender.feedback import FeedbackModel, feedback_model, like_chances, model_numbers, remember_list
from recommender.profile import NoProfile, TasteProfile, build_profile, load_profile, song_evidence
from recommender.ranker import (
    HORIZON_DAYS, Ranker, candidate_features, chances, contributions, load_ranker, rating_numbers, rediscover_feedback,
)
from recommender.scoring import (
    DORMANT_DAYS,
    FAMILIARITY,
    GENRE_BOOST,
    MORE_GENRES,
    POPULARITY_WEIGHT,
    WESTERN,
    check_familiarity,
    discovery_candidates,
    rediscovery_candidates,
    score_candidates,
    score_songs,
)
from recommender.sessions import like_parts, like_score, shared_sessions, song_space

# At most this many songs by one artist in a list of recommendations.
MAX_PER_ARTIST = 2
# Songs new to you leave out an artist once you disliked this many of their songs there and liked none.
# You know the artist's songs you play, but the rest of their catalogue can still be not for you, and a
# dislike counts too little against hundreds of plays to show it. Set from the author's first 46 ratings,
# replayed in time order: it would have spared 7 dislikes and cost 1 like (docs/results.md).
STRIKES = 3
# How much a list trades score for variety, per mode (see the docstring). Chosen on the valid window of
# evaluate.py: the largest of 0, 0.1, 0.2, 0.3 and 0.5 that lost none of the songs you went on to play from
# the top 10 and top 30. Rediscover lists are varied already, and any variety lost some.
VARIETY = {"discover": 0.5, "rediscover": 0.0}
# Variety picks from this many of the best songs, at least.
SHORTLIST = 300

MODES = ("discover", "rediscover")
RANKERS = ("learned", "formula")
# Every table also has song_key and artist_key, which identify the song when you rate it (feedback.py).
REDISCOVER_COLUMNS = [
    "track_id", "title", "artist", "genres", "score", "chance", "liked_chance", "song_match", "genre_match", "artist_match", "plays",
    "last_played", "why", "song_key", "artist_key",
]
# What each feature of the learned model is about, for "mostly because of ...". They name the topic, not a
# direction: a feature can raise a chance by being high or, with a negative weight, by being low.
REASONS = {
    "song_match": "how much you played it lately",
    "genre_match": "its genres",
    "artist_match": "its artist",
    "log_listens": "how often you played it",
    "log_days_dormant": "when you last played it",
    "log_days_known": "how long you have known it",
    "finish_rate": "how often you finished it",
    "skip_rate": "how often you skipped it",
    "artist_recent": "how much you play its artist this month",
    "genre_recent": "how much you play its genre this month",
    "has_genres": "its genres",
}
DISCOVER_COLUMNS = [
    "title", "artist", "genres", "score", "liked_chance", "genre_match", "artist_match", "similar_match", "popularity",
    "genre_pick", "listeners", "link", "why", "song_key", "artist_key",
]
SIMILAR_COLUMNS = [
    "track_id", "title", "artist", "genres", "similarity", "session_similarity", "genre_similarity", "plays", "why",
    "song_key", "artist_key",
]
# The why of a song that only variety put on the list.
FOR_VARIETY = "picked for variety: its genres differ from the songs above it"
# A song you never played next to the one you named, but with session similarity at least this, is
# "played with the same songs".
SAME_COMPANY = 0.3


def profile_or_build(library: Library) -> TasteProfile:
    """The saved profile, or a new one (saved) if there is none yet or your plays or ratings are newer than it."""
    try:
        profile = load_profile()
    except NoProfile:
        print("Note: no saved taste profile yet, so one is built and saved now")
        return build_profile(library=library)
    newer_ratings = len(library.ratings) and library.ratings.rated_at.max() > profile.built_at
    if profile.as_of < library.ends_at or newer_ratings:
        print(f"Note: your {'ratings are' if newer_ratings else 'history is'} newer than your profile, so it is rebuilt and saved now")
        return build_profile(library=library)
    return profile


def struck_artists(library: Library) -> dict[str, str]:
    """artist_key -> name of the artists with at least STRIKES dislikes and no like among your ratings of songs new to you."""
    rated = current_ratings(library.ratings)
    rated = rated[rated["mode"] == "discover"]
    likes = rated.rating.eq(1).groupby(rated.artist_key).sum()
    dislikes = rated.rating.eq(-1).groupby(rated.artist_key).sum()
    names = rated.drop_duplicates("artist_key").set_index("artist_key").artist_name
    return {artist: names[artist] for artist in dislikes.index[(dislikes >= STRIKES) & (likes == 0)]}


def rated_keys(library: Library, ratings: tuple[int, ...] = (1, -1, NEUTRAL)) -> set[str]:
    """song_keys of the songs whose current rating is one of ratings (by default any: liked, disliked or neutral)."""
    rated = current_ratings(library.ratings)
    return set(rated.song_key[rated.rating.isin(ratings)])


def artist_phrase(weight: float, name: str) -> str:
    if weight >= 0.6:
        return f"{name} is one of your most played artists"
    if weight >= 0.3:
        return f"you play {name} often"
    if weight > 0:
        return f"you have played {name} a little"
    if weight == 0:
        return f"{name} is new to you"
    return f"you skip {name} more than you play them"


def similar_phrase(links: dict[str, float], profile: TasteProfile) -> str | None:
    """The two artists you like that point to this one most strongly on Last.fm, if any."""
    liked = profile.artists.weight.clip(lower=0)
    strength = {seed: liked.get(seed.strip(), 0.0) * match for seed, match in links.items()}
    names = [seed for seed in sorted(strength, key=lambda seed: -strength[seed]) if strength[seed] > 0][:2]
    return f"Last.fm finds them similar to {' and '.join(names)}" if names else None


def genre_phrase(song_genres: list[str], profile: TasteProfile) -> str:
    """The song's genres that count most toward its genre_match, or why it has none."""
    if not song_genres:
        return "Last.fm has no genres for it"
    weights = profile.genres.reindex(song_genres).fillna(0.0)
    contributions = (weights / np.arange(1, len(song_genres) + 1)).sort_values(ascending=False)
    liked = [genre for genre, value in contributions.items() if value > 0][:2]
    if liked:
        return f"{' and '.join(liked)} {'is a genre' if len(liked) == 1 else 'are genres'} you play lately"
    return f"{', '.join(song_genres[:2])}: genres you rarely play"


def count_text(n: int) -> str:
    """1234567 -> '1.2M', 45678 -> '46K'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n / 1000:.0f}K" if n >= 1000 else str(n)


def varied_top(scores: np.ndarray, genre_rows, artist_keys: np.ndarray, variety: float, top_n: int) -> np.ndarray:
    """Positions of the songs to list, in list order, from songs ranked best first.

    scores, genre_rows (features.py) and artist_keys describe the ranked songs, in that order. At most
    MAX_PER_ARTIST songs per artist. With variety above 0, each next song is the one with the best
    (1 - variety) * score - variety * likeness, among the best max(SHORTLIST, 3 * top_n): score is rescaled
    to 0-1 over them, and likeness is the song's genre similarity to the most similar song listed so far.
    A song without genres counts as averagely alike.
    """
    check_variety(variety)
    seen = pd.Series(artist_keys).groupby(artist_keys).cumcount().to_numpy()
    if variety == 0:
        return np.flatnonzero(seen < MAX_PER_ARTIST)[:top_n]
    shortlist = min(len(scores), max(SHORTLIST, 3 * top_n))
    score = scores[:shortlist]
    span = score.max() - score.min() if shortlist else 0.0
    score = (score - score.min()) / span if span > 0 else np.zeros(shortlist)
    rows = genre_rows[:shortlist]
    alike = (rows @ rows.T).toarray()
    has_genres = np.asarray(rows.sum(axis=1)).ravel() > 0
    neutral = np.median(alike[np.ix_(has_genres, has_genres)]) if has_genres.sum() > 1 else 0.0
    alike[~has_genres, :] = neutral
    alike[:, ~has_genres] = neutral

    listed, per_artist = [], {}
    likeness = np.zeros(shortlist)
    available = np.ones(shortlist, dtype=bool)
    while len(listed) < top_n and available.any():
        best = int(np.argmax(np.where(available, (1 - variety) * score - variety * likeness, -np.inf)))
        available[best] = False
        if per_artist.get(artist_keys[best], 0) >= MAX_PER_ARTIST:
            continue
        per_artist[artist_keys[best]] = per_artist.get(artist_keys[best], 0) + 1
        listed.append(best)
        likeness = np.maximum(likeness, alike[best])
    return np.array(listed, dtype=int)


def check_variety(variety: float) -> None:
    if not 0 <= variety <= 1:
        raise ValueError(f"variety must be between 0 and 1, not {variety}")


def list_positions(scores: np.ndarray, genre_rows, artist_keys: np.ndarray, variety: float, top_n: int) -> tuple[np.ndarray, set[int]]:
    """varied_top(), and the positions it listed only for variety: those the plain list (variety 0) leaves out."""
    listed = varied_top(scores, genre_rows, artist_keys, variety, top_n)
    plain = set(varied_top(scores, genre_rows, artist_keys, 0.0, top_n)) if variety > 0 else set(listed)
    return listed, set(listed) - plain


def learned_ranker(library: Library) -> Ranker | None:
    """The saved rediscover model, or None (with a note) if there is none yet. Notes when it is out of date."""
    model = load_ranker()
    if model is None:
        print("Note: no trained model yet, so the hand-set formula ranks them: run python -m recommender.ranker")
    elif model.trained_until < library.ends_at:
        print(
            f"Note: the model learned from your history up to {model.trained_until:%d %b %Y}: "
            "run python -m recommender.ranker to update it"
        )
    return model


def rediscover(
    library: Library, profile: TasteProfile, top_n: int, familiarity: float, dormant_days: float, ranker: str,
    variety: float, exclude: set[str] | frozenset[str] = frozenset(), exclude_artists: set[str] | frozenset[str] = frozenset(),
) -> pd.DataFrame:
    evidence = song_evidence(library, profile.as_of)
    candidates = rediscovery_candidates(library, profile.as_of, dormant_days, evidence)
    if candidates.empty:
        print(f"Note: no songs you liked have gone unplayed for {dormant_days:g} days; try a smaller dormant_days")
        return pd.DataFrame(columns=REDISCOVER_COLUMNS)

    scores = score_songs(profile, library, candidates.index.to_numpy(), familiarity, evidence)
    scores["listens"] = candidates.listens.to_numpy()
    model = learned_ranker(library) if ranker == "learned" else None
    learned = rediscover_feedback(library)
    if model is not None or learned.model is not None:
        # The same songs in the same order: both come from rediscovery_candidates().
        features = candidate_features(library, profile.as_of, profile, dormant_days)
    if model is not None:
        scores["score"] = scores["chance"] = chances(model, features)
        reasons = contributions(model, features)
    else:
        scores["chance"] = np.nan
    scores["liked_chance"] = np.nan
    if learned.model is not None:
        scores["liked_chance"] = like_chances(learned, rating_numbers(features), library.songs.genres.iloc[scores.song])
    keys = song_key(library.songs.artist_name, library.songs.track_name)
    scores = scores[~keys.iloc[scores.song].isin(rated_keys(library) | set(exclude)).to_numpy()]  # rated or excluded
    scores = scores[~library.songs.artist_key.iloc[scores.song].isin(set(exclude_artists)).to_numpy()]
    # Songs with the same score: the one you played more first.
    ranked = scores.sort_values(["score", "listens"], ascending=False)
    order = ranked.score.to_numpy()
    if learned.model is not None:  # as in discover: the SHORTLIST best by score, in order of your chance of liking each
        ranked = pd.concat([ranked.iloc[:SHORTLIST].sort_values(["liked_chance", "score"], ascending=False), ranked.iloc[SHORTLIST:]])
        order = np.concatenate([ranked.liked_chance.to_numpy()[:SHORTLIST], np.zeros(max(len(ranked) - SHORTLIST, 0))])
    genre_rows = genre_matrix(library.songs.genres.iloc[ranked.song].tolist(), genre_vocabulary(library.songs.genres))
    listed, for_variety = list_positions(
        order, genre_rows, library.songs.artist_key.iloc[ranked.song].to_numpy(), variety, top_n
    )

    rows = []
    for position, item in zip(listed, ranked.iloc[listed].itertuples()):
        song = library.songs.iloc[item.song]
        history = candidates.loc[item.song]
        song_genres = genre_list(song.genres)
        why = [
            artist_phrase(item.artist_match, song.artist_name),
            genre_phrase(song_genres, profile),
            f"you played it {int(history.plays)} times, last on {history.last_played_at:%d %b %Y}",
        ]
        if model is not None:
            raised = reasons.loc[item.song].sort_values(ascending=False)
            topics = list(dict.fromkeys(REASONS[feature] for feature, value in raised.items() if value > 0))[:2]
            chance = f"{item.chance:.0%} chance you play it again within {HORIZON_DAYS} days"
            why.append(f"{chance}, mostly because of {' and '.join(topics)}" if topics else chance)
        if not np.isnan(item.liked_chance):
            why.append(f"{item.liked_chance:.0%} chance you like it now, learned from your {learned.ratings} ratings")
        if position in for_variety:
            why.append(FOR_VARIETY)
        rows.append({
            "track_id": song.track_id,
            "title": song.track_name,
            "artist": song.artist_name,
            "genres": ", ".join(song_genres),
            "score": item.score,
            "chance": item.chance,
            "liked_chance": item.liked_chance,
            "song_match": item.song_match,
            "genre_match": item.genre_match,
            "artist_match": item.artist_match,
            "plays": int(history.plays),
            "last_played": history.last_played_at.date(),
            "why": "; ".join(why),
            "song_key": keys.iloc[item.song],
            "artist_key": song.artist_key,
        })
    return pd.DataFrame(rows, columns=REDISCOVER_COLUMNS)


def discover(
    library: Library, profile: TasteProfile, top_n: int, familiarity: float, new_artists_only: bool, variety: float,
    exclude: set[str] | frozenset[str] = frozenset(), exclude_artists: set[str] | frozenset[str] = frozenset(),
    mainstream: float = POPULARITY_WEIGHT, genres: Sequence[str] = MORE_GENRES, genre_boost: float = GENRE_BOOST,
    western: float = WESTERN,
) -> pd.DataFrame:
    if library.candidates.empty:
        print("Note: no pool of new songs yet: run python scripts/fetch_candidates.py (about 10 minutes the first time)")
        return pd.DataFrame(columns=DISCOVER_COLUMNS)
    songs = discovery_candidates(library, profile.as_of)
    songs = songs[~songs.song_key.isin(rated_keys(library) | set(exclude))].reset_index(drop=True)  # rated or excluded
    struck = struck_artists(library)
    if struck.keys() & set(songs.artist_key):
        names = ", ".join(sorted(name for artist, name in struck.items() if artist in set(songs.artist_key)))
        print(f"Note: left out {names}: you disliked {STRIKES} or more of their songs new to you and liked none")
    songs = songs[~songs.artist_key.isin(struck.keys() | set(exclude_artists))].reset_index(drop=True)
    if new_artists_only:
        played = library.plays.song[library.plays.started_at < profile.as_of].unique()
        songs = songs[~songs.artist_key.isin(set(library.songs.artist_key.iloc[played]))].reset_index(drop=True)
    if songs.empty:
        print("Note: no songs in the pool qualify")
        return pd.DataFrame(columns=DISCOVER_COLUMNS)

    learned = feedback_model(library.ratings, library.candidates)
    played = set(library.songs.artist_key.iloc[library.plays.song[library.plays.started_at < profile.as_of].unique()])
    ranked, listed, for_variety = rank_discover(
        library, profile, songs, familiarity, variety, top_n, learned, mainstream, genres, genre_boost, western
    )

    rows = []
    for position, item in zip(listed, ranked.iloc[listed].itertuples()):
        song = songs.loc[item.Index]
        song_genres = genre_list(song.genres)
        if song.artist_key in played or item.artist_match == 0:
            why = [artist_phrase(item.artist_match, song.artist_name)]
        else:  # an artist you never played, known to the profile only from your ratings
            why = [f"you {'liked' if item.artist_match > 0 else 'disliked'} songs by {song.artist_name} here"]
        if item.artist_match == 0 and (similar := similar_phrase(song.similar_to, profile)):
            why.append(similar)
        elif item.artist_match == 0 and song.source == "genre":
            why.append(f"one of Last.fm's top {song.via_genre} artists, a genre you play a lot")
        if genre_boost > 0 and item.genre_pick > 0:
            picked = next(genre for genre in song_genres if genre in set(genres))
            why.append(f"{picked}, a genre you asked for more of")
        why += [
            genre_phrase(song_genres, profile),
            f"#{song.artist_rank} of their songs on Last.fm, {count_text(int(song.listeners))} listeners",
        ]
        if not np.isnan(item.liked_chance):
            why.append(f"{item.liked_chance:.0%} chance you like it, learned from your {learned.ratings} ratings")
        if position in for_variety:
            why.append(FOR_VARIETY)
        rows.append({
            "title": song.track_name,
            "artist": song.artist_name,
            "genres": ", ".join(song_genres),
            "score": item.score,
            "liked_chance": item.liked_chance,
            "genre_match": item.genre_match,
            "artist_match": item.artist_match,
            "similar_match": item.similar_match,
            "popularity": item.popularity,
            "genre_pick": item.genre_pick,
            "listeners": int(song.listeners),
            "link": "https://open.spotify.com/search/" + quote(f"{song.artist_name} {song.track_name}"),
            "why": "; ".join(why),
            "song_key": song.song_key,
            "artist_key": song.artist_key,
        })
    return pd.DataFrame(rows, columns=DISCOVER_COLUMNS)


def rank_discover(
    library: Library, profile: TasteProfile, songs: pd.DataFrame, familiarity: float, variety: float, top_n: int,
    learned: FeedbackModel | None = None, mainstream: float = POPULARITY_WEIGHT, genres: Sequence[str] = (),
    genre_boost: float = 0.0, western: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray, set[int]]:
    """The discover scores of songs (a pool's rows), best first; the positions in it to list; those listed for variety.

    With a feedback model in use (feedback.py), the SHORTLIST best songs by score come first, in order of
    the model's chance that you like each (liked_chance); otherwise liked_chance is NaN.
    """
    scores = score_candidates(profile, library, songs, familiarity, mainstream, genres, genre_boost, western)
    scores["listeners"] = songs.listeners.to_numpy()
    ranked = scores.sort_values(["score", "listeners"], ascending=False)
    ranked["liked_chance"] = np.nan
    order = ranked.score.to_numpy()
    if learned is not None and learned.model is not None:
        best = ranked.iloc[:SHORTLIST].assign(liked_chance=lambda top: like_chances(
            learned, model_numbers(top, songs.loc[top.index]), songs.genres.loc[top.index]
        ))
        ranked = pd.concat([best.sort_values(["liked_chance", "score"], ascending=False), ranked.iloc[SHORTLIST:]])
        order = ranked.liked_chance.fillna(0.0).to_numpy()
    genre_rows = genre_matrix(songs.genres.iloc[ranked.index].tolist(), genre_vocabulary(songs.genres))
    listed, for_variety = list_positions(order, genre_rows, songs.artist_key.to_numpy()[ranked.index], variety, top_n)
    return ranked, listed, for_variety


def recommend_songs(
    top_n: int = 10,
    mode: str = "discover",
    familiarity: float = FAMILIARITY,
    new_artists_only: bool = False,
    dormant_days: float = DORMANT_DAYS,
    ranker: str = "learned",
    variety: float | None = None,
    profile: TasteProfile | None = None,
    library: Library | None = None,
    exclude: set[str] | frozenset[str] = frozenset(),
    exclude_artists: set[str] | frozenset[str] = frozenset(),
    mainstream: float = POPULARITY_WEIGHT,
    genres: Sequence[str] = MORE_GENRES,
    genre_boost: float = GENRE_BOOST,
    western: float = WESTERN,
) -> pd.DataFrame:
    """Recommended songs, best first: one row per song, with its score, the score's parts and why it is there.

    mode is "discover" (songs you've never played; new_artists_only keeps only artists you've never played)
    or "rediscover" (songs you liked but haven't played for dormant_days). Rediscover ranks with the model
    from python -m recommender.ranker (score = chance of playing it again within 90 days), or with the
    hand-set formula if ranker="formula" or no model is trained; familiarity only changes the formula and
    discover. variety, 0 to 1, trades score for genres the list doesn't have yet (None: VARIETY of the mode).
    In discover, mainstream is the weight of Last.fm popularity, genre_boost how much songs in genres are
    favoured, and western how far songs by Western (not Asian) artists are pushed down (scoring.py).
    Uses your saved taste profile, rebuilding it if there is none or your plays or ratings are newer. Songs
    you rated are left out, and so are the song_keys in exclude, such as songs shown already, and the songs
    of the artist_keys in exclude_artists. Discover also leaves out artists you struck out (STRIKES). An
    empty table means nothing qualifies, with a note saying why.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
    if ranker not in RANKERS:
        raise ValueError(f"ranker must be one of {', '.join(RANKERS)}, not {ranker!r}")
    check_familiarity(familiarity)
    variety = VARIETY[mode] if variety is None else variety
    check_variety(variety)
    library = library or load_library()
    profile = profile or profile_or_build(library)
    if mode == "rediscover":
        return rediscover(library, profile, top_n, familiarity, dormant_days, ranker, variety, exclude, exclude_artists)
    return discover(library, profile, top_n, familiarity, new_artists_only, variety, exclude, exclude_artists, mainstream, genres, genre_boost, western)


def find_song(library: Library, query: str) -> int:
    """The most played song whose title, "artist - title" or "title - artist" contains query, ignoring case."""
    songs = library.songs
    wanted = query.casefold().strip()
    title = songs.track_name.str.casefold()
    artist = songs.artist_name.str.casefold()
    for haystack in (title, artist + " - " + title, title + " - " + artist):
        matches = np.flatnonzero(haystack.str.contains(wanted, regex=False).to_numpy())
        if len(matches):
            return int(matches[0])  # songs are ordered most played first
    raise LookupError(f"No song in your listening history matches {query!r}")


def times(n: int) -> str:
    return f"{n} time{'s' if n != 1 else ''}"


def similar_songs(
    query: str, top_n: int = 10, same_artist: bool = False, library: Library | None = None,
    exclude: set[str] | frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Songs in your history most like the song that matches query (see find_song): played in the same company
    (session_similarity), with the same genres (genre_similarity), and a little by how much you play them lately.
    Songs you disliked are left out, and so are the song_keys in exclude (the website's songs shown or rated already)."""
    library = library or load_library()
    seed = find_song(library, query)
    songs = library.songs
    space = song_space(library, library.ends_at)
    seed_genres = genre_list(songs.genres.iloc[seed])
    if not seed_genres and not np.abs(space.vectors[seed]).any():
        raise LookupError(
            f"Last.fm has no genres for {songs.track_name.iloc[seed]} by {songs.artist_name.iloc[seed]}, and you "
            "never played it near other songs, so there is nothing to compare it on yet"
        )
    session, genre = (part[0] for part in like_parts(space, np.array([seed])))
    similarity = like_score(space, session, genre)
    together = shared_sessions(space, seed)
    evidence = song_evidence(library, library.ends_at)
    keys = song_key(songs.artist_name, songs.track_name)

    keep = (np.arange(len(songs)) != seed) & ((session > 0) | (genre > 0))
    keep &= ~keys.isin(rated_keys(library, (-1,)) | set(exclude)).to_numpy()  # songs you disliked, or excluded
    if not same_artist:
        keep &= (songs.artist_key != songs.artist_key.iloc[seed]).to_numpy()
    if songs.is_music.iloc[seed]:
        keep &= songs.is_music.to_numpy()
    found = np.flatnonzero(keep)
    found = found[np.argsort(-similarity[found], kind="stable")][:top_n]

    print(
        f"Songs like {songs.track_name.iloc[seed]} by {songs.artist_name.iloc[seed]} "
        f"({', '.join(seed_genres) or 'no genres on Last.fm'}):"
    )
    result = []
    for number in found:
        song = songs.iloc[number]
        song_genres = genre_list(song.genres)
        shared = [g for g in song_genres if g in seed_genres]
        plays = int(evidence.plays.iloc[number])
        why = []
        if together[number] > 0:
            why.append(f"you played them in the same session {times(int(together[number]))}")
        elif session[number] >= SAME_COMPANY:
            why.append("you play it with the same songs")
        if shared:
            why.append(f"shares {', '.join(shared)}")
        why.append(f"you played it {times(plays)}")
        result.append({
            "track_id": song.track_id,
            "title": song.track_name,
            "artist": song.artist_name,
            "genres": ", ".join(song_genres),
            "similarity": similarity[number],
            "session_similarity": session[number],
            "genre_similarity": genre[number],
            "plays": plays,
            "why": "; ".join(why),
            "song_key": keys.iloc[number],
            "artist_key": song.artist_key,
        })
    return pd.DataFrame(result, columns=SIMILAR_COLUMNS)


def print_list(table: pd.DataFrame, value: str) -> None:
    """Two or three lines per song, which line up even with Japanese or Korean titles."""
    for rank, row in enumerate(table.itertuples(), 1):
        genres = f"  [{row.genres}]" if row.genres else ""
        number = getattr(row, value)
        shown = f"{number:.0%}" if value in ("chance", "liked_chance") else f"{number:.2f}"
        print(f"{rank:>3}. {row.title} - {row.artist}{genres}  {value.replace('_', ' ')} {shown}")
        print(f"     {row.why}")
        if "link" in table.columns:
            print(f"     {row.link}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=10, help="how many songs (default 10)")
    parser.add_argument("--familiarity", type=float, help=f"0 to 1 (default {FAMILIARITY})")
    parser.add_argument("--new-artists", action="store_true", help="only artists you have never played")
    parser.add_argument("--rediscover", action="store_true", help="songs from your history instead of new ones")
    parser.add_argument("--dormant-days", type=float, default=DORMANT_DAYS, help=f"with --rediscover; default {DORMANT_DAYS}")
    parser.add_argument("--formula", action="store_true", help="with --rediscover: the hand-set formula, not the learned model")
    parser.add_argument(
        "--variety", type=float,
        help=f"0 to 1: trade score for genres the list lacks (default {VARIETY['discover']:g}, with --rediscover {VARIETY['rediscover']:g})",
    )
    parser.add_argument("--like", metavar="SONG", help='songs like this one instead: a title, or "artist - title"')
    parser.add_argument("--same-artist", action="store_true", help="with --like: include the same artist's songs")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # song names in any script, also when the output goes to a file
    familiarity = FAMILIARITY if args.familiarity is None else args.familiarity

    try:
        if args.like:
            table = similar_songs(args.like, args.top, args.same_artist)
            print_list(table, "similarity")
            offer_rating(table, "like")
            return
        mode = "rediscover" if args.rediscover else "discover"
        ranker = "formula" if args.formula else "learned"
        if mode == "rediscover" and ranker == "learned" and args.familiarity is not None:
            print("Note: --familiarity changes discover and the hand-set formula (--formula); the learned model has its own weights")
        variety = VARIETY[mode] if args.variety is None else args.variety
        table = recommend_songs(args.top, mode, familiarity, args.new_artists, args.dormant_days, ranker, variety)
        learned = mode == "rediscover" and table.chance.notna().any()
        if learned:
            print(
                f"Songs to rediscover, by the model learned from your history (unplayed for {args.dormant_days:g} days, "
                f"variety {variety:g}):"
            )
        elif len(table) and mode == "rediscover":
            print(
                f"Songs to rediscover, by the hand-set formula (familiarity {familiarity:g}, unplayed for "
                f"{args.dormant_days:g} days, variety {variety:g}):"
            )
        elif len(table):
            artists = "artists new to you" if args.new_artists else "artists new to you and ones you know"
            by_model = ", in order of the model learned from your ratings" if table.liked_chance.notna().any() else ""
            print(f"Songs new to you, by {artists} (familiarity {familiarity:g}, variety {variety:g}{by_model}):")
        value = "chance" if learned else "liked_chance" if mode == "discover" and table.liked_chance.notna().any() else "score"
        print_list(table, value)
        offer_rating(table, mode)
    except (FileNotFoundError, RuntimeError, LookupError, ValueError, NoProfile) as error:
        sys.exit(str(error))


def offer_rating(table: pd.DataFrame, mode: str) -> None:
    """Keep the printed list for rating by number, and say how; a note instead if it can't be saved."""
    if table.empty:
        return
    try:
        remember_list(table, mode)
    except RuntimeError as error:
        print(f"Note: the list wasn't kept, so it can't be rated by number: {error}")
        return
    print("Rate them: python -m recommender.feedback like 1 3, or dislike 2 (numbers of this list)")


if __name__ == "__main__":
    main()
