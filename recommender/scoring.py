"""How songs are scored: the candidates of each mode, and the score that ranks them.

recommend.py turns these scores into lists with explanations, evaluate.py tests
them on listening they haven't seen, and ranker.py learns a score for songs to
rediscover from your history instead of using fixed weights.

Every song gets a score from parts that each run from -1 to 1:

    song_match     how much you played the song itself lately: its weight in
                   the profile divided by your top song's, 0 for a new song
    genre_match    the cosine between the song's genre row and your genre
                   weights: how close its genres are to the mix you play lately
    artist_match   your profile's weight for the song's artist, 0 for an
                   artist you don't know
    similar_match  how strongly the artists you like point to the song's
                   artist on Last.fm (discover only): the sum, over your
                   artists whose Last.fm similar artists include it, of your
                   weight for them times Last.fm's match, divided by the
                   largest such sum in the pool
    popularity     the share of the candidates with fewer Last.fm listeners
                   than the song, 0 to 1 (discover only)
    genre_pick     how much the song is in the genres you asked for more of
                   (discover only): 1 if one of them is its main genre, 1/2 if
                   one is its second, 1/3 its third and so on (the best of
                   them), 0 if it has none of them
    western        1 for a song whose artist doesn't make Asian music
                   (candidates.csv asian), 0 otherwise (discover only)

    rediscover:  taste = (1 - familiarity) * genre_match + familiarity * artist_match
                 score = SONG_SHARE * song_match + (1 - SONG_SHARE) * taste
    discover:    taste = (1 - familiarity) * (genre_match + similar_match) / 2 + familiarity * artist_match
                 score = taste + popularity_weight * popularity + genre_boost * genre_pick - less_western * western

So familiarity moves the weight from "sounds like what you play" to "by an
artist you play": 0 treats new and known artists alike, 1 ranks by artist
alone. In discover, an artist you skip more than you play (artist_match below
0) counts in full whatever the familiarity, which only sets how much it helps
to be by an artist you like. A song without genres gets the median
genre_match of the other candidates, so that missing data neither helps nor
hurts it.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from recommender.config import CONFIG
from recommender.data import Library, candidate_pool
from recommender.features import cosine_to, genre_list, genre_matrix, genre_vocabulary
from recommender.profile import TasteProfile, song_evidence

# Weight of artist_match in the score, from 0 to 1 (see above).
FAMILIARITY = 0.5
# Weight of song_match, your history with the song itself, against taste.
# Chosen on the valid window of evaluate.py: adding it ranked dormant songs
# you went back to clearly better than taste alone.
SONG_SHARE = 0.5
# Weight of popularity in the discover score (the website's Mainstream). First
# chosen on the valid pool by the best AUC among 0, 0.25 and 0.5, which was 0.5,
# then set to 0 for less mainstream lists. On the author's data, the top 100 at
# 0 held more of the songs later played in both windows (3 against 1 in valid,
# 39 against 17 in test), at the cost of AUC (0.85 to 0.72, 0.94 to 0.90) and 2
# of the 5 hits in the test top 10 (docs/results.md). AUC also counts the order
# far down the pool.
POPULARITY_WEIGHT = 0.0
# Genres you want more of, as candidates.csv names them, and how much genre_pick adds ([taste] in
# config.toml; the website lets you pick any genre of the pool). A preference, not a measured choice, so
# evaluate.py leaves it at 0.
MORE_GENRES = tuple(CONFIG["taste"]["more_genres"])
GENRE_BOOST = CONFIG["taste"]["genre_boost"]
# How much a song by a Western (not Asian) artist is pushed down, famous or not ([taste] less_western in
# config.toml; the website's Less Western). A preference, not a measured choice, so evaluate.py leaves it
# at 0. On the author's data, values up to 0.5 kept the same finds in the offline top 10 and top 100; only
# AUC fell a little.
WESTERN = CONFIG["taste"]["less_western"]
# A song to rediscover is one you haven't played for this many days...
DORMANT_DAYS = 90
# ...and gave at least this many listens (3 = three full plays, or six halves).
LIKED_LISTENS = 3.0


def check_familiarity(familiarity: float) -> None:
    if not 0 <= familiarity <= 1:
        raise ValueError(f"familiarity must be between 0 and 1, not {familiarity}")


def taste_parts(
    profile: TasteProfile, genre_lists: pd.Series, artist_keys: pd.Series, vocabulary: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """genre_match and artist_match for songs with these genres and artists (artist_key() names)."""
    profile_vector = profile.genres.reindex(list(vocabulary)).fillna(0.0).to_numpy()
    genre_match = cosine_to(genre_matrix(genre_lists.tolist(), vocabulary), profile_vector)
    has_genres = genre_lists.map(lambda g: len(genre_list(g)) > 0).to_numpy()
    if has_genres.any():
        genre_match[~has_genres] = np.median(genre_match[has_genres])
    artist_match = profile.artists.weight.reindex(artist_keys).fillna(0.0).to_numpy()
    return genre_match, artist_match


def score_songs(
    profile: TasteProfile,
    library: Library,
    songs: np.ndarray,
    familiarity: float = FAMILIARITY,
    evidence: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The rediscover score and its parts for the given songs (numbers of rows in library.songs), in that order.

    evidence is song_evidence() as of profile.as_of, if you have it already.
    """
    check_familiarity(familiarity)
    if evidence is None:
        evidence = song_evidence(library, profile.as_of)
    song_match = (evidence.weight / evidence.weight.max()).to_numpy()[songs]
    vocabulary = genre_vocabulary(library.songs.genres)
    genre_match, artist_match = taste_parts(
        profile, library.songs.genres.iloc[songs], library.songs.artist_key.iloc[songs], vocabulary
    )
    taste = (1 - familiarity) * genre_match + familiarity * artist_match
    return pd.DataFrame({
        "song": songs,
        "song_match": song_match,
        "genre_match": genre_match,
        "artist_match": artist_match,
        "score": SONG_SHARE * song_match + (1 - SONG_SHARE) * taste,
    })


def similar_match(profile: TasteProfile, similar_to: pd.Series) -> np.ndarray:
    """For each {seed artist: Last.fm match} in similar_to: the sum of your weight for the seed times the match,
    over the seeds you like, divided by the largest such sum."""
    liked = profile.artists.weight.clip(lower=0)
    sums = similar_to.map(
        lambda links: sum(liked.get(seed.strip(), 0.0) * match for seed, match in links.items())
    ).to_numpy(dtype=float)
    top = sums.max() if len(sums) else 0.0
    return sums / top if top > 0 else sums


def genre_pick(genre_lists: pd.Series, genres: Sequence[str]) -> np.ndarray:
    """For each song, 1 / the place of the first of genres among its genres (1 for its main genre), 0 for none."""
    wanted = set(genres)
    return np.array([
        next((1.0 / place for place, genre in enumerate(genre_list(song_genres), 1) if genre in wanted), 0.0)
        for song_genres in genre_lists
    ])


def score_candidates(
    profile: TasteProfile, library: Library, songs: pd.DataFrame, familiarity: float = FAMILIARITY,
    popularity_weight: float = POPULARITY_WEIGHT, genres: Sequence[str] = (), genre_boost: float = 0.0,
    less_western: float = 0.0,
) -> pd.DataFrame:
    """The discover score and its parts for songs from a pool of candidates.csv, in the order of songs.

    genre_boost (for genres) and less_western are 0 unless asked for: recommend.py passes GENRE_BOOST and
    WESTERN, evaluate.py leaves them out.
    """
    check_familiarity(familiarity)
    # Genres only the candidates have get columns too, with weight 0 in your profile:
    # a song that is half country counts as half unlike you, not as all pop.
    vocabulary = genre_vocabulary(pd.concat([library.songs.genres, songs.genres], ignore_index=True))
    genre_match, artist_match = taste_parts(profile, songs.genres, songs.artist_key, vocabulary)
    similar = similar_match(profile, songs.similar_to)
    popularity = songs.listeners.rank(pct=True).to_numpy()
    picked = genre_pick(songs.genres, genres)
    western = (~songs.asian.fillna(True).astype(bool)).to_numpy(dtype=float)
    # A dislike doesn't depend on how adventurous you feel: negative artist_match always counts in full.
    familiar = familiarity * np.maximum(artist_match, 0) + np.minimum(artist_match, 0)
    taste = (1 - familiarity) * (genre_match + similar) / 2 + familiar
    return pd.DataFrame({
        "genre_match": genre_match,
        "artist_match": artist_match,
        "similar_match": similar,
        "popularity": popularity,
        "genre_pick": picked,
        "western": western,
        "score": taste + popularity_weight * popularity + genre_boost * picked - less_western * western,
    })


def rediscovery_candidates(
    library: Library, as_of: pd.Timestamp, dormant_days: float = DORMANT_DAYS, evidence: pd.DataFrame | None = None
) -> pd.DataFrame:
    """The song_evidence() rows of songs with at least LIKED_LISTENS listens before as_of but none in the dormant_days before it."""
    if evidence is None:
        evidence = song_evidence(library, as_of)
    dormant = evidence.last_played_at < as_of - pd.Timedelta(dormant_days, unit="D")
    keep = dormant & (evidence.listens >= LIKED_LISTENS) & library.songs.is_music.to_numpy()
    return evidence[keep]


def discovery_candidates(library: Library, as_of: pd.Timestamp, pool_name: str = "now") -> pd.DataFrame:
    """The songs of a pool of candidates.csv that you hadn't played before as_of."""
    songs = candidate_pool(library, pool_name)
    return songs[~(songs.first_played_at < as_of)].reset_index(drop=True)
