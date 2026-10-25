"""Build your taste profile from your listening, save it, and load it again.

    python -m recommender.profile [--as-of YYYY-MM-DD]

The profile is a weight for every genre and every artist you listen to: how
much you like them lately. recommend.py compares songs with it. It is built in
three steps.

1. Each play is a "listen" (gold.plays.listen): 1 if you finished it, 0 if you
   skipped it within 30 seconds, otherwise the share of the song you heard.
   An early skip also counts against the song, as SKIP_PENALTY listens.
2. Each song gets its evidence: those listens and skips, each counting half as
   much for every HALF_LIFE_DAYS of age, plus LIKE_BONUS if it is in your
   Liked Songs and PLAYLIST_BONUS for each playlist you added it to. Its
   weight is the log of that, keeping the sign, so heavy rotation doesn't
   drown out everything else: a song played 300 times counts about 2.5 times
   as much as one played 10 times, not 30 times.
   Your ratings of recommendations (feedback.py) add RATING_LISTENS listens
   for a like and take as many away for a dislike, fading with age like
   plays. A rated song you never played counts as a song of its own, with the
   genres and artist it was shown with.
3. The genre weights are the average of the songs' genre rows (features.py),
   each counted with its song's weight. An artist's weight is the log of all
   the evidence for their songs, divided by your top artist's, so it runs from
   1 (your top artist) down to -1 for an artist you skip far more than you play.

Tracks that aren't music to you (gold.tracks.is_music, from [not_music] in
config.toml) are left out: white noise or recitations are a different kind of
listening, and could otherwise be one of your largest "genres".

Without --as-of the profile describes the end of your listening history, with
all your ratings, also those given since, and is saved in
data/user/recommender.duckdb, replacing the one there. With --as-of it only
uses what you played, liked and rated before that date, and is shown but not
saved: this is how evaluate.py tests the recommender on later listening, and
how you can see what your taste was like in, say, 2021.
"""

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import duckdb
import numpy as np
import pandas as pd

if not __package__:
    # Started as a file (VS Code's Run button, or python recommender/profile.py) instead of with
    # python -m: Python then looks for imports in recommender/, so add the project folder first.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recommender.data import OPINIONS, RECOMMENDER_DB, SPOTIFY_DB, Library, current_ratings, load_library, song_key
from recommender.features import genre_list, genre_matrix, genre_vocabulary

# A play counts half as much for every this many days between it and the
# moment the profile describes, so the profile follows your taste as it moves.
HALF_LIFE_DAYS = 180
# An early skip counts as this many listens against the song. Skips are a
# weaker signal than listens: you may just not be in the mood for it.
SKIP_PENALTY = 0.5
# Explicit feedback, in listens: a song in your Liked Songs, and each playlist
# you added it to. Only the technical log has these, for the months it covers.
LIKE_BONUS = 5.0
PLAYLIST_BONUS = 3.0
# Your rating of a recommendation, in listens: plus for a like, minus for a dislike. As much as a song
# in Liked Songs. Chosen on the author's data by a rule fixed beforehand: of 5, 20 and 50, the one with
# which evaluate.py's feedback replay found the most songs later played in the valid window, the smallest
# if tied. All three tied (none in lists 2 to 5), so 5; on the test window 20 and 50 found one more of 40
# (docs/results.md).
RATING_LISTENS = 5.0

INFO_COLUMNS = {
    "as_of": "The moment the profile describes: plays and feedback from after it were not used, nor ratings for a past moment.",
    "built_at": "When the profile was built.",
    "songs": "Songs the profile was built from: played before as_of, or rated (tracks that aren't music left out).",
    "plays": "Their plays.",
    "ratings": "Your ratings of recommendations that the profile used.",
    "rating_listens": "RATING_LISTENS when it was built.",
    "half_life_days": "HALF_LIFE_DAYS when it was built.",
    "skip_penalty": "SKIP_PENALTY when it was built.",
    "like_bonus": "LIKE_BONUS when it was built.",
    "playlist_bonus": "PLAYLIST_BONUS when it was built.",
}
TABLE_COMMENTS = {
    "profile_genres": "Your taste profile: a weight per genre. Built by python -m recommender.profile.",
    "profile_artists": "Your taste profile: a weight per artist, -1 to 1. Built by python -m recommender.profile.",
    "profile_info": "When and how the taste profile in profile_genres and profile_artists was built. One row.",
}
COLUMN_COMMENTS = {
    "profile_genres": {
        "genre": "Last.fm genre, as in gold.track_genres.",
        "weight": "Average genre row of your songs, each counted with its song's weight. Negative if you mostly skip it.",
    },
    "profile_artists": {
        "artist_key": "The artist's name as Spotify writes it: artists are told apart by name only, letter case included.",
        "artist_name": "The artist's name as Spotify writes it.",
        "weight": "Log of the evidence for the artist's songs, divided by your top artist's: 1 for your top artist.",
    },
    "profile_info": INFO_COLUMNS,
}


class NoProfile(Exception):
    pass


class TasteProfile(NamedTuple):
    as_of: pd.Timestamp
    built_at: pd.Timestamp
    genres: pd.Series  # weight by genre, largest first
    artists: pd.DataFrame  # by artist_key: artist_name, weight (-1 to 1), largest first
    songs: int
    plays: int
    ratings: int = 0


def sql_text(text: str) -> str:
    """text as an SQL string literal."""
    return "'" + text.replace("'", "''") + "'"


def as_timestamp(value, library: Library) -> pd.Timestamp:
    """A date or time; one without a time zone is taken to be in the time zone of the database's timestamps."""
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(library.ends_at.tz) if stamp.tzinfo is None else stamp


def song_evidence(library: Library, as_of: pd.Timestamp) -> pd.DataFrame:
    """What your plays and feedback before as_of say about each song: one row per song in library.songs."""
    plays = library.plays[library.plays.started_at < as_of]
    decay = 0.5 ** ((as_of - plays.started_at).dt.total_seconds() / 86400 / HALF_LIFE_DAYS)
    skips = plays.skipped_early.astype(float)
    per_song = pd.DataFrame({
        "song": plays.song,
        "plays": 1,
        "finished": plays.finished.astype(int),
        "listens": plays.listen,
        "early_skips": skips,
        "recent_listens": plays.listen * decay,
        "recent_skips": skips * decay,
        "last_played_at": plays.started_at,
    }).groupby("song").agg({
        "plays": "sum", "finished": "sum", "listens": "sum", "early_skips": "sum",
        "recent_listens": "sum", "recent_skips": "sum", "last_played_at": "max",
    })

    feedback = library.feedback[library.feedback.ts < as_of]
    likes = feedback[feedback.event.isin(["like", "unlike"])].groupby("song").event.last() == "like"
    playlist_moves = feedback.event.map({"playlist_add": 1, "playlist_remove": -1}).fillna(0)
    playlists = playlist_moves.groupby(feedback.song).sum().clip(lower=0)

    evidence = per_song.reindex(range(len(library.songs)))
    evidence[["plays", "finished", "listens", "early_skips", "recent_listens", "recent_skips"]] = (
        evidence[["plays", "finished", "listens", "early_skips", "recent_listens", "recent_skips"]].fillna(0)
    )
    evidence["liked"] = likes.reindex(evidence.index, fill_value=False).astype(bool)
    evidence["playlists"] = playlists.reindex(evidence.index, fill_value=0)
    evidence["evidence"] = (
        evidence.recent_listens
        - SKIP_PENALTY * evidence.recent_skips
        + LIKE_BONUS * evidence.liked
        + PLAYLIST_BONUS * evidence.playlists
    )
    evidence.loc[~library.songs.is_music.values, "evidence"] = 0.0
    evidence["weight"] = log_weight(evidence.evidence)
    return evidence


def log_weight(evidence: pd.Series) -> pd.Series:
    """The log of evidence, keeping its sign."""
    return np.sign(evidence) * np.log1p(evidence.abs())


def rating_evidence(
    library: Library, as_of: pd.Timestamp, before: pd.Timestamp | None, listens: float = RATING_LISTENS
) -> tuple[pd.Series, pd.DataFrame]:
    """What your ratings (all, or those before `before`) say as of as_of, at `listens` listens each: for songs of
    your history (index: song number), and for rated songs you never played (genres, artist_key, artist_name,
    evidence). A rating fades like a play; one given after as_of counts in full. Neutral ratings say nothing."""
    rated = current_ratings(library.ratings, before=before)
    rated = rated[rated.rating.isin(OPINIONS)]
    external = pd.DataFrame(columns=["genres", "artist_key", "artist_name", "evidence"])
    if rated.empty:
        return pd.Series(dtype=float), external
    age_days = ((as_of - pd.to_datetime(rated.rated_at, utc=True)).dt.total_seconds() / 86400).clip(lower=0)
    value = rated.rating.astype(float).to_numpy() * listens * 0.5 ** (age_days.to_numpy() / HALF_LIFE_DAYS)
    numbers = pd.Series(np.arange(len(library.songs)), index=song_key(library.songs.artist_name, library.songs.track_name).to_numpy())
    known = rated.song_key.isin(numbers.index).to_numpy()
    in_history = pd.Series(value[known], index=numbers[rated.song_key[known]].to_numpy())
    external = rated.loc[~known, ["genres", "artist_key", "artist_name"]].assign(evidence=value[~known]).reset_index(drop=True)
    return in_history, external


def build_profile(
    as_of=None, library: Library | None = None, save: bool | None = None, rating_listens: float = RATING_LISTENS
) -> TasteProfile:
    """Your taste profile from what you played, liked and rated before as_of (default: all of it).

    Without as_of it describes the end of your history and uses all your ratings, also the ones given since;
    it is then saved, replacing the saved one, unless save=False. rating_listens is what one rating counts
    (evaluate.py tries others).
    """
    library = library or load_library()
    if as_of is None:
        as_of, before = library.ends_at, None
        save = True if save is None else save
    else:
        as_of = before = as_timestamp(as_of, library)
    evidence = song_evidence(library, as_of)
    in_history, external = rating_evidence(library, as_of, before, rating_listens)
    if len(in_history):
        evidence["evidence"] = evidence.evidence.add(in_history.groupby(level=0).sum(), fill_value=0.0)
        evidence.loc[~library.songs.is_music.values, "evidence"] = 0.0
        evidence["weight"] = log_weight(evidence.evidence)
    used = evidence.weight != 0
    if not used.any() and external.empty:
        raise NoProfile(f"No music plays before {as_of:%d %b %Y}, so there is nothing to learn your taste from")

    # Your songs, then the rated songs you never played.
    genres = pd.concat([library.songs.genres, external.genres], ignore_index=True)
    song_evidences = pd.concat([evidence.evidence, external.evidence.astype(float)], ignore_index=True)
    artist_keys = pd.concat([library.songs.artist_key, external.artist_key], ignore_index=True)
    artist_names = pd.concat([library.songs.artist_name, external.artist_name], ignore_index=True)
    weights = log_weight(song_evidences).where(genres.map(lambda g: len(genre_list(g)) > 0), 0.0).to_numpy()
    vocabulary = genre_vocabulary(genres)
    genre_weights = genre_matrix(genres.tolist(), vocabulary).T @ weights / max(np.abs(weights).sum(), 1e-12)
    genre_series = pd.Series(genre_weights, index=list(vocabulary)).loc[lambda s: s != 0].sort_values(ascending=False)

    artist_evidence = song_evidences.groupby(artist_keys).sum().loc[lambda s: s != 0]
    artist_weights = log_weight(artist_evidence)
    artists = pd.DataFrame({
        "artist_name": artist_names.groupby(artist_keys).first().reindex(artist_weights.index),
        "weight": artist_weights / artist_weights.abs().max(),
    }).sort_values("weight", ascending=False)

    profile = TasteProfile(
        as_of=as_of,
        built_at=pd.Timestamp.now(tz=library.ends_at.tz),
        genres=genre_series,
        artists=artists,
        songs=int(used.sum()) + int((external.evidence != 0).sum()),
        plays=int(evidence.plays[used].sum()),
        ratings=len(in_history) + len(external),
    )
    if save:
        save_profile(profile)
    return profile


def save_profile(profile: TasteProfile, db: Path = RECOMMENDER_DB) -> None:
    db.parent.mkdir(parents=True, exist_ok=True)
    genres = profile.genres.rename_axis("genre").rename("weight").reset_index()
    artists = profile.artists.rename_axis("artist_key").reset_index()
    info = pd.DataFrame([{
        "as_of": profile.as_of, "built_at": profile.built_at, "songs": profile.songs, "plays": profile.plays,
        "ratings": profile.ratings, "rating_listens": RATING_LISTENS, "half_life_days": HALF_LIFE_DAYS,
        "skip_penalty": SKIP_PENALTY, "like_bonus": LIKE_BONUS, "playlist_bonus": PLAYLIST_BONUS,
    }])
    try:
        con = duckdb.connect(str(db))
    except duckdb.IOException as error:
        raise RuntimeError(f"Could not open {db} to save the profile: {error}. Close other programs using it.") from error
    try:
        con.begin()
        for table, frame in (("profile_genres", genres), ("profile_artists", artists), ("profile_info", info)):
            con.register("frame", frame)
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM frame")
            con.unregister("frame")
            con.execute(f"COMMENT ON TABLE {table} IS {sql_text(TABLE_COMMENTS[table])}")
            for column, text in COLUMN_COMMENTS[table].items():
                con.execute(f"COMMENT ON COLUMN {table}.{column} IS {sql_text(text)}")
        con.commit()
    finally:
        con.close()


def load_profile(db: Path = RECOMMENDER_DB) -> TasteProfile:
    """The saved profile. Raises NoProfile if there is none yet."""
    hint = "(run python -m recommender.profile to build it)"
    if not db.exists():
        raise NoProfile(f"No saved taste profile: {db} doesn't exist yet {hint}")
    con = duckdb.connect(str(db), read_only=True)
    try:
        info = con.sql("SELECT * FROM profile_info").df().iloc[0]
        genres = con.sql("SELECT genre, weight FROM profile_genres ORDER BY weight DESC").df()
        artists = con.sql("SELECT artist_key, artist_name, weight FROM profile_artists ORDER BY weight DESC").df()
    except duckdb.CatalogException as error:
        raise NoProfile(f"No saved taste profile in {db.name} {hint}") from error
    finally:
        con.close()
    return TasteProfile(
        as_of=info.as_of,
        built_at=info.built_at,
        genres=genres.set_index("genre").weight,
        artists=artists.set_index("artist_key"),
        songs=int(info.songs),
        plays=int(info.plays),
        ratings=int(info.get("ratings", 0)),  # profiles saved before phase 5 have no ratings column
    )


def describe(profile: TasteProfile, top: int = 15) -> str:
    """The profile in words: your top genres and artists, and the artists you skip most."""
    liked_genres = profile.genres[profile.genres > 0]
    ratings = f" and {profile.ratings:,} of your ratings" if profile.ratings else ""
    lines = [
        f"Taste profile as of {profile.as_of:%d %b %Y}, from {profile.songs:,} songs, {profile.plays:,} plays{ratings} "
        f"(music only)",
        "  Top genres, as a share of all positive genre weight:",
        *(f"    {share:>4.0%}  {genre}" for genre, share in (liked_genres / liked_genres.sum()).head(top).items()),
        "  Top artists (1 = your top artist):",
        *(f"    {row.weight:5.2f}  {row.artist_name}" for row in profile.artists.head(top).itertuples()),
    ]
    disliked = profile.artists[profile.artists.weight < 0].sort_values("weight").head(5)
    if len(disliked):
        lines.append("  Artists you skip more than you play:")
        lines += [f"    {row.weight:5.2f}  {row.artist_name}" for row in disliked.itertuples()]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", help="only use plays and likes before this date, and don't save (YYYY-MM-DD)")
    parser.add_argument("--db", type=Path, default=SPOTIFY_DB, help="spotify.duckdb built by scripts/build_gold.py")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # song names in any script, also when the output goes to a file

    try:
        library = load_library(args.db)
        profile = build_profile(args.as_of, library)
    except (FileNotFoundError, RuntimeError, NoProfile) as error:
        sys.exit(str(error))
    print(describe(profile))
    if args.as_of:
        print("Note: a profile as of a past date is shown, not saved")
    else:
        print(f"Saved to {RECOMMENDER_DB}")


if __name__ == "__main__":
    main()
