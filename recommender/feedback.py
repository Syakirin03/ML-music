"""Your ratings of recommendations: like or dislike the songs you are shown, and what is learned from them.

    python -m recommender.feedback                       your ratings so far, and the model learned from them
    python -m recommender.feedback --check               also how well it predicts your later ratings from earlier ones
    python -m recommender.feedback like 1 3              like songs 1 and 3 of the last list you were shown
    python -m recommender.feedback dislike 2 "word"      dislike song 2, and the newest shown song matching "word"
    python -m recommender.feedback neutral 4             song 4 was fine, no opinion: it leaves your lists, nothing else
    python -m recommender.feedback clear 2               take back your rating of song 2
    python -m recommender.feedback --restore             put your ratings back from ratings.csv into an empty database

Every list that python -m recommender.recommend prints is kept (the table
shown), so a number means that song of the last list, and words find the
newest song you were shown whose "title - artist" contains them, ignoring case.

What a rating does:

1. The song isn't recommended again, and a song you disliked isn't offered as
   one like another (--like) either. A neutral rating does only this: it says
   nothing about your taste, so steps 2 and 3 leave it out.
2. It counts in your taste profile (profile.py) as RATING_LISTENS listens of
   the song, or minus that many for a dislike, fading with age like plays.
   The genres and the artist of a song you liked weigh more, and those of one
   you disliked less, also for songs you never played. The next
   recommendation rebuilds the profile with it.
3. Once you have rated at least MIN_EACH songs new to you up and MIN_EACH down,
   a logistic regression learns from those ratings what says whether you like
   a song: MODEL_PARTS of the discover score as they were when you saw it,
   MODEL_FACTS about the song from the pool (Asian or not, its Last.fm
   listeners, its rank among its artist's songs), and its genres.
   It is cross-validated on your ratings: each rating is predicted by a model
   that didn't learn from it. If the model's AUC beats the AUC of the hand-set
   score on the same ratings by at least MODEL_MARGIN, discover orders its
   best songs by the model's chance that you like each. Your ratings are all of
   songs the score put near the top, so the model reorders those (the 300
   best, recommend.SHORTLIST) and nothing further down. --check predicts
   your ratings in time order instead, each block from the ratings before it,
   which is how the model is really used.

Nothing can rebuild your ratings, so the ratings table is only ever added to:
rating a song again, or taking a rating back, adds a row, and the newest row
of a song counts. After every rating, data/user/ratings.csv is rewritten with
all of them, as a backup; --restore reads it back into a database that has
lost them. The recommender itself never sends them anywhere.

Run it from the project folder with python -m recommender.feedback, or as a
file, for example with VS Code's Run button (which passes no options).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

if not __package__:
    # Started as a file (VS Code's Run button, or python recommender/feedback.py) instead of with
    # python -m: Python then looks for imports in recommender/, so add the project folder first.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recommender.data import NEUTRAL, OPINIONS, RATINGS_CSV, RECOMMENDER_DB, current_ratings, load_library
from recommender.features import genre_list, genre_matrix
from recommender.metrics import BOOTSTRAP_RUNS, auc
from recommender.profile import sql_text

RATINGS = {"like": 1, "dislike": -1, "neutral": NEUTRAL, "clear": 0}
# The learned model needs at least this many likes and this many dislikes of songs new to you...
MIN_EACH = 15
# ...and must beat the hand-set score's AUC on your ratings by at least this much, cross-validated, to be used.
# Fixed before there were any ratings.
MODEL_MARGIN = 0.05
# What the model learns from: these parts of the discover score as they were when you saw the song...
MODEL_PARTS = ["genre_match", "artist_match", "similar_match"]
# ...these facts about the song from the pool: Asian or not, log10 of its Last.fm listeners and of its rank
# among its artist's songs. Popularity (the share of the pool with fewer listeners) is left out: it shifts
# whenever the pool grows, so the value you saw no longer matches today's, while listeners say the same.
MODEL_FACTS = ["asian", "log_listeners", "log_rank"]
# ...and one column per genre that at least this many of your rated songs have (main genre 1, second 1/2, ...).
# Chosen by a rule fixed beforehand, on the author's 227 ratings (docs/results.md): of four models, this one
# predicted later ratings from earlier ones best.
MIN_GENRE_RATINGS = 5
# A song missing from the pool gets this rank among its artist's songs (half of TRACKS_PER_ARTIST).
MIDDLE_RANK = 25
# The forward check (--check) predicts your ratings after the first FORWARD_START, FORWARD_BLOCK at a time.
FORWARD_START = 80
FORWARD_BLOCK = 20
# The model's cross-validation is repeated with this many different splits of your ratings into 5 parts.
CV_REPEATS = 10
SEED = 0
# What is kept of a shown song's score: the parts, per list (the columns of recommend.py's tables).
PARTS = {
    "discover": ["genre_match", "artist_match", "similar_match", "popularity", "liked_chance"],
    "rediscover": ["chance", "song_match", "genre_match", "artist_match", "liked_chance"],
    "like": ["similarity", "session_similarity", "genre_similarity"],
}

SONG_COLUMNS = {
    "song_key": "The song: artist and title without what Spotify adds to it (recommender/data.py song_key).",
    "track_id": "Spotify id of the version you played most; NULL for a song from the Last.fm pool.",
    "artist_key": "The artist's name as the recommender knows it.",
    "artist_name": "The artist's name as shown.",
    "track_name": "The title as shown.",
    "genres": "The song's genres, most tagged first, separated by |.",
}
SHOWN_COLUMNS = {
    "list_id": "The list: 1 for the first ever shown, then counting up.",
    "shown_at": "When the list was printed.",
    "mode": "The kind of list: discover (songs new to you), rediscover, or like (songs like one you named).",
    "position": "The song's number in the list, from 1.",
    **SONG_COLUMNS,
    "score": "The song's score in the list (for like: its similarity).",
    "parts": "The score's parts when the song was shown, as JSON.",
}
COLUMN_COMMENTS = {
    "shown": SHOWN_COLUMNS,
    "ratings": {
        "rated_at": "When you rated the song.",
        "rating": (
            "1 like, -1 dislike, 2 neutral (heard, no opinion: the song leaves your lists but counts nowhere), "
            "0 a rating taken back. The newest row of a song counts."
        ),
        "mode": SHOWN_COLUMNS["mode"],
        **SONG_COLUMNS,
        "list_id": "The shown list the song was rated from.",
        "position": SHOWN_COLUMNS["position"],
        "score": SHOWN_COLUMNS["score"],
        "parts": SHOWN_COLUMNS["parts"],
    },
}
TABLE_COMMENTS = {
    "shown": "Every list of recommendations printed, song by song. Added to by python -m recommender.recommend.",
    "ratings": "Your ratings of recommendations, never changed, only added to. Nothing can rebuild them: back them up.",
}
TYPES = {
    "list_id": "INTEGER", "shown_at": "TIMESTAMPTZ", "rated_at": "TIMESTAMPTZ", "rating": "TINYINT", "mode": "VARCHAR",
    "position": "INTEGER", "score": "DOUBLE", "parts": "VARCHAR", **{column: "VARCHAR" for column in SONG_COLUMNS},
}


class FeedbackModel(NamedTuple):
    model: Pipeline | None  # None while it isn't used
    message: str  # why it is or isn't used
    ratings: int  # ratings of songs new to you it was judged on
    scaler: StandardScaler | None = None  # how its MODEL_PARTS and MODEL_FACTS are scaled
    genres: dict[str, int] | None = None  # its genre columns
    numbers: list[str] | None = None  # the names of its inputs before the genres


def connect(db: Path) -> duckdb.DuckDBPyConnection:
    """db opened for writing, with the shown and ratings tables there."""
    db.parent.mkdir(parents=True, exist_ok=True)
    try:
        con = duckdb.connect(str(db))
    except duckdb.IOException as error:
        raise RuntimeError(f"Could not open {db} to save: {error}. Close other programs using it.") from error
    for table, columns in COLUMN_COMMENTS.items():
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(f'{c} {TYPES[c]}' for c in columns)})")
        con.execute(f"COMMENT ON TABLE {table} IS {sql_text(TABLE_COMMENTS[table])}")
        for column, text in columns.items():
            con.execute(f"COMMENT ON COLUMN {table}.{column} IS {sql_text(text)}")
    return con


def insert(con: duckdb.DuckDBPyConnection, table: str, frame: pd.DataFrame) -> None:
    columns = list(COLUMN_COMMENTS[table])
    con.register("frame", frame[columns])
    con.execute(f"INSERT INTO {table} SELECT {', '.join(columns)} FROM frame")
    con.unregister("frame")


def remember_list(table: pd.DataFrame, mode: str, db: Path = RECOMMENDER_DB) -> int:
    """Keep a list of recommendations (a table of recommend.py) in the shown table; returns its list_id."""
    con = connect(db)
    try:
        list_id = int(con.sql("SELECT coalesce(max(list_id), 0) + 1 FROM shown").fetchone()[0])
        parts = [c for c in PARTS[mode] if c in table.columns]
        frame = pd.DataFrame({
            "list_id": list_id,
            "shown_at": pd.Timestamp.now(tz="UTC"),
            "mode": mode,
            "position": np.arange(1, len(table) + 1),
            "song_key": table.song_key.to_numpy(),
            "track_id": table.track_id.to_numpy() if "track_id" in table.columns else None,
            "artist_key": table.artist_key.to_numpy(),
            "artist_name": table.artist.to_numpy(),
            "track_name": table.title.to_numpy(),
            # No genre name has ", " in it, so the shown genres split back into the list.
            "genres": table.genres.map(lambda g: "|".join(g.split(", ")) if g else None).to_numpy(),
            "score": (table.similarity if mode == "like" else table.score).to_numpy(dtype=float),
            "parts": [json.dumps({c: none_if_nan(row[c]) for c in parts}) for _, row in table.iterrows()],
        })
        insert(con, "shown", frame)
        return list_id
    finally:
        con.close()


def none_if_nan(value):
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)


def find_shown(shown: pd.DataFrame, item: str, list_id: int | None = None) -> pd.Series:
    """The shown song an item names: a number of list list_id (default: the last list), or words in the newest
    shown "title - artist"."""
    if shown.empty:
        raise LookupError("No list has been shown yet: run python -m recommender.recommend first")
    if item.isdigit():
        last = shown[shown.list_id == (shown.list_id.max() if list_id is None else list_id)]
        match = last[last.position == int(item)]
        if match.empty:
            raise LookupError(f"The list has songs 1 to {last.position.max()}, not {item}")
        return match.iloc[0]
    text = (shown.track_name + " - " + shown.artist_name).str.casefold()
    match = shown[text.str.contains(item.casefold().strip(), regex=False)]
    if match.empty:
        raise LookupError(f"No song you were shown matches {item!r}")
    return match.sort_values(["list_id", "position"], ascending=[False, True]).iloc[0]


def rate(
    items: list[str], action: str, db: Path = RECOMMENDER_DB, csv: Path = RATINGS_CSV, list_id: int | None = None
) -> pd.DataFrame:
    """Rate the shown songs that items name (see find_shown) with action like, dislike, neutral or clear; returns the new rows."""
    con = connect(db)
    try:
        shown = con.sql("SELECT * FROM shown").df()
        rows = pd.DataFrame([find_shown(shown, item, list_id) for item in items])
        return add_ratings(con, rows, action, csv)
    finally:
        con.close()


def rate_rows(rows: pd.DataFrame, action: str, db: Path = RECOMMENDER_DB, csv: Path = RATINGS_CSV) -> pd.DataFrame:
    """Rate songs given as rows of the shown or ratings table (all their columns but rated_at and rating)."""
    con = connect(db)
    try:
        return add_ratings(con, rows, action, csv)
    finally:
        con.close()


def add_ratings(con: duckdb.DuckDBPyConnection, rows: pd.DataFrame, action: str, csv: Path) -> pd.DataFrame:
    rows = rows.drop_duplicates("song_key").assign(rated_at=pd.Timestamp.now(tz="UTC"), rating=RATINGS[action])
    if isinstance(rows.genres.iloc[0], list):  # rows of current_ratings() carry the genres as a list
        rows["genres"] = rows.genres.map(lambda g: "|".join(g) or None)
    con.begin()
    insert(con, "ratings", rows)
    con.commit()
    backup(con, csv)
    return rows


def backup(con: duckdb.DuckDBPyConnection, csv: Path) -> None:
    """Rewrite csv with every rating, via a temporary file so a crash never leaves half a backup."""
    columns = ", ".join(COLUMN_COMMENTS["ratings"])
    table = con.sql(f"SELECT {columns} FROM ratings ORDER BY rated_at, rowid").df()
    temporary = csv.with_suffix(".tmp")
    table.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(csv)


def restore(db: Path = RECOMMENDER_DB, csv: Path = RATINGS_CSV) -> int:
    """Read csv back into db's ratings table, if that is empty; returns the number of ratings restored."""
    if not csv.exists():
        raise FileNotFoundError(f"No backup to restore: {csv} doesn't exist")
    con = connect(db)
    try:
        if con.sql("SELECT count(*) FROM ratings").fetchone()[0]:
            raise RuntimeError(f"{db.name} still has ratings, so nothing was restored: the backup is only for an empty one")
        table = pd.read_csv(csv, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
        table["rated_at"] = pd.to_datetime(table.rated_at, utc=True)
        con.begin()
        insert(con, "ratings", table)
        con.commit()
        return len(table)
    finally:
        con.close()


def rated_songs(ratings: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Your current likes and dislikes of songs new to you, oldest first, and their rows of the now pool (NaN where
    missing). Neutral ratings teach the model nothing, so they are left out."""
    rated = current_ratings(ratings)
    rated = rated[(rated["mode"] == "discover") & rated.rating.isin(OPINIONS)].sort_values("rated_at").reset_index(drop=True)
    pool = candidates[candidates.pool == "now"].drop_duplicates("song_key").set_index("song_key")
    songs = pool.reindex(rated.song_key)[["asian", "listeners", "artist_rank", "genres"]].reset_index(drop=True)
    return rated, songs


def model_numbers(parts: pd.DataFrame, songs: pd.DataFrame) -> pd.DataFrame:
    """MODEL_PARTS (from parts) and MODEL_FACTS (from the pool rows songs, in the same order) of songs.

    A song missing from the pool counts as Asian, with the median listeners of the others and MIDDLE_RANK.
    """
    listeners = songs.listeners.astype(float)
    return pd.DataFrame({
        **{part: parts[part].astype(float).to_numpy() for part in MODEL_PARTS},
        "asian": songs.asian.astype(float).fillna(1.0).to_numpy(),
        "log_listeners": np.log10(listeners.fillna(listeners.median()) + 1).to_numpy(),
        "log_rank": np.log10(songs.artist_rank.astype(float).fillna(MIDDLE_RANK)).to_numpy(),
    })


def stored_parts(rows: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """The score parts names as kept with rated or shown songs (their parts column), one row per row of rows:
    with these columns even when there are no rows, as for someone who hasn't rated anything yet."""
    return pd.DataFrame([json.loads(parts) for parts in rows.parts], columns=names)


def genre_columns(genre_lists: pd.Series) -> dict[str, int]:
    """The genres of at least MIN_GENRE_RATINGS of these songs, each a column of the model."""
    counts = genre_lists.map(genre_list).explode().dropna().value_counts()
    return {genre: column for column, genre in enumerate(sorted(counts[counts >= MIN_GENRE_RATINGS].index))}


def model_rows(numbers: pd.DataFrame, genre_lists: pd.Series, scaler: StandardScaler, genres: dict[str, int]) -> sparse.csr_matrix:
    """What the model sees of each song: its numbers scaled by scaler, then its genre row."""
    return sparse.hstack([sparse.csr_matrix(scaler.transform(numbers)), genre_matrix(genre_lists.tolist(), genres)]).tocsr()


def fit_model(
    numbers: pd.DataFrame, genre_lists: pd.Series, liked: np.ndarray, genres: dict[str, int]
) -> tuple[StandardScaler, Pipeline]:
    """Logistic regression with the default penalty, on the numbers scaled to mean 0 and standard deviation 1
    and the genre rows, everything then scaled to at most 1."""
    scaler = StandardScaler().fit(numbers)
    model = make_pipeline(MaxAbsScaler(), LogisticRegression(C=1.0, max_iter=5000))
    return scaler, model.fit(model_rows(numbers, genre_lists, scaler, genres), liked)


def predict(
    numbers: pd.DataFrame, genre_lists: pd.Series, scaler: StandardScaler, model: Pipeline, genres: dict[str, int]
) -> np.ndarray:
    return model.predict_proba(model_rows(numbers, genre_lists, scaler, genres))[:, 1]


def learn_from_ratings(
    rated: pd.DataFrame, numbers: pd.DataFrame, genre_lists: pd.Series, songs: str, baseline: str
) -> FeedbackModel:
    """The model learned from rated (oldest first, with rating and score) and their inputs, if there are MIN_EACH
    likes and dislikes and it beats the score the songs were ranked by (baseline names it) by MODEL_MARGIN,
    cross-validated. songs says which ratings these are, for the message."""
    likes, dislikes = int((rated.rating == 1).sum()), int((rated.rating == -1).sum())
    if likes < MIN_EACH or dislikes < MIN_EACH:
        return FeedbackModel(
            None, f"not in use yet: {likes} likes and {dislikes} dislikes of {songs} so far, and it needs {MIN_EACH} of each",
            len(rated),
        )
    liked = (rated.rating == 1).to_numpy()
    genres = genre_columns(genre_lists)
    learned = []
    for repeat in range(CV_REPEATS):
        chances = np.zeros(len(rated))
        for train, test in StratifiedKFold(5, shuffle=True, random_state=SEED + repeat).split(numbers, liked):
            scaler, model = fit_model(numbers.iloc[train], genre_lists.iloc[train], liked[train], genres)
            chances[test] = predict(numbers.iloc[test], genre_lists.iloc[test], scaler, model, genres)
        learned.append(auc(liked, chances))
    learned = float(np.mean(learned))
    formula = auc(liked, rated.score.to_numpy(dtype=float))
    comparison = (
        f"on your {len(rated)} ratings of {songs}, it reaches AUC {learned:.2f} (cross-validated) "
        f"against {formula:.2f} for {baseline}"
    )
    if learned < formula + MODEL_MARGIN:
        return FeedbackModel(None, f"not in use: {comparison}, less than {MODEL_MARGIN:g} better", len(rated))
    scaler, model = fit_model(numbers, genre_lists, liked, genres)
    return FeedbackModel(model, f"in use: {comparison}", len(rated), scaler, genres, list(numbers.columns))


def forward_lines(
    rated: pd.DataFrame, numbers: pd.DataFrame, genre_lists: pd.Series, first: int, songs: str, baseline: str,
    rng: np.random.Generator,
) -> list[str]:
    """How well the model predicts later ratings from earlier ones: each block of FORWARD_BLOCK ratings after
    the first `first` is predicted by a model learned only from the ratings before it, genre columns included."""
    if len(rated) < first + FORWARD_BLOCK:
        return [f"Forward check: it needs {first + FORWARD_BLOCK} ratings of {songs}, and you have {len(rated)}"]
    liked = (rated.rating == 1).to_numpy()
    chances = np.full(len(rated), np.nan)
    for start in range(first, len(rated), FORWARD_BLOCK):
        train, test = np.arange(start), np.arange(start, min(start + FORWARD_BLOCK, len(rated)))
        genres = genre_columns(genre_lists.iloc[train])
        scaler, model = fit_model(numbers.iloc[train], genre_lists.iloc[train], liked[train], genres)
        chances[test] = predict(numbers.iloc[test], genre_lists.iloc[test], scaler, model, genres)
    later = np.arange(first, len(rated))
    y, learned, score = liked[later], chances[later], rated.score.to_numpy(dtype=float)[later]
    differences = []
    for _ in range(BOOTSTRAP_RUNS):
        pick = rng.integers(0, len(later), len(later))
        if 0 < y[pick].sum() < len(pick):
            differences.append(auc(y[pick], learned[pick]) - auc(y[pick], score[pick]))
    low, high = np.percentile(differences, [2.5, 97.5])
    return [
        f"Forward check: your ratings {first + 1} to {len(rated)} of {songs} ({len(later)}, {int(y.sum())} likes), "
        f"each block of {FORWARD_BLOCK} predicted by a model learned only from the ratings before it:",
        f"    the model{'':<{max(len(baseline) - 9, 0)}}  AUC {auc(y, learned):.2f}",
        f"    {baseline}  AUC {auc(y, score):.2f}",
        f"    difference {auc(y, learned) - auc(y, score):+.2f} (95% range {low:+.2f} to {high:+.2f})",
    ]


def feedback_model(ratings: pd.DataFrame, candidates: pd.DataFrame) -> FeedbackModel:
    """The model learned from your ratings of songs new to you, if it has enough of them and beats the score (docstring).

    candidates is Library.candidates: the now pool gives each rated song its MODEL_FACTS and genres.
    """
    rated, songs = rated_songs(ratings, candidates)
    numbers = model_numbers(stored_parts(rated, MODEL_PARTS), songs)
    return learn_from_ratings(rated, numbers, songs.genres, "songs new to you", "the hand-set score")


def like_chances(model: FeedbackModel, numbers: pd.DataFrame, genre_lists: pd.Series) -> np.ndarray:
    """The model's chance that you like each song, from its inputs (model_numbers(), or for songs to rediscover
    ranker.rating_numbers()) and its genres."""
    return predict(numbers, genre_lists, model.scaler, model.model, model.genres)


def model_weights(model: FeedbackModel) -> pd.Series:
    """What the model learned, largest first: for a number, how much one standard deviation more changes the
    log-odds of a like; for a genre, how much it changes them for the rated song with the most of that genre."""
    names = model.numbers + [f"genre: {genre}" for genre in model.genres]
    coefficients = model.model[-1].coef_[0].copy()  # per largest value of each input (MaxAbsScaler)
    coefficients[:len(model.numbers)] /= model.model[0].scale_[:len(model.numbers)]
    weights = pd.Series(coefficients, index=names)
    return weights.reindex(weights.abs().sort_values(ascending=False).index)


def weight_lines(model: FeedbackModel, top: int = 12) -> list[str]:
    """What the model learned, as lines to print (nothing while it isn't in use)."""
    if model.model is None:
        return []
    weights = model_weights(model).head(top)
    return [
        "    what it learned (change in log-odds of a like: per standard deviation, or for the song with most of a genre):",
        "    " + ", ".join(f"{name} {weight:+.2f}" for name, weight in weights.items()),
    ]


def forward_check(ratings: pd.DataFrame, candidates: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    """forward_lines() for your ratings of songs new to you, after the first FORWARD_START."""
    rated, songs = rated_songs(ratings, candidates)
    numbers = model_numbers(stored_parts(rated, MODEL_PARTS), songs)
    return forward_lines(rated, numbers, songs.genres, FORWARD_START, "songs new to you", "the hand-set score", rng)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", nargs="?", choices=list(RATINGS), help="like, dislike, neutral or clear")
    parser.add_argument("items", nargs="*", help="numbers in the last list, or words of a title or artist you were shown")
    parser.add_argument("--restore", action="store_true", help=f"read your ratings back from {RATINGS_CSV.name}")
    parser.add_argument("--check", action="store_true", help="also how well the model predicts your later ratings from earlier ones")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    try:
        if args.restore:
            print(f"Restored {restore():,} ratings from {RATINGS_CSV}")
            return
        if args.action:
            if not args.items:
                sys.exit(f"Say which songs to {args.action}: numbers in the last list, or words of a title or artist")
            for row in rate(args.items, args.action).itertuples():
                done = {"like": "Liked", "dislike": "Disliked", "neutral": "Neutral on", "clear": "Took back your rating of"}[args.action]
                print(f"{done}: {row.track_name} - {row.artist_name}")
            print(f"Saved to {RECOMMENDER_DB} and backed up to {RATINGS_CSV}; your next recommendations use it")
            return
        library = load_library()
    except (FileNotFoundError, RuntimeError, LookupError) as error:
        sys.exit(str(error))
    ratings = library.ratings

    rated = current_ratings(ratings)
    print(
        f"Your ratings: {len(rated)} songs ({(rated.rating == 1).sum()} liked, {(rated.rating == -1).sum()} disliked, "
        f"{(rated.rating == NEUTRAL).sum()} neutral), "
        f"{(rated['mode'] == 'discover').sum()} of them songs new to you"
    )
    for row in rated.tail(10).itertuples():
        word = {1: "like", -1: "dislike", NEUTRAL: "neutral"}[row.rating]
        print(f"    {word:<8}{row.rated_at:%d %b %Y}  {row.track_name} - {row.artist_name}  ({row.mode})")
    model = feedback_model(ratings, library.candidates)
    print(f"The model learned from your ratings: {model.message}")
    for line in weight_lines(model):
        print(line)
    if args.check:
        print("\n".join(forward_check(ratings, library.candidates, np.random.default_rng(SEED))))
    if len(ratings):
        print(f"Backup of your ratings: {RATINGS_CSV}")
    else:
        print("Rate the songs of a list with python -m recommender.feedback like 1 3, or dislike 2")


if __name__ == "__main__":
    main()
