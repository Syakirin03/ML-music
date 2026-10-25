"""Learn from your history how likely you are to go back to a song: a supervised model for songs to rediscover.

    python -m recommender.ranker
    python -m recommender.ranker --ratings    only the model learned from your ratings of songs to rediscover

The rediscover score in scoring.py adds its parts with weights set by hand.
This module learns the weights from your history instead. As a supervised
learning problem:

    examples  on the first day of every month from FIRST_CUTOFF_YEARS after
              your first play on (a
              cutoff), the songs recommend_songs(mode="rediscover") would
              have offered then: liked (3+ listens) but unplayed for 90 days
    features  FEATURES below: numbers known at the cutoff, from plays before it only
    label     1 if you went back to the song, with at least RETURN_LISTENS
              listens in the HORIZON_DAYS after the cutoff, else 0
    model     logistic regression: the chance of a 1 is
              1 / (1 + e^-(b + w1*x1 + w2*x2 + ...)), where each x is a
              feature scaled to mean 0 and standard deviation 1 over the
              training examples, so the weights w can be compared

Training finds the weights whose chances fit the labels best (the smallest log
loss), with a penalty on large weights so the model doesn't learn noise: the
smaller the regularization C, the stronger the penalty. C is chosen on the
valid window from REGULARIZATIONS. Gradient-boosted trees, which can also
learn bends and interactions that a weighted sum can't, are trained alongside
for comparison.

A model may only learn from what was known before the moment it is judged on.
evaluate.py therefore trains one per window, from the cutoffs whose
HORIZON_DAYS ended before the window began. The valid and test windows hold
few songs gone back to (43 and 71 in the author's data), so they can disagree by chance; this
command also judges the models month by month on every cutoff from
ROLLING_FROM_YEARS after your first play until the valid window (rolling
origin), each by a model trained
only on earlier cutoffs. That is what decided that recommend_songs(mode=
"rediscover") ranks with the logistic regression rather than the hand-set
formula: on the author's data it was better in 32 of 45 months (docs/results.md).

It then trains one model on every cutoff whose HORIZON_DAYS have passed, shows
what it learned and whether its chances match what happened, and saves it in
data/user/recommender.duckdb (ranker_weights, ranker_info).

Your ratings of songs to rediscover answer another question: not whether you
play a song again, but whether you want it back when it is offered. On the
author's ratings, the model above told likes from dislikes only a little (AUC
0.58 on 158 ratings). So a second logistic regression learns from those
ratings (rediscover_feedback): the same FEATURES, with song_match,
genre_match and artist_match as they were when you saw the song, plus one
column per genre of at least feedback.MIN_GENRE_RATINGS rated songs. It is
used under the rules of the one for songs new to you (feedback.py): MIN_EACH
likes and dislikes, and a cross-validated AUC above that of the chance the
songs were listed by, by MODEL_MARGIN. The rest of FEATURES is worked out
again as of the end of your history, which is what you saw until a newer
export. --ratings shows it, with a forward check: your ratings in time order,
each block predicted from the ratings before it.

Run it from the project folder with python -m recommender.ranker, or as a
file, for example with VS Code's Run button. It takes about a minute.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import NamedTuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

if not __package__:
    # Started as a file (VS Code's Run button, or python recommender/ranker.py) instead of with
    # python -m: Python then looks for imports in recommender/, so add the project folder first.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recommender.data import (
    OPINIONS, RECOMMENDER_DB, Library, current_ratings, cutoff_after, load_library, song_key, window_listens,
)
from recommender.feedback import SEED as FEEDBACK_SEED, FeedbackModel, forward_lines, learn_from_ratings, stored_parts, weight_lines
from recommender.features import genre_list
from recommender.metrics import auc
from recommender.profile import TasteProfile, build_profile, song_evidence, sql_text
from recommender.scoring import DORMANT_DAYS, FAMILIARITY, SONG_SHARE, rediscovery_candidates, score_songs

# The first cutoff comes this many years after your first play (data.cutoff_after()): before, there are too
# few liked songs to learn from.
FIRST_CUTOFF_YEARS = 2
# How long after a cutoff a song may be played again to count as going back to it.
HORIZON_DAYS = 90
# Going back means at least this many listens; evaluate.py's returns task uses it too.
RETURN_LISTENS = 1.0
# "Lately" for artist_recent and genre_recent: the days before the cutoff.
RECENT_DAYS = 30
# The regularization C values tried; the one with the best AUC on the valid
# window is used. Within 0.005 of the best, the smallest C (strongest penalty) wins.
REGULARIZATIONS = (0.01, 0.1, 1.0, 10.0)
# The rolling check judges the cutoffs from this many years after your first play: earlier ones have too
# little to learn from.
ROLLING_FROM_YEARS = 4
SEED = 0
# The forward check of the model learned from your ratings of songs to rediscover predicts those after the first
# this many. Fixed before the check was first run, with the author's 158 such ratings (docs/results.md).
RATINGS_FORWARD_START = 60
# The parts of FEATURES that a rating keeps as they were when you saw the song.
SHOWN_PARTS = ["song_match", "genre_match", "artist_match"]

FEATURES = {
    "song_match": "how much you played the song lately (as in scoring.py)",
    "genre_match": "how close its genres are to your taste lately (as in scoring.py)",
    "artist_match": "your profile's weight for its artist (as in scoring.py)",
    "log_listens": "log of all its listens: how much you ever played it",
    "log_days_dormant": "log of the days since you last played it",
    "log_days_known": "log of the days since you first played it",
    "finish_rate": "share of its plays that you finished",
    "skip_rate": "share of its plays that you skipped within 30 seconds",
    "artist_recent": f"log of your listens to its artist in the {RECENT_DAYS} days before",
    "genre_recent": f"share of your listens in the {RECENT_DAYS} days before that were in its main genre",
    "has_genres": "1 if Last.fm has genres for it",
}

TABLE_COMMENTS = {
    "ranker_weights": "The learned rediscover model: one weight per feature. Built by python -m recommender.ranker.",
    "ranker_info": "When and how the model in ranker_weights was trained. One row.",
}
COLUMN_COMMENTS = {
    "ranker_weights": {
        "feature": "The feature; recommender/ranker.py describes each.",
        "mean": "Its mean over the training examples, subtracted before weighing.",
        "scale": "Its standard deviation over the training examples, divided by before weighing.",
        "weight": "The learned weight: how much one standard deviation more of the feature moves the log odds of going back.",
    },
    "ranker_info": {
        "trained_at": "When the model was trained.",
        "trained_until": "Only cutoffs whose horizon ended by then were used.",
        "horizon_days": "Going back means a listen within this many days after a cutoff.",
        "cutoffs": "Monthly cutoffs trained on.",
        "examples": "Songs trained on, over all cutoffs.",
        "positives": "How many of them you went back to.",
        "regularization": "C of the logistic regression, chosen on the valid window.",
        "intercept": "b, the log odds of going back for a song with every feature at its mean.",
    },
}


class Ranker(NamedTuple):
    weights: pd.DataFrame  # by feature: mean, scale, weight
    intercept: float
    trained_until: pd.Timestamp
    horizon_days: int


def candidate_features(
    library: Library, as_of: pd.Timestamp, profile: TasteProfile | None = None, dormant_days: float = DORMANT_DAYS
) -> pd.DataFrame:
    """FEATURES of the songs recommend_songs(mode="rediscover") would offer at as_of; index: song number."""
    profile = profile or build_profile(as_of, library)
    evidence = song_evidence(library, as_of)
    candidates = rediscovery_candidates(library, as_of, dormant_days, evidence)
    songs = candidates.index.to_numpy()
    parts = score_songs(profile, library, songs, FAMILIARITY, evidence)
    first_played = library.plays.groupby("song").started_at.min().reindex(songs)
    plays = library.plays
    recent = plays[(plays.started_at >= as_of - pd.Timedelta(RECENT_DAYS, unit="D")) & (plays.started_at < as_of)]
    artist_keys = library.songs.artist_key
    artist_recent = recent.listen.groupby(artist_keys.iloc[recent.song].to_numpy()).sum()
    main_genre = library.songs.genres.map(lambda g: (genre_list(g) or [None])[0])
    genre_recent = recent.listen.groupby(main_genre.iloc[recent.song].to_numpy()).sum() / max(recent.listen.sum(), 1e-9)

    def days_since(times: pd.Series) -> np.ndarray:
        return ((as_of - times).dt.total_seconds() / 86400).to_numpy()

    return pd.DataFrame({
        "song_match": parts.song_match.to_numpy(),
        "genre_match": parts.genre_match.to_numpy(),
        "artist_match": parts.artist_match.to_numpy(),
        "log_listens": np.log1p(candidates.listens.to_numpy()),
        "log_days_dormant": np.log(days_since(candidates.last_played_at)),
        "log_days_known": np.log(days_since(first_played)),
        "finish_rate": (candidates.finished / candidates.plays).to_numpy(),
        "skip_rate": (candidates.early_skips / candidates.plays).to_numpy(),
        "artist_recent": np.log1p(artist_recent.reindex(artist_keys.iloc[songs]).fillna(0.0).to_numpy()),
        "genre_recent": genre_recent.reindex(main_genre.iloc[songs]).fillna(0.0).to_numpy(),
        "has_genres": main_genre.iloc[songs].notna().astype(float).to_numpy(),
    }, index=pd.Index(songs, name="song"))


def horizon_end(cutoff: pd.Timestamp) -> pd.Timestamp:
    return cutoff + pd.Timedelta(HORIZON_DAYS, unit="D")


def went_back(library: Library, songs: pd.Index, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """1 for the songs with at least RETURN_LISTENS listens in [start, end), else 0."""
    return (window_listens(library, start, end).reindex(songs).fillna(0.0) >= RETURN_LISTENS).astype(int).to_numpy()


def monthly_cutoffs(library: Library, until: pd.Timestamp) -> list[pd.Timestamp]:
    """Midnight on the first day of each month from FIRST_CUTOFF_YEARS after your first play on, whose HORIZON_DAYS
    end by until."""
    months = pd.date_range(cutoff_after(library, FIRST_CUTOFF_YEARS), until, freq="MS")
    return [month for month in months if horizon_end(month) <= until]


def examples(library: Library, cutoffs: list[pd.Timestamp]) -> pd.DataFrame:
    """Every cutoff's candidates with their FEATURES, the cutoff, and the label."""
    frames = []
    for cutoff in cutoffs:
        features = candidate_features(library, cutoff)
        features["label"] = went_back(library, features.index, cutoff, horizon_end(cutoff))
        features["cutoff"] = cutoff
        frames.append(features)
    return pd.concat(frames) if frames else pd.DataFrame(columns=[*FEATURES, "label", "cutoff"])


def fit_logistic(train: pd.DataFrame, regularization: float) -> Pipeline:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=regularization, max_iter=5000))
    return model.fit(train[list(FEATURES)], train.label)


def fit_trees(train: pd.DataFrame) -> HistGradientBoostingClassifier:
    """Gradient-boosted trees with modest settings, fixed beforehand rather than tuned; trees need no scaling."""
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, random_state=SEED
    )
    return model.fit(train[list(FEATURES)], train.label)


def trained_before(all_examples: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """The examples a model judged at start may learn from: cutoffs whose HORIZON_DAYS ended by start."""
    return all_examples[all_examples.cutoff.map(horizon_end) <= start]


def learnable(all_examples: pd.DataFrame, start: pd.Timestamp) -> bool:
    """Whether a model judged at start has something to learn from: songs gone back to, and songs not, before it."""
    return trained_before(all_examples, start).label.nunique() == 2


def choose_regularization(
    library: Library, all_examples: pd.DataFrame, valid_features: pd.DataFrame | None = None
) -> tuple[float, dict[float, float]]:
    """The C of REGULARIZATIONS with the best AUC on the valid window, and the AUC of each.

    Relevant there is what evaluate.py's returns task counts: going back during the whole valid window. When
    the valid window has no song gone back to, or only those, AUC can't choose (each is NaN), so the smallest C,
    the strongest penalty, is kept.
    """
    start, end = library.split_starts["valid"], library.split_starts["test"]
    features = valid_features if valid_features is not None else candidate_features(library, start)
    relevant = went_back(library, features.index, start, end).astype(bool)
    if not 0 < relevant.sum() < len(relevant):
        return min(REGULARIZATIONS), dict.fromkeys(REGULARIZATIONS, float("nan"))
    train = trained_before(all_examples, start)
    scores = {
        c: auc(relevant, fit_logistic(train, c).predict_proba(features[list(FEATURES)])[:, 1]) for c in REGULARIZATIONS
    }
    best = max(scores.values())
    return min(c for c, value in scores.items() if value >= best - 0.005), scores


def window_chances(
    library: Library, start: pd.Timestamp, all_examples: pd.DataFrame, regularization: float
) -> dict[str, pd.Series]:
    """Chances of going back to each rediscovery candidate at start (index: song number), by models trained only
    on the cutoffs whose HORIZON_DAYS ended by start."""
    train = trained_before(all_examples, start)
    features = candidate_features(library, start)
    x = features[list(FEATURES)]
    return {
        "logistic regression": pd.Series(fit_logistic(train, regularization).predict_proba(x)[:, 1], index=features.index),
        "gradient-boosted trees": pd.Series(fit_trees(train).predict_proba(x)[:, 1], index=features.index),
    }


def rolling_check(library: Library, all_examples: pd.DataFrame, regularization: float) -> pd.DataFrame:
    """AUC per monthly cutoff from ROLLING_FROM_YEARS after your first play until the valid window, of the hand-set
    formula and of models trained only on earlier cutoffs, judged on going back within HORIZON_DAYS."""
    rows = []
    first = cutoff_after(library, ROLLING_FROM_YEARS)
    for cutoff, month in all_examples.groupby("cutoff"):
        if cutoff < first or horizon_end(cutoff) > library.split_starts["valid"]:
            continue
        relevant = month.label.to_numpy().astype(bool)
        if not 0 < relevant.sum() < len(relevant) or not learnable(all_examples, cutoff):
            continue
        train = trained_before(all_examples, cutoff)
        x = month[list(FEATURES)]
        taste = (1 - FAMILIARITY) * month.genre_match + FAMILIARITY * month.artist_match
        rows.append({
            "cutoff": cutoff,
            "returned": int(relevant.sum()),
            "formula": auc(relevant, (SONG_SHARE * month.song_match + (1 - SONG_SHARE) * taste).to_numpy()),
            "logistic regression": auc(relevant, fit_logistic(train, regularization).predict_proba(x)[:, 1]),
            "gradient-boosted trees": auc(relevant, fit_trees(train).predict_proba(x)[:, 1]),
        })
    return pd.DataFrame(rows)


def as_ranker(model: Pipeline, until: pd.Timestamp) -> Ranker:
    """A fitted logistic regression pipeline as plain numbers: what is saved, and all that predicting needs."""
    scaler, regression = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    weights = pd.DataFrame(
        {"mean": scaler.mean_, "scale": scaler.scale_, "weight": regression.coef_[0]}, index=pd.Index(list(FEATURES), name="feature")
    )
    return Ranker(weights, float(regression.intercept_[0]), until, HORIZON_DAYS)


def chances(ranker: Ranker, features: pd.DataFrame) -> np.ndarray:
    """The ranker's chance of going back to each song: 1 / (1 + e^-(b + sum of weight * scaled feature))."""
    w = ranker.weights
    log_odds = ranker.intercept + ((features[w.index] - w["mean"]) / w["scale"]).to_numpy() @ w["weight"].to_numpy()
    return 1 / (1 + np.exp(-log_odds))


def contributions(ranker: Ranker, features: pd.DataFrame) -> pd.DataFrame:
    """weight * scaled feature for each song and feature: how much each feature moved the song's log odds."""
    w = ranker.weights
    return (features[w.index] - w["mean"]) / w["scale"] * w["weight"]


def save_ranker(ranker: Ranker, info: dict, db: Path = RECOMMENDER_DB) -> None:
    db.parent.mkdir(parents=True, exist_ok=True)
    weights = ranker.weights.reset_index()
    info_frame = pd.DataFrame([{
        "trained_at": pd.Timestamp.now(tz=ranker.trained_until.tz), "trained_until": ranker.trained_until,
        "horizon_days": ranker.horizon_days, **info, "intercept": ranker.intercept,
    }])
    try:
        con = duckdb.connect(str(db))
    except duckdb.IOException as error:
        raise RuntimeError(f"Could not open {db} to save the model: {error}. Close other programs using it.") from error
    try:
        con.begin()
        for table, frame in (("ranker_weights", weights), ("ranker_info", info_frame)):
            con.register("frame", frame)
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM frame")
            con.unregister("frame")
            con.execute(f"COMMENT ON TABLE {table} IS {sql_text(TABLE_COMMENTS[table])}")
            for column, text in COLUMN_COMMENTS[table].items():
                con.execute(f"COMMENT ON COLUMN {table}.{column} IS {sql_text(text)}")
        con.commit()
    finally:
        con.close()


def ranker_summary(db: Path = RECOMMENDER_DB) -> dict | None:
    """What the saved model was trained on (ranker_info: cutoffs, examples, positives, built_at, ...), or None."""
    if not db.exists():
        return None
    con = duckdb.connect(str(db), read_only=True)
    try:
        info = con.sql("SELECT * FROM ranker_info").df()
    except duckdb.CatalogException:
        return None
    finally:
        con.close()
    return info.iloc[0].to_dict() if len(info) else None


def load_ranker(db: Path = RECOMMENDER_DB) -> Ranker | None:
    """The saved model, or None if there is none or it was trained on other features."""
    if not db.exists():
        return None
    con = duckdb.connect(str(db), read_only=True)
    try:
        weights = con.sql("SELECT feature, mean, scale, weight FROM ranker_weights").df().set_index("feature")
        info = con.sql("SELECT trained_until, horizon_days, intercept FROM ranker_info").df().iloc[0]
    except duckdb.CatalogException:
        return None
    finally:
        con.close()
    if list(weights.index) != list(FEATURES):
        return None
    return Ranker(weights, float(info.intercept), info.trained_until, int(info.horizon_days))


def rating_numbers(features: pd.DataFrame, parts: pd.DataFrame | None = None) -> pd.DataFrame:
    """FEATURES of songs to rediscover (candidate_features()) as the model learned from ratings takes them, with
    SHOWN_PARTS from parts (one row per song, the same order) when given."""
    numbers = features[list(FEATURES)].reset_index(drop=True)
    if parts is not None:
        for part in SHOWN_PARTS:
            numbers[part] = parts[part].astype(float).to_numpy()
    return numbers


def rated_rediscover(library: Library) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Your current likes and dislikes of songs to rediscover, oldest first, with their inputs and genres. A rated song that
    has since stopped being one to rediscover (fewer than LIKED_LISTENS listens, or not music) is left out."""
    rated = current_ratings(library.ratings)
    rated = rated[(rated["mode"] == "rediscover") & rated.rating.isin(OPINIONS)].sort_values("rated_at")
    keys = song_key(library.songs.artist_name, library.songs.track_name)
    number_of = pd.Series(np.arange(len(keys)), index=keys.to_numpy())
    number_of = number_of[~number_of.index.duplicated()]
    features = candidate_features(library, library.ends_at, dormant_days=0)
    songs = rated.song_key.map(number_of)
    keep = songs.isin(features.index).to_numpy()
    rated, songs = rated[keep].reset_index(drop=True), songs[keep].astype(int).to_numpy()
    numbers = rating_numbers(features.loc[songs], stored_parts(rated, SHOWN_PARTS))
    return rated, numbers, library.songs.genres.iloc[songs].reset_index(drop=True)


def rediscover_feedback(library: Library) -> FeedbackModel:
    """The model learned from your ratings of songs to rediscover, if it qualifies (see the docstring)."""
    rated, numbers, genres = rated_rediscover(library)
    return learn_from_ratings(rated, numbers, genres, "songs to rediscover", "the chance they were listed by")


def rediscover_forward_check(library: Library) -> list[str]:
    rated, numbers, genres = rated_rediscover(library)
    rng = np.random.default_rng(FEEDBACK_SEED)
    return forward_lines(rated, numbers, genres, RATINGS_FORWARD_START, "songs to rediscover", "the chance they were listed by", rng)


def calibration(predicted: np.ndarray, actual: np.ndarray) -> list[str]:
    """Lines comparing predicted chances with what happened, in bands of predicted chance."""
    bands = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)]
    lines = [f"    {'predicted':<12}{'songs':>6}{'predicted, on average':>24}{'went back':>12}"]
    for low, high in bands:
        inside = (predicted >= low) & (predicted < high)
        if inside.any():
            label = f"{low:.0%}-{min(high, 1):.0%}"
            lines.append(f"    {label:<12}{inside.sum():>6}{predicted[inside].mean():>24.1%}{actual[inside].mean():>12.1%}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratings", action="store_true", help="only the model learned from your ratings, with its forward check")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    started = time.monotonic()
    try:
        library = load_library()
    except (FileNotFoundError, RuntimeError) as error:
        sys.exit(str(error))
    if args.ratings:
        model = rediscover_feedback(library)
        print(f"The model learned from your ratings of songs to rediscover: {model.message}")
        for line in weight_lines(model) + rediscover_forward_check(library):
            print(line)
        return

    cutoffs = monthly_cutoffs(library, library.ends_at)
    all_examples = examples(library, cutoffs)
    if not learnable(all_examples, library.split_starts["valid"]):
        sys.exit(
            f"Too little history to learn from yet. The examples start {FIRST_CUTOFF_YEARS} years after your first play "
            f"({cutoff_after(library, FIRST_CUTOFF_YEARS):%d %b %Y}), and the model needs some, with songs you went back "
            f"to and songs you didn't, before the valid window ({library.split_starts['valid']:%d %b %Y}). Until then, "
            "songs to rediscover are ranked by the hand-set formula."
        )
    print(
        f"Examples: {len(cutoffs)} monthly cutoffs, {cutoffs[0]:%b %Y} to {cutoffs[-1]:%b %Y}: {len(all_examples):,} songs, "
        f"{all_examples.label.sum():,} of them gone back to within {HORIZON_DAYS} days ({all_examples.label.mean():.1%})"
    )

    regularization, valid_aucs = choose_regularization(library, all_examples)
    print(f"Regularization C, chosen on the valid window ({library.split_starts['valid']:%d %b %Y}) by AUC:")
    for c, value in valid_aucs.items():
        print(f"    C={c:<6g} AUC {value:.3f}{'   <- chosen' if c == regularization else ''}")

    months = rolling_check(library, all_examples, regularization)
    if months.empty:
        print(
            f"Month by month: nothing to judge yet. The check starts {ROLLING_FROM_YEARS} years after your first play "
            f"({cutoff_after(library, ROLLING_FROM_YEARS):%b %Y}), and judges months whose {HORIZON_DAYS} days end "
            "before the valid window."
        )
    else:
        print(
            f"Month by month, {months.cutoff.min():%b %Y} to {months.cutoff.max():%b %Y} ({len(months)} cutoffs, "
            f"{months.returned.sum():,} songs gone back to), each model trained only on earlier cutoffs:"
        )
        for name in ("formula", "logistic regression", "gradient-boosted trees"):
            line = f"    {name:<24} mean AUC {months[name].mean():.3f}"
            if name != "formula":
                gain = months[name] - months.formula
                margin = 2 * gain.std(ddof=1) / np.sqrt(len(gain))
                line += f", {gain.mean():+.3f} +- {margin:.3f} against the formula, better in {(gain > 0).sum()} of {len(gain)} months"
            print(line)

    ranker = as_ranker(fit_logistic(all_examples, regularization), library.ends_at)
    print(f"What the model learned, trained on all {len(cutoffs)} cutoffs. Per standard deviation more of a feature:")
    for feature, row in ranker.weights.reindex(ranker.weights.weight.abs().sort_values(ascending=False).index).iterrows():
        print(f"    {row.weight:+.2f}  odds x{np.exp(row.weight):<5.2f} {feature:<17}{FEATURES[feature]}")

    test_start = library.split_starts["test"]
    test_model = as_ranker(fit_logistic(trained_before(all_examples, test_start), regularization), test_start)
    test_features = candidate_features(library, test_start)
    actual = went_back(library, test_features.index, test_start, horizon_end(test_start))
    print(
        f"Do its chances mean what they say? Songs to rediscover on {test_start:%d %b %Y}, by a model trained only on "
        f"earlier cutoffs, and whether you went back within {HORIZON_DAYS} days:"
    )
    print("\n".join(calibration(chances(test_model, test_features), actual)))

    try:
        save_ranker(ranker, {
            "cutoffs": len(cutoffs), "examples": len(all_examples), "positives": int(all_examples.label.sum()),
            "regularization": regularization,
        })
    except RuntimeError as error:
        sys.exit(str(error))
    print(f"Saved to {RECOMMENDER_DB} (ranker_weights, ranker_info) in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
