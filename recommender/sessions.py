"""Songs you play together: song vectors learned from your listening sessions.

    python -m recommender.sessions

Genres say how Last.fm tags a song; your sessions say what you actually play
with it. Two pop songs share all their genres, but you may play one with
your rock songs and never the other. This module learns that from the order of
your plays, the way word2vec learns what words mean from the words around
them (unsupervised learning: there are no labels, only co-occurrence):

1. Count how often each pair of songs was played within WINDOW plays of each
   other in a session (gold.sessions: plays with pauses of at most 30
   minutes). Early skips and tracks that aren't music to you are left out.
2. Turn the counts into PPMI, positive pointwise mutual information: the log
   of how much more often the pair was played together than two songs with
   their numbers of plays would be by chance, or 0 if not more often. This
   takes popularity out: a song in heavy rotation is played near everything,
   but not more often than its plays predict. Rarely played partners get a
   little less credit (CONTEXT_SMOOTHING), since a single chance meeting
   makes a large ratio.
3. Compress the PPMI table, one row and one column per song, to DIMENSIONS
   numbers per song by a truncated singular value decomposition (SVD): the
   closest DIMENSIONS-column approximation of the table. Songs played near
   the same songs get similar rows even if you never played them next to
   each other. Each row is scaled to length 1, so the similarity of two
   songs is the dot product of their rows, a cosine: near 1 for songs you
   play in the same company, near 0 for unrelated ones. A song never played
   near another one has a row of zeros.

similar_songs() in recommend.py ranks the songs like one you name by
LIKE_SESSION * session similarity + LIKE_GENRE * genre similarity +
LIKE_ROTATION * song_match (how much you played the song lately, as in
scoring.py). This command checks that choice, the way ranker.py checks its
model: on the first day of every quarter from FIRST_CUTOFF_YEARS after your
first play until the valid window, vectors are learned from the sessions before it, and each session in
the TEST_DAYS after it becomes a test. For each song in the session, the other
songs you played in it should rank high among all the songs you knew by then.
Shuffled sessions (shuffle on for at least SHUFFLED of the plays) are left
out, because what comes up in a shuffle doesn't depend on the song before it:
nothing ranks them better than how much you play each song.

The vectors were also tried, on the author's data, as a feature of the
learned rediscover model and as taste groups (k-means clusters of your songs,
each with its own genre profile) for discover. Neither helped
(docs/results.md), so they are used for similar songs only.

Run it from the project folder with python -m recommender.sessions, or as a
file, for example with VS Code's Run button. It takes about a minute.
"""

import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

if not __package__:
    # Started as a file (VS Code's Run button, or python recommender/sessions.py) instead of with
    # python -m: Python then looks for imports in recommender/, so add the project folder first.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recommender.data import Library, cutoff_after, load_library
from recommender.features import genre_matrix, genre_vocabulary
from recommender.metrics import auc, precision_at
from recommender.profile import song_evidence

# Two plays of a session count as played together when at most this many plays apart.
WINDOW = 5
# Numbers per song vector.
DIMENSIONS = 64
# Partners count with their number of plays to this power when the chance level is
# computed (word2vec's 0.75), so a rare partner doesn't make a pair look stronger than it is.
CONTEXT_SMOOTHING = 0.75
# The song vectors are the SVD's left singular vectors times the singular values to this
# power: 1 would let the strongest few directions dominate, 0 would weigh them all equally.
SINGULAR_POWER = 0.5
# Songs like one you name: the weights of session similarity, genre similarity and song_match.
# Chosen on the quarterly check below, on the author's data (docs/results.md).
LIKE_SESSION = 0.5
LIKE_GENRE = 0.5
LIKE_ROTATION = 0.1
# The quarterly check: when the first cutoff comes (years after your first play), how long after each cutoff its
# test sessions run, when a session counts as shuffled, and how many of each session's songs are tried as the seed.
FIRST_CUTOFF_YEARS = 2
TEST_DAYS = 90
SHUFFLED = 0.5
SEEDS_PER_SESSION = 5
SEED = 0


class SongSpace(NamedTuple):
    """What songs like one you name are ranked by, as of one moment; one row per song of library.songs."""

    vectors: np.ndarray  # session vectors: length 1, or zeros for a song never played near another
    genres: sparse.csr_matrix  # genre rows (features.py)
    song_match: np.ndarray  # how much you played the song lately, as in scoring.py: up to 1, 0 for none
    together: sparse.csr_matrix  # sessions x songs: 1 where the song was heard in the session


def heard(library: Library, start: pd.Timestamp | None, end: pd.Timestamp) -> pd.DataFrame:
    """Plays started in [start, end) that weren't skipped early, music only; start None: from the beginning."""
    plays = library.plays
    keep = (plays.started_at < end) & ~plays.skipped_early & library.songs.is_music.to_numpy()[plays.song]
    if start is not None:
        keep &= plays.started_at >= start
    return plays[keep]


def cooccurrence(library: Library, as_of: pd.Timestamp) -> sparse.csr_matrix:
    """Songs x songs: how often the two were heard within WINDOW plays of each other in a session before as_of."""
    plays = heard(library, None, as_of)
    songs, sessions = plays.song.to_numpy(), plays.session_id.to_numpy()
    rows, columns = [], []
    for distance in range(1, WINDOW + 1):
        first, second = songs[:-distance], songs[distance:]
        pair = (sessions[:-distance] == sessions[distance:]) & (first != second)
        rows += [first[pair], second[pair]]
        columns += [second[pair], first[pair]]
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    n = len(library.songs)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(n, n))


def ppmi(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    """Positive pointwise mutual information of a symmetric table of counts (step 2 of the docstring)."""
    counts = sparse.coo_matrix(counts)
    total = counts.sum()
    plays = np.asarray(counts.sum(axis=1)).ravel()
    partner = plays**CONTEXT_SMOOTHING
    partner = partner / partner.sum()
    pmi = np.log(counts.data / total / (plays[counts.row] / total) / partner[counts.col])
    keep = pmi > 0
    return sparse.csr_matrix((pmi[keep], (counts.row[keep], counts.col[keep])), shape=counts.shape)


def session_vectors(library: Library, as_of: pd.Timestamp) -> np.ndarray:
    """One vector per song of library.songs, learned from the sessions before as_of (step 3 of the docstring)."""
    table = ppmi(cooccurrence(library, as_of))
    if table.nnz == 0:
        return np.zeros((len(library.songs), DIMENSIONS))
    dimensions = min(DIMENSIONS, min(table.shape) - 1)
    left, singular, _ = randomized_svd(table, dimensions, random_state=SEED)
    vectors = left * singular**SINGULAR_POWER
    lengths = np.linalg.norm(vectors, axis=1)
    lengths[lengths == 0] = 1.0
    return vectors / lengths[:, None]


def song_space(library: Library, as_of: pd.Timestamp) -> SongSpace:
    plays = heard(library, None, as_of)
    sessions = pd.factorize(plays.session_id)[0]
    together = sparse.csr_matrix(
        (np.ones(len(plays)), (sessions, plays.song.to_numpy())), shape=(sessions.max() + 1 if len(plays) else 0, len(library.songs))
    )
    together.data[:] = 1.0  # a song heard twice in a session counts once
    weight = song_evidence(library, as_of).weight
    return SongSpace(
        vectors=session_vectors(library, as_of),
        genres=genre_matrix(library.songs.genres.tolist(), genre_vocabulary(library.songs.genres)),
        song_match=(weight / weight.max()).to_numpy(),
        together=together,
    )


def like_parts(space: SongSpace, seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Session and genre similarity of every song to each seed: two arrays of seeds x songs."""
    session = space.vectors[seeds] @ space.vectors.T
    genre = (space.genres[seeds] @ space.genres.T).toarray()
    return session, genre


def like_score(space: SongSpace, session: np.ndarray, genre: np.ndarray) -> np.ndarray:
    """The ranking of similar_songs() (see the docstring), from like_parts()."""
    return LIKE_SESSION * session + LIKE_GENRE * genre + LIKE_ROTATION * space.song_match


def shared_sessions(space: SongSpace, seed: int) -> np.ndarray:
    """For every song: in how many sessions you heard both it and seed."""
    return np.asarray(space.together.T @ space.together[:, seed].toarray()).ravel()


def together_rankings(space: SongSpace, seeds: np.ndarray) -> dict[str, np.ndarray]:
    """The rankings the check compares, each seeds x songs."""
    session, genre = like_parts(space, seeds)
    lately = np.broadcast_to(space.song_match, session.shape)
    return {
        "how much you played it lately": lately,
        # similar_songs() before phase 4: genres, songs with the same genres in order of how much you like them.
        "genres (--like before phase 4)": genre + 1e-6 * lately,
        "session vectors": session + 1e-6 * lately,
        f"--like: {LIKE_SESSION:g} session + {LIKE_GENRE:g} genre + {LIKE_ROTATION:g} lately": like_score(space, session, genre),
    }


def together_test(
    library: Library, start: pd.Timestamp, end: pd.Timestamp, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, float]]:
    """The songs-played-together test for the sessions that start in [start, end), with vectors learned before start.

    Returns one row per session and ranking (auc and p@10, averaged over the session's seeds), and for each
    ranking how alike its lists are: the share of one seed's top 10 that is also in another seed's, on
    average over pairs of different seeds. 1 means every song gets the same list, such as your most played.
    """
    space = song_space(library, start)
    known = np.zeros(len(library.songs), dtype=bool)
    known[heard(library, None, start).song.unique()] = True
    candidates = np.flatnonzero(known)

    window = library.plays[(library.plays.started_at >= start) & (library.plays.started_at < end)]
    shuffled = window.groupby("session_id").shuffle.mean() >= SHUFFLED
    plays = heard(library, start, end)
    plays = plays[known[plays.song.to_numpy()] & ~shuffled.reindex(plays.session_id).to_numpy()]

    rows, lists = [], {}  # lists: ranking -> {seed: its top 10}
    for session, group in plays.groupby("session_id"):
        songs = np.unique(group.song.to_numpy())
        if len(songs) < 2:
            continue
        seeds = rng.choice(songs, min(len(songs), SEEDS_PER_SESSION), replace=False)
        rankings = together_rankings(space, seeds)
        relevant = np.isin(candidates, songs)
        for name, scores in rankings.items():
            aucs, precisions = [], []
            for row, seed in enumerate(seeds):
                others = candidates != seed
                ranked = scores[row, candidates][others]
                aucs.append(auc(relevant[others], ranked))
                precisions.append(precision_at(relevant[others], ranked, 10))
                lists.setdefault(name, {})[seed] = candidates[others][np.argsort(-ranked, kind="stable")[:10]]
            rows.append({"session": session, "ranking": name, "auc": np.mean(aucs), "p@10": np.mean(precisions)})
    return pd.DataFrame(rows, columns=["session", "ranking", "auc", "p@10"]), {
        name: lists_alike(list(tops.values()), len(library.songs)) for name, tops in lists.items()
    }


def lists_alike(tops: list[np.ndarray], songs: int) -> float:
    """The share of one top-10 list that is also in another, on average over all pairs of the lists."""
    if len(tops) < 2:
        return float("nan")
    rows = np.repeat(np.arange(len(tops)), [len(top) for top in tops])
    member = sparse.csr_matrix((np.ones(len(rows)), (rows, np.concatenate(tops))), shape=(len(tops), songs))
    shared = (member @ member.T).toarray()
    pairs = len(tops) * (len(tops) - 1)
    return float((shared.sum() - np.trace(shared)) / pairs / 10)


def quarterly_check(library: Library) -> pd.DataFrame:
    """together_test() for every quarter from FIRST_CUTOFF_YEARS after your first play whose TEST_DAYS end before the
    valid window."""
    rng = np.random.default_rng(SEED)
    first = cutoff_after(library, FIRST_CUTOFF_YEARS)
    rows = []
    for cutoff in pd.date_range(first, library.split_starts["valid"], freq="QS"):
        end = cutoff + pd.Timedelta(TEST_DAYS, unit="D")
        if end > library.split_starts["valid"]:
            break
        sessions, alike = together_test(library, cutoff, end, rng)
        for name, result in sessions.groupby("ranking", sort=False):
            rows.append({
                "cutoff": cutoff, "ranking": name, "sessions": len(result), "auc": result.auc.mean(),
                "p@10": result["p@10"].mean(), "alike": alike[name],
            })
    return pd.DataFrame(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    started = time.monotonic()
    try:
        library = load_library()
    except (FileNotFoundError, RuntimeError) as error:
        sys.exit(str(error))

    space = song_space(library, library.ends_at)
    with_vector = (np.abs(space.vectors).sum(axis=1) > 0) & library.songs.is_music.to_numpy()
    print(f"Session vectors from all your sessions: {with_vector.sum():,} of {len(library.songs):,} songs have one")

    months = quarterly_check(library)
    if months.empty:
        sys.exit(
            f"Too little history for the quarterly check: it starts {FIRST_CUTOFF_YEARS} years after your first play "
            f"({cutoff_after(library, FIRST_CUTOFF_YEARS):%b %Y}) and needs a quarter that ends before the valid window."
        )
    quarters = months.pivot(index="cutoff", columns="ranking", values="auc")
    precision = months.pivot(index="cutoff", columns="ranking", values="p@10")
    names = list(dict.fromkeys(months.ranking))
    old = names[1]
    print(
        f"Songs played together, quarter by quarter, {quarters.index.min():%b %Y} to {quarters.index.max():%b %Y} "
        f"({len(quarters)} quarters, {months.groupby('cutoff').sessions.first().sum():,} sessions with shuffle mostly off), "
        "vectors learned only from earlier sessions:"
    )
    print(f"    {'ranked by':<48}{'mean AUC':>9}{'P@10':>7}{'lists alike':>13}  against {old}")
    for name in names:
        result = months[months.ranking == name]
        line = f"    {name:<48}{result.auc.mean():>9.3f}{result['p@10'].mean():>7.3f}{result.alike.mean():>13.0%}"
        if name != old:
            gain = quarters[name] - quarters[old]
            margin = 2 * gain.std(ddof=1) / np.sqrt(len(gain))
            line += f"  AUC {gain.mean():+.3f} +- {margin:.3f}, better in {(gain > 0).sum()} of {len(gain)}"
            top = precision[name] - precision[old]
            line += f"; P@10 {top.mean():+.3f}, better in {(top > 0).sum()}"
        print(line)
    print(f"Done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
