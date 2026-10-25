"""Ranking metrics, and the report table that evaluate.py prints.

evaluate.py's docstring explains each metric. Every ranking metric takes
`relevant` (bool, one per song) and `scores` (higher first, one per song);
list_diversity() measures a list of songs instead.
"""

from typing import NamedTuple

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import ndcg_score

# Resamples for the 95% ranges: enough to place them to about 0.01. More is
# slow on the discover pools of 20,000 songs.
BOOTSTRAP_RUNS = 300
RANDOM_ORDERS = 200


class Task(NamedTuple):
    name: str
    description: str
    relevant: np.ndarray  # bool, one per song ranked
    rankings: dict[str, np.ndarray]  # name -> score per song, higher first
    notes: tuple[str, ...] = ()  # lines printed after the table


def auc(relevant: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve from ranks, ties counting half (the Mann-Whitney U statistic)."""
    positives = relevant.sum()
    negatives = len(relevant) - positives
    ranks = rankdata(scores)
    return (ranks[relevant].sum() - positives * (positives + 1) / 2) / (positives * negatives)


def auc_range(relevant: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """The 2.5% and 97.5% points of AUC over bootstrap resamples of the songs."""
    values = []
    for _ in range(BOOTSTRAP_RUNS):
        pick = rng.integers(0, len(relevant), len(relevant))
        if 0 < relevant[pick].sum() < len(pick):
            values.append(auc(relevant[pick], scores[pick]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def relevant_in_top(relevant: np.ndarray, scores: np.ndarray, k: int) -> float:
    """How many relevant songs are in the top k. Songs tied at the cut count with their average relevance."""
    k = min(k, len(scores))
    cut = np.sort(scores)[::-1][k - 1]
    above = scores > cut
    tied = scores == cut
    return float(relevant[above].sum() + (k - above.sum()) * relevant[tied].mean())


def precision_at(relevant: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of relevant songs in the top k."""
    return relevant_in_top(relevant, scores, k) / min(k, len(scores))


def recall_at(relevant: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of all relevant songs that are in the top k."""
    return relevant_in_top(relevant, scores, k) / relevant.sum()


def list_diversity(genre_rows) -> float:
    """1 minus the average genre similarity of the pairs of songs in a list, from their genre rows (features.py).

    0 when all the songs have the same genres, 1 when no two share one. Songs without genres are left out;
    NaN when fewer than two songs have genres.
    """
    rows = genre_rows[np.asarray(genre_rows.sum(axis=1)).ravel() > 0]
    n = rows.shape[0]
    if n < 2:
        return float("nan")
    similarity = (rows @ rows.T).toarray()
    return float(1 - (similarity.sum() - np.trace(similarity)) / (n * (n - 1)))


def metrics(relevant: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auc": auc(relevant, scores),
        "p@10": precision_at(relevant, scores, 10),
        "ndcg@10": float(ndcg_score([relevant.astype(float)], [scores], k=10)),
        "r@100": recall_at(relevant, scores, 100),
    }


def report(task: Task, rng: np.random.Generator) -> None:
    relevant = task.relevant
    print(f"  {task.name}: {len(relevant):,} {task.description}")
    if not 0 < relevant.sum() < len(relevant):
        print(f"    Note: {'all' if relevant.all() else 'none'} of them are relevant in this window, so there is nothing to rank")
        return
    print(f"    {relevant.sum():,} relevant ({relevant.mean():.1%}: what a random pick gets)")
    print(f"    {'ranked by':<32}{'AUC (95% range)':<20}{'P@10':>6}{'NDCG@10':>9}{'R@100':>7}")
    shuffles = [metrics(relevant, rng.random(len(relevant))) for _ in range(RANDOM_ORDERS)]
    r = {name: np.mean([m[name] for m in shuffles]) for name in shuffles[0]}
    print(f"    {'random':<32}{r['auc']:.2f}{'':<16}{r['p@10']:>6.2f}{r['ndcg@10']:>9.2f}{r['r@100']:>7.2f}")
    for name, scores in task.rankings.items():
        m = metrics(relevant, scores)
        low, high = auc_range(relevant, scores, rng)
        print(f"    {name:<32}{m['auc']:.2f} ({low:.2f}-{high:.2f}){'':<4}{m['p@10']:>6.2f}{m['ndcg@10']:>9.2f}{m['r@100']:>7.2f}")
    for line in task.notes:
        print(f"    {line}")
