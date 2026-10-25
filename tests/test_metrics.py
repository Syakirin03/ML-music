"""The measures every test of the recommender reports, and how a ranking becomes a list."""

import numpy as np
import pytest

from recommender.features import genre_matrix, genre_vocabulary
from recommender.metrics import auc, list_diversity, precision_at, recall_at
from recommender.recommend import MAX_PER_ARTIST, varied_top


def genre_rows(genre_lists):
    return genre_matrix(genre_lists, genre_vocabulary(genre_lists))


def test_auc_is_the_chance_a_relevant_song_ranks_above_another():
    relevant = np.array([True, True, False, False])
    assert auc(relevant, np.array([4, 3, 2, 1])) == 1.0
    assert auc(relevant, np.array([1, 2, 3, 4])) == 0.0
    assert auc(relevant, np.array([1, 1, 1, 1])) == 0.5  # ties count half


def test_precision_and_recall_at_k():
    relevant = np.array([True, False, True, False, False])
    scores = np.array([5, 4, 3, 2, 1])
    assert precision_at(relevant, scores, 2) == 0.5
    assert recall_at(relevant, scores, 3) == 1.0


def test_list_diversity_runs_from_one_genre_to_none_shared():
    assert list_diversity(genre_rows([["pop"], ["pop"], ["pop"]])) == pytest.approx(0.0)
    assert list_diversity(genre_rows([["pop"], ["rock"], ["jazz"]])) == pytest.approx(1.0)


def test_without_variety_a_list_is_the_score_order_with_the_artist_cap():
    artists = np.array(["a", "a", "a", "b", "c"])
    listed = varied_top(np.array([5.0, 4, 3, 2, 1]), genre_rows([["pop"]] * 5), artists, variety=0, top_n=4)
    assert MAX_PER_ARTIST == 2
    assert listed.tolist() == [0, 1, 3, 4]  # artist a's third song waits


def test_variety_trades_some_score_for_a_genre_not_listed_yet():
    rows = genre_rows([["pop"], ["pop"], ["jazz"]])
    listed = varied_top(np.array([1.0, 0.9, 0.8]), rows, np.array(["a", "b", "c"]), variety=0.5, top_n=2)
    assert listed.tolist() == [0, 2]  # jazz before a second pop song
