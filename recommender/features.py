"""Turn songs into rows of numbers that can be compared with each other and with your profile.

For now a song is described by its Last.fm genres: one column per genre in
your library, with 1 for the song's main genre, 1/2 for its second, 1/3 for
its third and so on (fetch_genres.py lists a song's genres most tagged first),
and each row scaled to length 1. Two songs are then as similar as the cosine
of the angle between their rows, which for rows of length 1 is simply their
dot product: 1 for the same genres in the same order, 0 for no genre in common.

Artists are not columns. The profile keeps one weight per artist instead (see
profile.py), because two songs either have the same artist or they don't.

Weighting rare genres up (TF-IDF) was tried on the held-out data and did not
rank better, so the rows use the plain weights.
"""

from collections.abc import Sequence

import numpy as np
from scipy import sparse


def genre_list(genres) -> list[str]:
    """A song's genres as a list: [] where Last.fm had none (NULL comes from DuckDB as NA or None)."""
    return list(genres) if isinstance(genres, (list, tuple, np.ndarray)) else []


def genre_vocabulary(genre_lists: Sequence) -> dict[str, int]:
    """The column of each genre, in alphabetical order."""
    names = sorted({genre for genres in genre_lists for genre in genre_list(genres)})
    return {genre: column for column, genre in enumerate(names)}


def genre_matrix(genre_lists: Sequence, vocabulary: dict[str, int]) -> sparse.csr_matrix:
    """One row per song, with length 1, or all zeros for a song without genres. Genres outside vocabulary are left out."""
    rows, columns, values = [], [], []
    for row, genres in enumerate(genre_lists):
        for rank, genre in enumerate(genre_list(genres), 1):
            if genre in vocabulary:
                rows.append(row)
                columns.append(vocabulary[genre])
                values.append(1.0 / rank)
    matrix = sparse.csr_matrix((values, (rows, columns)), shape=(len(genre_lists), len(vocabulary)))
    lengths = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    lengths[lengths == 0] = 1.0
    return sparse.csr_matrix(sparse.diags(1.0 / lengths) @ matrix)


def cosine_to(matrix: sparse.csr_matrix, vector: np.ndarray) -> np.ndarray:
    """Cosine between each row of matrix (rows of length 1 or 0) and vector; 0 for rows of zeros."""
    length = np.linalg.norm(vector)
    if length == 0:
        return np.zeros(matrix.shape[0])
    return np.asarray(matrix @ vector).ravel() / length
