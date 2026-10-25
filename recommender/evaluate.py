"""Test the recommender on listening it has not seen: does it rank what you went on to like first?

    python -m recommender.evaluate [--window valid|test|both]

The test replays the past. For a window of your history (the valid and test
splits of gold.plays), it builds your profile as of the window's start, from
earlier plays and likes only, ranks songs with it, and then checks the ranking
against what you actually did during the window. Four tasks:

    discover   the songs recommend_songs() would have offered at the window's
               start: the window's pool in candidates.csv (built from your
               taste at that date), less the songs you had played. Relevant:
               the ones you went on to play during the window.
    new songs  the songs you first played during the window. Relevant: the
               ones that stuck, with at least STUCK_LISTENS listens in the
               window; the explicit check uses likes and playlist adds instead.
    returns    the songs recommend_songs(mode="rediscover") would have offered
               at the window's start (liked, but unplayed for DORMANT_DAYS).
               Relevant: the ones you went back to, with at least
               RETURN_LISTENS listens in the window. Besides the hand-set
               score, ranked by the learned models of ranker.py, each trained
               only on the monthly cutoffs whose 90 days ended before the
               window began.
    together   songs like one you name (similar_songs): for each session of
               the window with shuffle mostly off, each of up to 5 of its
               songs that you knew is the seed, and relevant are the other
               songs of the session. Scored per seed, then averaged per
               session; sessions.py explains it and runs it quarter by quarter.

Each ranking is scored four ways:

    AUC      the chance that a random relevant song is ranked above a random
             other one: 0.5 is a coin flip, 1 is perfect. In brackets, the
             range it would take in 95% of reruns on similar data (bootstrap):
             two rankings whose ranges overlap a lot are not really different.
    P@10     how many of the top 10 are relevant: what you would notice.
    NDCG@10  like P@10, but a relevant song counts more the higher it is.
    R@100    how many of all the relevant songs are in the top 100: for the
             large discover pools, where few songs are relevant.

The rankings score every song. The list recommend_songs() shows is its top 10
with at most 2 songs per artist and some variety, so for discover and returns
the lines below the table score that list too, at variety 0 and 0.5: how many
of its songs are relevant, its diversity (1 minus the average genre
similarity of its pairs of songs: 0 when all ten have the same genres) and
how many main genres it has.

"random" is the average of 200 random orders: the level to beat. The valid
window is for choosing settings such as FAMILIARITY; the test window, last,
is the honest estimate. The valid window stops where the test window starts,
so choosing settings on it never looks at test listening.

Run it from the project folder with python -m recommender.evaluate, or as a
file, for example with VS Code's Run button (which passes no options).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if not __package__:
    # Started as a file (VS Code's Run button, or python recommender/evaluate.py) instead of with
    # python -m: Python then looks for imports in recommender/, so add the project folder first.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recommender.data import RATING_COLUMNS, Library, load_library, song_key, window_listens
from recommender.features import genre_list, genre_matrix, genre_vocabulary
from recommender.metrics import BOOTSTRAP_RUNS, Task, list_diversity, report
from recommender.profile import RATING_LISTENS, NoProfile, build_profile, song_evidence
from recommender.ranker import (
    RETURN_LISTENS, choose_regularization, examples, learnable, monthly_cutoffs, window_chances,
)
from recommender.recommend import VARIETY, rank_discover, varied_top
from recommender.scoring import (
    DORMANT_DAYS, FAMILIARITY, discovery_candidates, rediscovery_candidates, score_candidates, score_songs
)
from recommender.sessions import SHUFFLED, together_test

# A new song stuck if it got at least this many listens in the window.
STUCK_LISTENS = 3.0
# Rankings scored for each task: the familiarity values tried for score.
FAMILIARITIES = (0.0, 0.25, 0.5, 0.75, 1.0)
# The variety values the shown lists are scored at.
SHOWN_VARIETIES = (0.0, 0.5)
# The feedback replay: how many lists of 10 one after another, and the values of RATING_LISTENS tried.
REPLAY_ROUNDS = 5
REPLAY_LISTENS = (5.0, 20.0, 50.0)
SEED = 0


def shown_lists(
    mode: str, relevant: np.ndarray, scores: np.ndarray, tiebreak: np.ndarray, genres: pd.Series, artist_keys: np.ndarray
) -> tuple[str, ...]:
    """Lines on the top 10 recommend_songs() would show for these scores, at each of SHOWN_VARIETIES.

    tiebreak orders songs with the same score, as recommend.py does; genres and artist_keys are per song.
    """
    order = np.lexsort((-tiebreak, -scores))
    genres = genres.iloc[order].reset_index(drop=True)
    rows = genre_matrix(genres.tolist(), genre_vocabulary(genres))
    lines = []
    for variety in SHOWN_VARIETIES:
        listed = varied_top(scores[order], rows, artist_keys[order], variety, 10)
        main_genres = genres.iloc[listed].map(lambda g: (genre_list(g) or [None])[0]).nunique()
        default = ", the default" if variety == VARIETY[mode] else ""
        lines.append(
            f"list shown, variety {variety:g}{default}: {int(relevant[order[listed]].sum())} of its {len(listed)} songs "
            f"relevant, diversity {list_diversity(rows[listed]):.2f}, {main_genres} main genres"
        )
    return tuple(lines)


def new_songs_task(library: Library, start: pd.Timestamp, end: pd.Timestamp, explicit: bool = False) -> Task:
    profile = build_profile(start, library)
    first_played = library.plays.groupby("song").started_at.min()
    songs = first_played[(first_played >= start) & (first_played < end)].index.to_numpy()
    songs = songs[library.songs.is_music.to_numpy()[songs]]
    if explicit:
        feedback = library.feedback[(library.feedback.ts >= start) & (library.feedback.ts < end)]
        liked = set(feedback.song[feedback.event.isin(["like", "playlist_add"])])
        relevant = np.isin(songs, list(liked))
        name, what = "new songs, explicit", "liked or added to a playlist"
    else:
        relevant = window_listens(library, start, end).reindex(songs).fillna(0).to_numpy() >= STUCK_LISTENS
        name, what = "new songs", f"stuck ({STUCK_LISTENS:g}+ listens)"
    # song_match is 0 for all of them: they were new at the start, so the score is taste alone.
    evidence = song_evidence(library, start)
    scores = {f: score_songs(profile, library, songs, f, evidence) for f in FAMILIARITIES}
    rankings = {"genre_match": scores[0.0].genre_match.to_numpy(), "artist_match": scores[0.0].artist_match.to_numpy()}
    rankings |= {f"score, familiarity {f:g}": scores[f].score.to_numpy() for f in FAMILIARITIES if 0 < f < 1}
    return Task(name, f"songs first played in the window; relevant: {what}", relevant, rankings)


def discover_task(library: Library, pool_name: str, start: pd.Timestamp, end: pd.Timestamp) -> Task:
    profile = build_profile(start, library)
    songs = discovery_candidates(library, start, pool_name)
    relevant = ((songs.first_played_at >= start) & (songs.first_played_at < end)).to_numpy()
    scores = {f: score_candidates(profile, library, songs, f) for f in FAMILIARITIES}
    rankings = {
        "popularity (Last.fm listeners)": songs.listeners.to_numpy().astype(float),
        "genre_match": scores[0.0].genre_match.to_numpy(),
        "artist_match": scores[0.0].artist_match.to_numpy(),
        "similar_match": scores[0.0].similar_match.to_numpy(),
    }
    rankings |= {f"score, familiarity {f:g}": scores[f].score.to_numpy() for f in FAMILIARITIES}
    # How many of the songs you found yourself in the window the pool could have offered at all.
    first_played = library.plays.groupby("song").started_at.min()
    found = first_played[(first_played >= start) & (first_played < end)].index.to_numpy()
    found = found[library.songs.is_music.to_numpy()[found]]
    found_keys = set(song_key(library.songs.artist_name.iloc[found], library.songs.track_name.iloc[found]))
    description = (
        f"songs in the {pool_name} pool you hadn't played; relevant: played in the window. "
        f"Of the {len(found_keys)} songs you first played in the window, {len(found_keys & set(songs.song_key))} are in the pool"
    )
    notes = shown_lists(
        "discover", relevant, scores[FAMILIARITY].score.to_numpy(), songs.listeners.to_numpy().astype(float),
        songs.genres, songs.artist_key.to_numpy(),
    )
    return Task("discover", description, relevant, rankings, notes)


def returns_task(
    library: Library, start: pd.Timestamp, end: pd.Timestamp, learned: dict[str, pd.Series] | None = None
) -> Task:
    """learned: ranker.window_chances() for this window, to rank alongside the formula."""
    profile = build_profile(start, library)
    evidence = song_evidence(library, start)
    candidates = rediscovery_candidates(library, start, DORMANT_DAYS, evidence)
    songs = candidates.index.to_numpy()
    relevant = window_listens(library, start, end).reindex(songs).fillna(0).to_numpy() >= RETURN_LISTENS
    scores = {f: score_songs(profile, library, songs, f, evidence) for f in FAMILIARITIES}
    rankings = {
        "your past listens": candidates.listens.to_numpy(),
        "played most recently": candidates.last_played_at.astype("int64").to_numpy().astype(float),
        "song_match": scores[0.0].song_match.to_numpy(),
        "genre_match": scores[0.0].genre_match.to_numpy(),
        "artist_match": scores[0.0].artist_match.to_numpy(),
    }
    rankings |= {f"score, familiarity {f:g}": scores[f].score.to_numpy() for f in FAMILIARITIES}
    rankings |= {f"learned: {name}": chance.reindex(songs).to_numpy() for name, chance in (learned or {}).items()}
    description = (
        f"liked songs unplayed for {DORMANT_DAYS} days at the start; relevant: played again ({RETURN_LISTENS:g}+ listens)"
    )
    # The list shown ranks by the learned logistic regression, or by the formula without one.
    shown = rankings.get("learned: logistic regression", scores[FAMILIARITY].score.to_numpy())
    notes = shown_lists(
        "rediscover", relevant, shown, candidates.listens.to_numpy(), library.songs.genres.iloc[songs],
        library.songs.artist_key.iloc[songs].to_numpy(),
    )
    return Task("returns", description, relevant, rankings, notes)


def replayed_ratings(listed: pd.DataFrame, liked: np.ndarray, dislikes: bool, at: pd.Timestamp) -> pd.DataFrame:
    """Ratings of pool songs as feedback.py would store them: a like for each liked one, a dislike for the others if dislikes."""
    rows = listed[liked | dislikes]
    return pd.DataFrame({
        "rated_at": at, "rating": np.where(liked[liked | dislikes], 1, -1), "mode": "discover",
        "song_key": rows.song_key.to_numpy(), "track_id": None, "artist_key": rows.artist_key.to_numpy(),
        "artist_name": rows.artist_name.to_numpy(), "track_name": rows.track_name.to_numpy(),
        "genres": rows.genres.to_numpy(), "list_id": None, "position": None, "score": np.nan, "parts": "{}",
    })[RATING_COLUMNS]


def feedback_replay(library: Library, pool_name: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Replays rating songs new to you: REPLAY_ROUNDS lists of 10 from the window's pool, each built after rating the
    lists before it, where you like the songs you went on to play in the window (and, with dislikes, dislike the rest)."""
    pool = discovery_candidates(library, start, pool_name)
    found = set(pool.song_key[((pool.first_played_at >= start) & (pool.first_played_at < end)).to_numpy()])
    variants = [("no ratings", None, False)]
    variants += [(f"likes, {listens:g} listens each", listens, False) for listens in REPLAY_LISTENS]
    variants += [(f"likes and dislikes, {RATING_LISTENS:g} each", RATING_LISTENS, True)]
    print(
        f"  feedback replay: {REPLAY_ROUNDS} lists of 10 songs new to you, one after another, each built after rating the "
        "ones before: liked are the songs you went on to play in the window"
    )
    print(f"    {'ratings':<32}{'relevant in list 1 to ' + str(REPLAY_ROUNDS):<28}{'lists 2-' + str(REPLAY_ROUNDS):>10}{'all':>6}")
    for name, listens, dislikes in variants:
        ratings, shown, hits = pd.DataFrame(columns=RATING_COLUMNS), set(), []
        for _ in range(REPLAY_ROUNDS):
            replay = library._replace(ratings=ratings)
            profile = build_profile(start, replay, rating_listens=listens or RATING_LISTENS)
            songs = pool[~pool.song_key.isin(shown)].reset_index(drop=True)
            ranked, listed, _ = rank_discover(replay, profile, songs, FAMILIARITY, VARIETY["discover"], 10)
            picked = songs.loc[ranked.index[listed]]
            liked = picked.song_key.isin(found).to_numpy()
            hits.append(int(liked.sum()))
            shown |= set(picked.song_key)
            if listens is not None:
                new = replayed_ratings(picked, liked, dislikes, start - pd.Timedelta(1, unit="s"))
                ratings = pd.concat([ratings, new], ignore_index=True) if len(ratings) else new
        print(f"    {name:<32}{'  '.join(str(h) for h in hits):<28}{sum(hits[1:]):>10}{sum(hits):>6}")


def together_report(library: Library, start: pd.Timestamp, end: pd.Timestamp, rng: np.random.Generator) -> None:
    """The songs-played-together task (sessions.py) for one window, with 95% ranges over sessions."""
    sessions, alike = together_test(library, start, end, rng)
    print(
        f"  together: {sessions.session.nunique():,} sessions with shuffle on for under {SHUFFLED:.0%} of the plays; "
        "seed: a song of the session you knew, relevant: its other songs"
    )
    if sessions.empty:
        print("    Note: no such sessions in this window")
        return
    print(f"    {'ranked by':<48}{'AUC (95% range)':<20}{'P@10':>6}{'lists alike':>13}")
    for name, result in sessions.groupby("ranking", sort=False):
        values = result.auc.to_numpy()
        means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(BOOTSTRAP_RUNS)]
        low, high = np.percentile(means, [2.5, 97.5])
        print(f"    {name:<48}{values.mean():.2f} ({low:.2f}-{high:.2f}){'':<4}{result['p@10'].mean():>6.2f}{alike[name]:>13.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", choices=["valid", "test", "both"], default="both")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    try:
        library = load_library()
    except (FileNotFoundError, RuntimeError) as error:
        sys.exit(str(error))
    starts = library.split_starts
    windows = {
        "valid": (starts["valid"], starts["test"]),
        "test": (starts["test"], library.ends_at),
    }

    # The learned rediscover models: one set of examples, and per window only the cutoffs it may learn from. A
    # history too short to learn from before a window leaves them out of that window.
    all_examples = examples(library, monthly_cutoffs(library, library.ends_at))
    regularization = choose_regularization(library, all_examples)[0] if learnable(all_examples, starts["valid"]) else None

    rng = np.random.default_rng(SEED)
    for name, (start, stop) in windows.items():
        if args.window not in (name, "both"):
            continue
        print(f"\n{name} window: {start:%d %b %Y} to {stop:%d %b %Y}, profile built from plays and likes before {start:%d %b %Y}")
        try:
            learned = {}
            if regularization is not None and learnable(all_examples, start):
                learned = window_chances(library, start, all_examples, regularization)
            else:
                print("  Note: too little history before this window to learn the rediscover model, so it is left out")
            tasks = [new_songs_task(library, start, stop), new_songs_task(library, start, stop, explicit=True),
                     returns_task(library, start, stop, learned)]
            if (library.candidates.pool == name).any():
                tasks.insert(0, discover_task(library, name, start, stop))
            else:
                print(f"  Note: no {name} pool in candidates.csv, so no discover test (run scripts/fetch_candidates.py)")
        except NoProfile as error:
            sys.exit(str(error))
        for task in tasks:
            report(task, rng)
        together_report(library, start, stop, rng)
        if (library.candidates.pool == name).any():
            feedback_replay(library, name, start, stop)


if __name__ == "__main__":
    main()
