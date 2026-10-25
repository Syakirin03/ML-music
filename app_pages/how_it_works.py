"""The How it works page: what the website is for, how a list is made, how the machine learns, how well it
does, and how it could do better.

Figures said to be "on the author's data" come from docs/results.md, where every test is recorded: they are a
reference, next to the command that measures yours. The others (your ratings, the models learned from them, the
trained history model) are read now.
"""

import html

import altair as alt
import pandas as pd
import streamlit as st

from app_pages import common, look
from recommender.feedback import MIN_EACH, MODEL_MARGIN
from recommender.profile import HALF_LIFE_DAYS, RATING_LISTENS
from recommender.ranker import FEATURES, HORIZON_DAYS, load_ranker, ranker_summary
from recommender.recommend import MAX_PER_ARTIST, SHORTLIST
from recommender.sessions import DIMENSIONS, WINDOW

SERIES = {"light": ("#2a78d6", "#e34948"), "dark": ("#3987e5", "#e66767")}
MUTED = "#898781"
# Learned weights shown per model: the largest by size.
TOP_WEIGHTS = 12
NAMES = {
    "genre_match": "Genres close to your taste", "artist_match": "Artist you play", "similar_match": "Similar to your artists",
    "asian": "Asian artist", "log_listeners": "Last.fm listeners", "log_rank": "Lower in the artist's charts",
    "song_match": "Played it a lot lately", "log_listens": "Listens in all", "log_days_dormant": "Days unplayed",
    "log_days_known": "Days since first played", "finish_rate": "Share finished", "skip_rate": "Share skipped",
    "artist_recent": "Artist played this month", "genre_recent": "Genre played this month", "has_genres": "Has genres",
}


def weight_name(feature: str) -> str:
    if feature.startswith("genre: "):
        return "Genre: " + look.genre_label(feature.removeprefix("genre: "))
    return NAMES.get(feature, feature)


def weights_chart(weights: pd.Series, kind: str, outcome: str = "a like") -> alt.Chart:
    """Diverging bars: blue makes the outcome likelier, red less likely."""
    up, down = SERIES[kind]
    likelier, less = f"Makes {outcome} likelier", f"Makes {outcome} less likely"
    frame = pd.DataFrame({"input": [weight_name(name) for name in weights.index], "weight": weights.to_numpy()})
    frame["direction"] = frame.weight.map(lambda value: likelier if value > 0 else less)
    bars = alt.Chart(frame).mark_bar(size=18, cornerRadiusEnd=4).encode(
        x=alt.X("weight:Q", title=f"Change in the log-odds of {outcome}"),
        y=alt.Y("input:N", sort=alt.EncodingSortField("weight", order="descending"), title=None,
                axis=alt.Axis(labelLimit=280, labelOverlap=False)),
        color=alt.Color("direction:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=[likelier, less], range=[up, down])),
        tooltip=[alt.Tooltip("input:N", title="Input"), alt.Tooltip("weight:Q", title="Weight", format="+.2f")],
    )
    zero = alt.Chart(pd.DataFrame({"weight": [0]})).mark_rule(color=MUTED, strokeWidth=1).encode(x="weight:Q")
    chart = (bars + zero).properties(height=30 * len(frame) + 50)
    return chart.configure(background="transparent").configure_view(strokeWidth=0)


kind = look.theme_kind() or "light"
rated = common.current()
models = common.learned_models(common.ratings_stamp(common.library().ratings))
ranker, trained = load_ranker(), ranker_summary()

look.hero(
    "How it works", "A recommender that learns from you",
    "This website recommends music from your own Spotify history: songs new to you, songs you loved and forgot, and "
    "songs like one you name. It runs only on your computer, and every like, neutral and dislike you give it "
    "teaches it something.",
    [(f"{len(rated):,}", "ratings so far"),
     ("2 of 2" if all(model["in_use"] for model in models.values()) else
      f"{sum(model['in_use'] for model in models.values())} of 2", "models learned from your ratings in use")],
)

st.header("What it is for")
st.markdown(
    "- **Find new music you will like**, without a streaming service deciding for you: songs you have never played, "
    "chosen for *your* taste, with dials for how mainstream, how familiar and which genres.\n"
    "- **Rediscover songs you loved** and stopped playing.\n"
    "- **Explore your own listening**: what you play, when, and how your taste moved (Your data).\n"
    "- **Learn how a recommender works**, end to end: data cleaning, features, models, honest tests."
)

st.header("How a list is made")
look.steps([
    ("Your taste profile.", "Every play counts as a listen (1 finished, 0 skipped early, else the share heard), "
                            f"counting half as much every {HALF_LIFE_DAYS} days. Your genres and artists get weights "
                            f"from those listens, and each like or dislike adds or takes away {RATING_LISTENS:g} listens."),
    ("Candidates.", "Songs new to you come from a pool built with Last.fm: the most played songs of your top artists, of "
                    "the artists similar to them, and of your genres' top artists. Songs to rediscover are songs with "
                    "at least 3 listens and none lately."),
    ("A score.", "Each candidate gets a score from parts between -1 and 1: how close its genres are to your taste, "
                 "whether you play its artist, how strongly your artists point to it on Last.fm, plus your dials "
                 "(Mainstream, More of these genres, Less Western)."),
    ("A learned model re-orders the best.", f"The {SHORTLIST} best by score are ordered by a model learned from your own "
                                            "ratings: your chance of liking each song. That chance is the big number on every card."),
    ("Variety.", "The list is built one song at a time, trading a little score for genres it doesn't have yet, with at "
                 f"most {MAX_PER_ARTIST} songs per artist."),
    ("You rate, it learns.", "Your choices are sent together once you have chosen the whole list or press Next songs: "
                             "likes and dislikes go into your profile and the next training of the models, and the next "
                             "list is built with them. Neutral only keeps the song off your lists."),
])

st.header("How the machine learns")
st.markdown(
    "Some parts are **learned from data** (trained models), some are **worked out by formula** from your plays, and "
    "a few are **set by hand** or chosen on held-out data. Telling them apart is the honest way to describe a system."
)
parts = pd.DataFrame([
    ("Taste profile (genre and artist weights)", "Worked out by formula", "Weighted averages of your listens; no training"),
    ("Score weights (familiarity 0.5, song share 0.5)", "Chosen on held-out data", "Picked on the valid window by rules fixed before looking"),
    ("Your chance of going back to a song", "Learned: logistic regression", "Trained on your history, month by month"),
    ("Your chance of liking a song new to you", "Learned: logistic regression", "Trained on your likes and dislikes"),
    ("Your chance of liking a song to rediscover", "Learned: logistic regression", "Trained on your likes and dislikes"),
    ("Songs you play together", "Learned without labels", "Co-occurrence in sessions, PPMI and SVD (song vectors)"),
    ("Mainstream, More of these genres, Less Western", "Set by you", "Your preferences; the tests leave them out"),
], columns=["Part", "How it is set", "From what"])
st.dataframe(parts, hide_index=True, width="stretch")

st.subheader("1. Learning from your history: will you go back to a song?")
if trained:
    st.markdown(
        f"A **supervised** problem: on the first day of every month since 2020 (a *cutoff*), take the songs you liked "
        f"but hadn't played for 90 days; the **label** is 1 if you played one again within {HORIZON_DAYS} days. That "
        f"gave {int(trained['cutoffs'])} cutoffs, {int(trained['examples']):,} examples and {int(trained['positives']):,} "
        f"returns. A **logistic regression** learns a weight for each of its {len(FEATURES)} features (how much you played "
        "the song, how long ago, how often you finished or skipped it, ...). It may only learn from the past: a model "
        "judged at a moment learns from cutoffs whose 90 days were over by then."
    )
else:
    st.markdown("Run `python -m recommender.ranker` to train it.")
if ranker is not None:
    weights = ranker.weights.weight.reindex(ranker.weights.weight.abs().sort_values(ascending=False).index)
    with st.container(key="panel-ranker"):
        st.caption("What it learned: per standard deviation more of each feature, the change in the log-odds of going back.")
        st.altair_chart(weights_chart(weights, kind, "going back"), width="stretch")
st.markdown(
    "How well it did on the author's data: checked month by month over 45 months, each month by a model trained "
    "only on the months before, it beat the hand-set formula in 32 of them, by 0.06 AUC on average, and on the test "
    "window its AUC was 0.82, against 0.73 for the formula. (AUC is the chance that a song gone back to is ranked "
    "above one that wasn't: 0.5 is guessing, 1 is perfect.) `python -m recommender.ranker` runs the same check on yours."
)

st.subheader("2. Learning from your ratings: will you like it?")
st.markdown(
    f"Every like and dislike is a labelled example. Two more logistic regressions learn from them: one for songs new to "
    f"you, one for songs to rediscover. Their inputs are the score's parts as they were when you saw the song, facts "
    f"about it (Asian or not, Last.fm listeners, chart position), and **its genres**, which turned out to matter most. "
    f"A model is only used once it has {MIN_EACH} likes and {MIN_EACH} dislikes, and only if, **cross-validated** (each "
    f"rating predicted by a model that didn't see it), it beats the score that listed the songs by at least "
    f"{MODEL_MARGIN:g} AUC: rules fixed before there were any ratings."
)
for mode, title in (("discover", "Songs new to you"), ("rediscover", "Songs to rediscover")):
    model = models[mode]
    with st.container(key=f"panel-model-{mode}"):
        st.markdown(f"**{title}**: {html.escape(model['message'])}.")
        if model["in_use"] and len(model["weights"]):
            st.caption(f"Its {TOP_WEIGHTS} largest weights. A number: per standard deviation more. A genre: for the rated "
                       "song with the most of it.")
            st.altair_chart(weights_chart(model["weights"].head(TOP_WEIGHTS), kind), width="stretch")
st.markdown(
    "They were also checked **forward in time**: trained on earlier ratings, predicting the later ones, as they are "
    "really used. On the author's ratings, songs new to you reached AUC 0.78 against 0.51 for the score (188 later "
    "ratings predicted), and songs to rediscover 0.67 against 0.57 (98). `python -m recommender.feedback --check` "
    "and `python -m recommender.ranker --ratings` run those checks on yours."
)

st.subheader("3. Learning without labels: songs you play together")
st.markdown(
    f"For **Songs like one you name**, songs get vectors from your sessions: two songs are alike when you play them "
    f"within {WINDOW} songs of each other more often than chance would (PPMI), and a truncated SVD squeezes that into "
    f"{DIMENSIONS} numbers per song. Nobody labels anything: the structure comes from your listening alone."
)

st.subheader("4. Testing it honestly")
st.markdown(
    "Your plays are split **in time order**: the oldest 80% to learn from, the next 10% (valid) to choose settings on, "
    "and the newest 10% (test) touched only to report. The test replays the past: it ranks the pool as of the test's "
    "first day and checks where the songs you went on to play ended up. On the author's data, songs new to you "
    "reached AUC 0.90, with 4.3 of the top 10 later played (P@10 0.43) and 53% of them in the top 100; "
    "`python -m recommender.evaluate` measures yours. Every number, and the rules for every choice, are in "
    "docs/results.md."
)

st.header("How it can get better")
look.steps([
    ("More ratings, spread wide.", "The models learn only from songs you rate, and those are the ones the score already "
                                   "liked. Rating songs of new artists and other genres teaches them the most."),
    ("A little exploration.", "Showing a few songs outside the top on purpose (a bandit) would teach the models what "
                              "the score misses, at the cost of some worse songs."),
    ("A fresh export.", "Spotify's next export brings your newest plays, and the songs you save to Liked Songs from "
                        "this website, which count as likes in your profile."),
    ("Richer song features.", "Audio features (tempo, energy, mood) or lyrics could tell songs of one genre apart. They "
                              "would mean sending your songs to another service, so they are left out for now."),
    ("Other people's taste.", "Big services learn from millions of listeners (collaborative filtering). With one "
                              "listener, Last.fm's similar artists stand in for that."),
    ("Better genres.", "Last.fm tags names, not songs, so artists sharing a name can get the wrong genre."),
])
look.footer()
