# 4. The recommender

The `recommender/` package reads the gold layer and recommends three kinds of songs. This chapter explains how each is made: which parts are worked out by formula, which are learned from data, and how every choice was tested. The [results page](results.md) has the numbers on the author's data. The commands at the end measure yours.

## From plays to songs

The recommender works with **songs**, not Spotify's track ids. Played tracks with the same artist and the same *plain* title are one song, the plain title being the title without what Spotify adds to it, such as " - Remastered 2011", " (feat. ...)", " - Japanese Ver." or " - Sped Up". The most played version gives the song its id and name. Artists are told apart by their exact name, letter case included. Tracks that `[not_music]` in `config.toml` marks as not music are left out of everything that follows.

Every play becomes a **listen**: 1 if you finished it, 0 if you skipped it within 30 seconds, and otherwise the share of it you heard. An early skip also counts half a listen *against* the song. If you have the technical log, a song in your Liked Songs counts 5 extra listens and each playlist you added it to counts 3. Your ratings on the website count 5 listens for a like and minus 5 for a dislike.

## Your taste profile (worked out, not learned)

`python -m recommender.profile` turns those listens into a profile, saved in `data/user/recommender.duckdb`:

1. **A weight per song.** Its evidence is its listens minus its skips, plus the bonuses, with every play counting half as much for every 180 days of age, so the profile follows your taste as it moves. The weight is the *log* of that, so the few songs you play hundreds of times don't drown out the rest.
2. **A weight per genre**: the average genre row of your songs, each counted with its song's weight. A song's genre row gives its main genre 1, its second 1/2 and its third 1/3, scaled to length 1 (*weighted multi-hot* encoding).
3. **A weight per artist**: the log of all the evidence for the artist's songs, scaled so that your top artist has 1. An artist you skip far more than you play goes toward -1.

The profile is rebuilt by itself whenever your history or your ratings are newer than it.

## Three kinds of recommendations

### Songs new to you

Candidates are the songs of the pool (chapter 3) that you've never played. Each gets parts that run from -1 to 1:

| Part | What it measures |
|---|---|
| `genre_match` | the cosine between the song's genre row and your genre weights |
| `artist_match` | your weight for its artist: 0 for an artist you've never played |
| `similar_match` | how strongly the artists you like point to its artist on Last.fm: your weight for each of them times Last.fm's match, summed and scaled |
| `popularity` | the share of the pool with fewer Last.fm listeners |
| `genre_pick` | 1 if one of the genres you asked for more of is its main genre, 1/2 if it's its second, and so on |
| `western` | 1 if its artist doesn't make Asian music, by its Last.fm tags or the script of its name |

and the score

```
score = (1 − familiarity) × (genre_match + similar_match) / 2 + familiarity × artist_match
        + mainstream × popularity + genre_boost × genre_pick − less_western × western
```

`familiarity` (0.5) weighs knowing the artist against how the song fits your genres and your artists' network. The other three are your dials, on the website and in `config.toml`: `mainstream` is 0 by default, and the genres and the "less Western" push are off until you set them.

### Songs to rediscover

Candidates are songs you gave at least 3 listens but haven't played for 90 days. A hand-set formula could rank them:

```
formula = 0.5 × song_match + 0.5 × ((1 − familiarity) × genre_match + familiarity × artist_match)
```

where `song_match` is the song's own weight over your top song's. But this is a question your history can answer directly. **Will you go back to a song?** is a supervised learning problem, and `python -m recommender.ranker` learns it:

- **Examples.** On the first day of every month, starting two years after your first play (a *cutoff*), take the songs rediscovery would have offered that day.
- **Label.** 1 if you played the song again within 90 days of the cutoff, else 0.
- **Features.** Eleven numbers known on that day, from earlier plays only: the formula's three parts, how much you ever played the song, the days since you last and first played it, how often you finished it and how often you skipped it, how much you played its artist and its main genre in the 30 days before, and whether it has genres.
- **Model.** A logistic regression on the standardised features (mean 0, standard deviation 1), so that its weights can be compared. Its penalty on large weights, `C`, is chosen on the valid window from 0.01, 0.1, 1 and 10. Gradient-boosted trees are trained alongside, for comparison only.
- **No peeking.** A model judged at a moment learns only from cutoffs whose 90 days had ended by then.
- **A rolling check.** Two test windows can disagree by chance, so the command also judges the model month by month, from four years after your first play until the valid window, each month by a model trained only on the months before. On the author's data this check decided that the model, not the formula, ranks songs to rediscover.

The model is saved as plain numbers (a weight, mean and scale per feature), not as a pickle, and recommendations use it from then on. Until it's trained, or if your history is too short, which the command tells you, the formula ranks instead.

### Songs like one you name

Genres say how Last.fm tags a song. Your sessions say what you actually play with it. `recommender/sessions.py` learns a vector per song from co-occurrence, the idea behind word2vec:

1. Count how often each pair of songs was played within 5 plays of each other in a session, leaving out early skips.
2. Turn the counts into **PPMI**, positive pointwise mutual information: how much more often the pair was played together than chance would give for songs played this much. That takes popularity out.
3. Compress the table to 64 numbers per song with a **truncated SVD**, and scale each row to length 1. Songs played in the same company get similar vectors, even if you never played them next to each other.

Songs like one you name are ranked by `0.5 × session similarity + 0.5 × genre similarity + 0.1 × song_match`. That blend was chosen on a quarterly check (`python -m recommender.sessions`), which learns vectors only from earlier sessions and leaves out shuffled sessions, since what comes next in a shuffle doesn't depend on the song before it. Nobody labels anything: this part is **unsupervised**.

## Making a list

A ranking isn't a list yet:

- **At most 2 songs per artist.**
- **Variety.** Ranked by score alone, a list is often all one genre. So the list is built one song at a time, *maximal marginal relevance*: each next song is the one with the best `(1 − variety) × score − variety × likeness`, where likeness is its genre similarity to the songs already listed. Variety is 0.5 for songs new to you, and 0 for songs to rediscover, whose lists are varied already.
- **Three strikes.** An artist with 3 dislikes among songs new to you, and no like, stops being offered.
- **Turns.** On the website, *Next songs* rests an artist once 2 of their songs have been shown, unless you liked one, so the next lists reach further.

## Learning from your ratings

Your history says well *what kind* of music you like, but not *which song* you'll love once you've heard it (the [results](results.md) show nothing in the history predicts that). So the website asks: **Like**, **Neutral** or **Dislike** for every song. Every rating is kept for good in `data/user/recommender.duckdb`, with a copy in `ratings.csv`. A like or dislike does three things, and a neutral only the first:

1. **The song leaves your lists.**
2. **It counts in your profile**, as 5 listens for or against.
3. **It becomes a labelled example** for two more logistic regressions, one for songs new to you and one for songs to rediscover.

Their inputs are the score's parts *as they were when you saw the song* (kept with every list shown), facts about the song (whether it's Asian, its Last.fm listeners, its rank among its artist's songs), and one column per genre that at least 5 of your rated songs have. A model is used only once it has **15 likes and 15 dislikes**, and only if, **cross-validated**, it beats the score that listed the songs by at least **0.05 AUC**. Cross-validated means each rating is predicted by a model that didn't see it. Both rules were fixed before there were any ratings, so they can't be bent to fit them. Once in use, a model reorders the 300 best songs by your chance of liking each, and that chance is the big number on every card.

A **forward check** tests them the way they're really used: ratings in time order, each block of 20 predicted by a model trained only on the ratings before it (`python -m recommender.feedback --check` for songs new to you, `python -m recommender.ranker --ratings` for songs to rediscover).

## What's learned and what's set

| Part | How it's set |
|---|---|
| Taste profile (genre and artist weights) | worked out from your listens by formula, no training |
| Score weights such as familiarity | chosen on the valid window by rules fixed before looking |
| Your chance of going back to a song | **learned**: logistic regression on your history |
| Your chance of liking a song | **learned**: two logistic regressions on your ratings |
| Songs you play together | **learned without labels**: PPMI and SVD |
| Mainstream, more of these genres, less Western | set by you; the tests leave them out |

## Testing it honestly

`python -m recommender.evaluate` replays the past. For the valid and the test window in turn, it builds your profile only from what you played before the window began, ranks songs with it, and checks the ranking against what you actually did in the window:

| Task | Candidates | Relevant |
|---|---|---|
| discover | the pool as it was at the window's start, minus songs played before | played in the window |
| returns | dormant favourites at the start | played again in the window |
| new songs | songs first played in the window | stuck: 3 or more listens |
| together | for sessions in the window, a song you knew as the seed | the session's other songs |

It reports four measures, with 95% ranges from bootstrap resampling:

- **AUC**: the chance that a relevant song is ranked above an irrelevant one. 0.5 is a coin flip.
- **P@10**: the share of relevant songs in the top 10.
- **NDCG@10**: the same, counting higher places more.
- **R@100**: the share of all relevant songs that made the top 100.

It also scores the list you'd actually be shown, with the artist cap and variety, and replays rating five lists in a row.

The rules that keep this honest:

- **Choose on the valid window, report the test window.** Settings were picked on valid, by a rule written down before looking. The test window, the newest 10% of plays, is the estimate you can trust.
- **Fix the rule before the run.** For example: replace the formula only if the model beats it in both windows, and keep a feature only if the month-by-month check improves. When a rule wasn't met, the change wasn't made. [results.md](results.md) records each rule and what happened.
- **Leave popularity out of the choices.** Popular songs reach you more often, so an offline test over-rewards recommending them.
- **Judge on many periods.** With a few dozen relevant songs per window, differences under about 0.07 AUC are noise. That's why the rolling and quarterly checks exist.

## Commands

```bash
python -m recommender.profile                       # build and show your taste profile
python -m recommender.profile --as-of 2021-06-30    # your taste back then (shown, not saved)
python -m recommender.recommend                     # 10 songs new to you
python -m recommender.recommend --new-artists       # only by artists you've never played
python -m recommender.recommend --familiarity 0.2   # rely less on knowing the artist
python -m recommender.recommend --variety 0         # the best songs, without trading score for variety
python -m recommender.ranker                        # train the rediscover model (about a minute)
python -m recommender.recommend --rediscover        # songs to rediscover, by that model
python -m recommender.recommend --rediscover --formula
python -m recommender.recommend --like "a song you know"
python -m recommender.feedback like 1 3             # rate songs 1 and 3 of the last list; dislike, neutral, clear
python -m recommender.feedback --check              # your ratings, the model learned from them, its forward check
python -m recommender.ranker --ratings              # the same for songs to rediscover
python -m recommender.sessions                      # the quarterly check of songs like one you name (about 30 s)
python -m recommender.evaluate                      # the replay of both windows (about a minute)
```

The same from Python, in a notebook or script started in the project folder:

```python
from recommender.recommend import recommend_songs, similar_songs

recommend_songs(top_n=10)                     # a DataFrame: score, its parts, why, a link
recommend_songs(top_n=10, mode="rediscover")  # with the chance of going back
similar_songs("a song you know", top_n=10)
```

Next: [5. The website](5-the-website.md).
