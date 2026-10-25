# Results on the author's data

What the recommender achieved on one listener's history, about 90,000 plays of about 4,200 songs over several years, measured on 25 October 2026. It's here as a reference, and to show the method: each setting with the rule that chose it, and the ideas that were tried and rejected. Your numbers will differ, and `python -m recommender.evaluate` measures them. Nothing here names a song, artist or genre the author plays.

How to read it: **AUC** is the chance that a relevant song ranks above an irrelevant one (0.5 is a coin flip, 1 is perfect). **P@10** is the share of relevant songs in the top 10, and **R@100** the share of all relevant songs that made the top 100. The **valid** window, about four months, is where settings were chosen. The **test** window, the last six months, was only looked at to report. Ranges are 95% bootstrap ranges.

## Songs new to you

The pool as it was at the start of the test window: 31,106 songs not played before. Relevant: the 73 that were played during the window.

| Ranked by | AUC | P@10 | R@100 |
|---|---|---|---|
| random | 0.50 | 0.00 | 0.00 |
| popularity (Last.fm listeners) | 0.79 | 0.10 | 0.01 |
| `genre_match` | 0.77 | 0.00 | 0.01 |
| `artist_match` | 0.85 | 0.43 | 0.08 |
| `similar_match` | 0.84 | 0.02 | 0.14 |
| **the score** (familiarity 0.5) | **0.90** (0.86–0.94) | **0.43** | **0.53** |

- **The valid window was hard.** AUC 0.72, and none of the 92 songs later played reached the top 10 (3 made the top 100). What was found then lay where the earlier taste didn't point, so test results alone would flatter the recommender.
- **The list you see** has at most 2 songs per artist and some variety. It held 2 songs later played in each window, with or without variety. Variety was chosen when ranking by score alone gave lists of a single main genre; with popularity at 0, the lists are varied already (diversity, 1 minus the average genre similarity of their pairs: 0.33 in valid, 0.68 in test).
- **The pool is the bigger limit.** Only 92 of the 324 songs first played in the valid window, and 73 of 251 in the test window, were in the pool at all.

## Songs to rediscover

At the start of the test window, 552 liked songs hadn't been played for 90 days. Relevant: the 71 that were played again during the window.

| Ranked by | Test AUC | Test P@10 | Valid AUC | Valid P@10 |
|---|---|---|---|---|
| random | 0.50 | 0.13 | 0.50 | 0.09 |
| past listens alone | 0.77 | 0.90 | 0.70 | 0.30 |
| the hand-set formula | 0.73 | 0.50 | 0.80 | 0.50 |
| **logistic regression** | **0.82** | **0.80** | 0.76 | 0.40 |
| gradient-boosted trees | 0.80 | 0.80 | 0.71 | 0.30 |

The two windows disagree, and with 71 and 43 relevant songs their ranges overlap. So the rule set beforehand, *replace the formula only if the model beats it in both windows*, wasn't met. The month-by-month check decided it, on 45 months that neither had been tuned on, each judged by a model trained only on earlier months:

| Month by month, 45 months, 1,642 songs gone back to | Mean AUC | Against the formula |
|---|---|---|
| the hand-set formula | 0.634 | |
| **logistic regression** | **0.694** | +0.060 ± 0.030, better in 32 of 45 months |
| gradient-boosted trees | 0.656 | +0.023 ± 0.033, better in 26 of 45 months |

What the model learned, per standard deviation more of a feature: the longer a song has gone unplayed, the less likely a return (−0.41). The more it was played lately and in all, the more likely (+0.35 and +0.32). Songs known longer come back a little more often (+0.19). Its artist and genres add almost nothing once those are known. Its regularisation, C = 0.01, was chosen on the valid window, where C hardly mattered (AUC 0.754 to 0.760).

Its chances are right in order but too low at the top lately: of the songs it gave a 20–40% chance, 49% were gone back to, against the 26% it predicted.

## Songs like one you name

Quarter by quarter, vectors learned only from earlier sessions; for each session in the next 90 days, how well a song of it ranks the session's other songs. Shuffled sessions are left out.

| Ranked by (23 quarters, 2,079 sessions) | Mean AUC | P@10 | Lists alike |
|---|---|---|---|
| how much it was played lately | 0.862 | 0.123 | 98% |
| genres | 0.783 | 0.052 | 7% |
| session vectors | 0.764 | 0.096 | 4% |
| **0.5 session + 0.5 genre + 0.1 lately** | **0.824** | **0.104** | 5% |

"Lists alike" is the share of one seed's top 10 that is also in another seed's. Ranking by how much a song is played lately wins AUC, but it gives every song the same list: favourites, not songs like the one named. The blend was the best of the rankings that give each song its own list: against genres alone, AUC +0.042 ± 0.020 (better in 21 of 23 quarters) and twice the P@10 (better in 22). In the test window (166 sessions) the blend's P@10 was 0.08 against 0.05 for genres, with AUC 0.79 against 0.80.

## Learning from ratings

**The adoption test.** With 227 ratings of songs new to you, four inputs were compared by a rule fixed before running it. Ratings in time order, each block of 20 after the first 80 predicted by a model trained only on the ratings before it (147 ratings predicted). The best would be adopted only if it beat the hand-set score by 0.05, the simpler one when two were within 0.01.

| Model | AUC on later ratings | Against the score (95% range) |
|---|---|---|
| A: the hand-set score | 0.61 | |
| B: logistic regression on the score's 4 parts | 0.65 | +0.05 (−0.05 to +0.14) |
| C: B + Asian or not, Last.fm listeners, rank | 0.66 | +0.05 (−0.05 to +0.16) |
| **D: C + genres** | **0.68** | +0.07 (−0.03 to +0.17) |
| E: gradient-boosted trees on C | 0.62 | +0.01 (−0.11 to +0.14) |

D won (C was 0.014 behind, outside the tie band). By a second rule fixed beforehand, popularity was then dropped, since its pool-relative value shifts whenever the pool changes: without it, D scored 0.673 against 0.677. With a few hundred examples, a straight line generalised better than trees, as it did for the rediscover model.

**Since then.** With 268 ratings of songs new to you, the forward check (188 ratings predicted) gives the model AUC 0.78 against 0.51 for the score: +0.27, range +0.17 to +0.37. Cross-validated, 0.77 against 0.50.

**Songs to rediscover.** With 158 ratings, the same test (98 predicted, after the first 60): the chance the songs were listed by, 0.57; logistic regression on it and the score's parts, 0.56; on the history model's 11 inputs, 0.56; **with genres, 0.67** (+0.10, range −0.06 to +0.25), which is in use (cross-validated 0.74 against 0.58). Only the genres taught the model anything new.

**What the first ratings showed.** Every "Next songs" page offered the same few artists: six pages of ten held just 8 of them, because every unplayed song of an artist you play gets that artist's boost. Two fixes followed. Artists now take turns, so six pages hold about 30. Three strikes (three dislikes and no like) stops an artist: replayed on the ratings in time order, it would have spared 7 dislikes and cost 1 like. Artists never played weren't liked more often than unplayed songs of familiar artists (28% of 99 against 34% of 110, a difference of −5 points, range −18 to +7), so familiarity stayed at 0.5.

## Settings, and the rules that chose them

| Setting | Value | Chosen by |
|---|---|---|
| `FAMILIARITY` | 0.5 | best valid-window AUC among 0.25, 0.5 and 0.75. Popularity alone scored higher (0.85) but was left out of the choice in advance, since offline tests over-reward it |
| `SONG_SHARE` | 0.5 | adding `song_match` to the rediscover formula raised its valid AUC from 0.75 to 0.80 |
| `POPULARITY_WEIGHT` | 0 | first 0.5, by valid AUC; set to 0 for less mainstream lists (table below) |
| `VARIETY` | 0.5 discover, 0 rediscover | the largest of 0, 0.1, 0.2, 0.3 and 0.5 that lost no later-played song from the valid top 10 and top 30. For rediscover lists, 0.5 would have cost 2 of 4 hits in valid and 2 of 8 in test |
| `RATING_LISTENS` | 5 | of 5, 20 and 50 listens per rating, the one whose replay found the most in lists 2 to 5 of the valid window, the smallest if tied. All tied |
| `STRIKES` | 3 | replayed on the first 46 ratings: spared 7 dislikes, cost 1 like |
| `MIN_EACH`, `MODEL_MARGIN` | 15, 0.05 | fixed before there were any ratings |
| `LIKE_SESSION`, `LIKE_GENRE`, `LIKE_ROTATION` | 0.5, 0.5, 0.1 | the quarterly check above |
| genre artists in the pool | 20 genres × 20 artists | more of the valid window's finds in the pool (87 to 92) without fewer in the top 100 (1 against 1); in test, 64 to 73 |

The popularity weight, measured when it was changed:

| Popularity weight | Test: finds in top 10 / top 100 | Valid: top 10 / top 100 | Median Last.fm listeners of the top 10 |
|---|---|---|---|
| 0.5 | 5 / 17 | 0 / 1 | 815,000 |
| 0.25 | 6 / 26 | 0 / 1 | 573,000 |
| **0** | 3 / 39 | 0 / 3 | 51,000 |
| −0.25 | 0 / 6 | 0 / 2 | almost none |

At 0 the top 100 held more than twice as many later finds in both windows, at the cost of AUC (0.94 to 0.90 in test, 0.85 to 0.72 in valid), which counts the order far down the pool too.

## What didn't work

Each was a one-off check, judged by a rule fixed before looking, and none is in the code:

- **Learned song weights instead of the formula's.** A logistic regression on six signals predicted likes about as well as the hand-set weight: AUC 0.88 against 0.87 (+0.01, range −0.02 to +0.04). A regression of the next 90 days' listens matched it month by month (Spearman 0.570 against 0.565, better in 24 of 45 months). The formula stayed.
- **A learned discover score.** A logistic regression fitted on the valid pool reached AUC 0.88 on test, less than the formula. With 87 positives, it learned the valid period's quirks.
- **A session feature for the rediscover model.** How close a dormant song is to what was played lately: month by month 0.692 and 0.691 against 0.694, better in only 17 and 11 of 45 months.
- **Several taste profiles.** Songs clustered into 3 to 8 groups by k-means on their session vectors, each group with its own genre profile. Valid AUC rose from 0.80 to 0.82, within the noise, but the test top 10 became a block of one genre (P@10 0.50 to 0.00). The vectors have no natural number of groups either: the silhouette of the favourite songs rises steadily from 0.30 for 2 groups to 0.54 for 10.
- **Predicting which new song will stick.** Among songs just heard for the first time, nothing ranked better than chance: the score's parts reached AUC 0.42 to 0.54, and a logistic regression on all earlier first listens 0.52 against 0.49 for random, over 2,894 new tracks. Last.fm's song-level numbers did little better (0.53 to 0.56). Which song you'll love has to be learned from your reactions, which is what the ratings are for.
- **Audio features.** A free audio-feature service, tested on 150 songs: it covered 80% of the songs and 92% of the plays, but told apart songs played in the same session no better than chance (AUC 0.51, against 0.54 for genres). It would also mean sending track ids to another service. Not added.
- **An artist's earlier ratings in the score.** They predicted the next rating of that artist only a little (AUC 0.68 over 39 ratings), and moved the score's AUC from 0.71 to 0.73, which is noise at that size.
