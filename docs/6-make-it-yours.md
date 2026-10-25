# 6. Make it yours

You have a working recommender. This chapter is about changing it: your settings, safe experiments, telling whether a change helped, and ideas worth trying.

## Your settings

Everything personal is in `config.toml` (copied from `config.example.toml`, which explains each setting). After a change, rerun the step it names:

| Setting | What it does | Rerun |
|---|---|---|
| `[home] timezone` | the time zone of every play's local hour, weekday and date | `build_database.py`, then `build_gold.py` |
| `[home] market` | your Spotify country, for the website's song search | restart the website |
| `[genres] extra` | genres your music has that MusicBrainz's list lacks | `fetch_genres.py`, then `build_gold.py` and `fetch_candidates.py` |
| `[genres] aliases` | other names for a genre: `{ "dnb" = "drum and bass" }` | the same |
| `[not_music] genres`, `pattern` | tracks to leave out of your taste: white noise, audiobooks, recitations | `build_gold.py`, then `fetch_candidates.py` |
| `[pool] favourite_artists` | artists whose scene the pool follows two steps out | `fetch_candidates.py` |
| `[taste] more_genres`, `genre_boost` | genres to favour among songs new to you, and how much | restart the website |
| `[taste] less_western` | how much to push down Western artists among songs new to you | restart the website |
| `[privacy] allow` | names of your artists that are also ordinary words in the code, which the privacy check may let through | nothing |

A typo in a setting's name, or text where a list belongs, stops every script with a message that names it.

## Experiment without risk

Your ratings are the one thing nothing can rebuild. Try things on copies:

- **The recommender and the website** read and write your own files in `data/user/`. Set the environment variable `MUSIC_RECOMMENDER_USER_DIR` to a folder holding a copy of it, and every command and the website use that folder instead:

  ```bash
  cp -r data/user /tmp/try-user
  MUSIC_RECOMMENDER_USER_DIR=/tmp/try-user python -m recommender.recommend   # Windows (PowerShell): $env:MUSIC_RECOMMENDER_USER_DIR = "..."
  ```

- **The builds** take `--db` (another database file) and the fetch scripts take `--out` (another CSV), so you can try a change to the pipeline without replacing your real tables.
- **Settings** can come from another file: set `MUSIC_PREFERENCE_CONFIG` to its path.
- **Backups**: `data/user/ratings.csv` is rewritten after every rating, and `python -m recommender.feedback --restore` reads it back into a database that has lost them. To back everything up, copy `data/user/`.

## Did it help? Measure it honestly

Most ideas that sound good don't help, and with one person's data it's easy to fool yourself. The project's routine:

1. **Write the rule down before you run anything.** For example: "adopt the new feature if the valid window's AUC rises by more than its 95% range, without losing hits in the top 10".
2. **Choose on the valid window only.** `python -m recommender.evaluate --window valid`.
3. **Look at the test window once,** as the honest estimate.
4. **For anything learned, run the checks over many periods:** `python -m recommender.ranker` (month by month), `python -m recommender.sessions` (quarter by quarter), and `--check` / `--ratings` for the models learned from ratings. Two windows of a few dozen relevant songs disagree by chance, but forty-five months rarely do.
5. **Keep what the rule says,** even when you liked the idea, and note the result in `reports/` (git ignores it, since results describe your listening).

[results.md](results.md) shows the routine on the author's data, including the ideas it rejected.

## Ideas to try

Roughly from the most promising:

- **Rate widely.** The models only learn from what you rate, and the score decides what you see. Rating songs from new artists and other genres teaches them the most. The sidebar's dials and *Only artists I have never played* help.
- **Explore on purpose.** Showing a few songs from outside the top of the list, a *bandit* strategy, would teach the models what the score misses, at the cost of some worse songs.
- **A fresh export.** Spotify's next export brings your newest plays, and the songs you saved from the website, which count as likes.
- **Use the gold layer for new models.** It's ready for more than this project uses: skip prediction (`gold.plays.label`, with the point-in-time `prev_*` and `*_prior_*` features), next-track prediction (`gold.sessions.track_ids`, `gold.track_transitions`), or ranking what Spotify served you (`gold.exposures`, if you have the technical log).
- **Listening context.** You may like different music in the morning than at night: `local_hour`, `is_weekend` and `platform_family` are in `gold.plays`.
- **Richer song features.** Audio features (tempo, energy, mood) or lyrics could tell apart songs that share all their genres. Spotify closed its audio-features API to new apps in November 2024, and other services would receive your track ids, a privacy trade-off to decide on knowingly. On the author's data, a free audio-feature service covered 80% of the songs but told apart songs played together barely better than chance.
- **Other people's taste.** Big services learn from millions of listeners (collaborative filtering). With one listener, Last.fm's similar artists stand in for that; Last.fm's listener data could do more.
- **Better genres.** Last.fm tags names, not songs, so artists sharing a name can get each other's genres. Release dates, labels or MusicBrainz relationships could correct them.

## Publish your own version

If you share your fork, your listening must stay out of it. Before you push, go through [the checklist in privacy.md](privacy.md#before-you-publish-your-own-version). The pre-commit hook and the tests do most of it for you.
