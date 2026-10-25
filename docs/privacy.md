# Privacy and security

A Spotify export is personal in ways that aren't obvious: it says where you were, when you slept, and what you listen to when no one is watching. This project runs entirely on your computer, and the repository is set up so that your data, keys and settings can't be committed by accident. This page lists what is personal, how each layer of protection works, and exactly what leaves your computer.

## What is personal here

| Where | What it holds | Kept out of git by |
|---|---|---|
| `data/raw/` | every play with its time, the IP addresses and devices you listened from, and in the technical log your email address | `.gitignore` |
| `data/processed/spotify.duckdb` | the same, cleaned and linked | `.gitignore` (and `*.duckdb` anywhere) |
| `data/processed/*.csv` | the genres of your tracks, and a pool built from your top artists | `.gitignore` |
| `data/interim/*.jsonl` | every track and artist the scripts asked Last.fm and Spotify about | `.gitignore` |
| `data/user/` | your ratings, taste profile and learned models, every list shown, the website's settings and wallpaper | `.gitignore` |
| `config.toml` | your time zone (roughly where you live), favourite artists and tastes | `.gitignore` |
| `secrets/` | your Last.fm key, your Spotify app's secret, your Spotify login | `.gitignore`; the login file is readable by you alone |
| notebook outputs | charts and tables of your listening | stripped as they are committed (`nbstripout`) |
| `reports/` | your own results and notes | `.gitignore` |

Your listening times deserve a special mention. The *Your data* page and the notebook chart when you listen, by hour and weekday: your daily routine. Keep such charts, and screenshots of them, to yourself.

## How the repository keeps it out

1. **`.gitignore`** excludes every folder and file above. Only `README.md` and `.gitkeep` placeholders inside them are committed.
2. **Notebooks are stripped on commit.** `.gitattributes` sends every `.ipynb` file through [nbstripout](https://github.com/kynan/nbstripout) as it is added. Your own copy keeps its outputs, and the committed copy has none. Turn it on in each clone with `nbstripout --install`.
3. **A privacy check runs before every commit.** `.githooks/pre-commit` runs `tests/test_privacy.py` on what git is about to commit, and stops the commit if it finds:
   - a file of yours (anything under `data/` except the placeholders, `secrets/`, `config.toml`, `reports/`, a database, a cache);
   - a notebook with outputs;
   - one of your keys or tokens inside any file;
   - a value of your `config.toml` (your time zone, favourite artists, not-music rules);
   - the name of any artist you've played 20 times or more.

   Turn it on in each clone with `git config core.hooksPath .githooks`. When the last check stops a commit over an ordinary word that happens to be one of your artists' names, add the word to `[privacy] allow` in `config.toml`.
4. **The tests use made-up data.** `tests/conftest.py` points the recommender at the default settings and an empty scratch folder before anything runs, so no test reads your settings or writes among your ratings. The pipeline tests build their databases from invented exports.
5. **GitHub Actions** runs the linter, `nbstripout --verify` and the tests on every push. On GitHub's servers your files don't exist, so the checks that compare with them skip themselves. The others still catch a tracked personal file or a notebook with outputs.

## What leaves your computer

Only these, and only when you run the step named:

| To | What it receives | When |
|---|---|---|
| Last.fm | the names of tracks, albums and artists in your history, and of your top artists and the artists similar to them, with your API key | `fetch_genres.py`, `fetch_candidates.py` |
| Spotify, as your app | the title and artist of each song new to you that the website shows, once per song (the answer is cached) | the website, if you set up an app |
| Spotify, in your browser | which songs each page shows, as the embedded players load (and who you are, if you're logged in to Spotify in that browser) | the website |
| Spotify, as you | which of the songs shown are in your Liked Songs, your playlists' names, and the songs you save or add | the website, if you connect your account |

Neither service receives your listening times. Last.fm never sees anything of your Spotify account. Streamlit's usage statistics are off (`gatherUsageStats = false`), and the website loads no fonts or scripts from elsewhere except Spotify's players.

**Spotify's terms** forbid training machine-learning models on content from its Web API. The models here learn from *your own export*, your personal data. The track ids that Spotify's search returns are used only to play and save songs, never as features.

## Security

- **The website only listens on your computer** (`address = "localhost"` in `.streamlit/config.toml`). It shows your listening and changes your ratings and your Spotify library, so don't expose it to a network, and don't deploy it.
- **The Spotify login checks a random `state` value** when Spotify sends you back, and accepts a given one for 15 minutes at most. A link someone else made can't log the website in to their account.
- **The login file is readable by you alone.** On macOS and Linux it gets permissions 600. On Windows it gets its folder's, which inside your user folder means only you and administrators: keep the project there.
- **Text from the data is escaped** before it goes into a page's HTML (song titles, artists, genres), so a title made up as HTML or script stays text. Track ids are checked against Spotify's format before they go into a player.
- **No key is in the code.** Keys are read from `secrets/` or from environment variables (`LASTFM_API_KEY`, `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`), and never printed.

## Before you publish your own version

1. Turn on the safety nets once per clone: `git config core.hooksPath .githooks` and `nbstripout --install`.
2. Run `git status` and read the list: only code and docs, nothing under `data/` but the placeholders, no `secrets/`, `reports/` or `config.toml`.
3. Run `pytest tests/test_privacy.py`, which the hook runs anyway, and `nbstripout --verify notebooks/*.ipynb`.
4. Never `git add -f` an ignored file, and don't remove lines from `.gitignore`.
5. Check what your commits say about you. They carry the name and email of `git config user.name` and `user.email`. GitHub gives you a private address for commits (Settings → Emails → *Keep my email addresses private*).
6. Blur screenshots: song titles, artists, ratings, and every chart of your listening.

## If something got out

- **A key:** on Last.fm, create a new API account and use its key; on the Spotify dashboard, make a new client secret for your app.
- **Your Spotify login:** remove the website's access on your Spotify account's [Manage apps](https://www.spotify.com/account/apps/) page, then delete `secrets/spotify_account.json`.
- **A file in a pushed commit:** deleting it in a new commit isn't enough, because the history keeps it. Remove it from the history (GitHub's guide: [removing sensitive data from a repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)), and assume that whatever was public has been copied.
