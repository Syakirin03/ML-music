# 2. Set up

This chapter installs the project, gets the keys it needs, and writes your personal settings. Keys and settings live in files that git ignores, so the code you share never contains them.

## 1. Get the code and Python

You need [Python 3.13](https://www.python.org/downloads/). Then:

```bash
git clone https://github.com/Syakirin03/ML-music.git
cd ML-music
```

Work in a virtual environment, so the exact versions this project needs don't clash with anything else on your computer:

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins every version the project was built and tested with. DuckDB in particular stays at 1.5.6: [the data reference](data-reference.md#duckdb-notes) explains which of its limits shaped the design.

Run every command in this tutorial from the project folder, with the environment active. On Windows, if names in Japanese, Korean or Arabic make Python print `UnicodeEncodeError`, add `-X utf8` (`python -X utf8 scripts/fetch_genres.py`), or set the environment variable `PYTHONUTF8=1`.

## 2. A Last.fm API key (required)

Spotify's export has no genres, and Spotify closed its own genre and audio-feature data to new apps. Last.fm has the next best thing: tags that its listeners give songs, albums and artists, and lists of artists similar to each other.

1. Log in to [Last.fm](https://www.last.fm), or make a free account.
2. Open [Create API account](https://www.last.fm/api/account/create). Any application name and description will do; leave the callback URL empty.
3. Copy the **API key** (this project doesn't need the shared secret).
4. Save it as the only line of `secrets/lastfm_api_key.txt`, or set the environment variable `LASTFM_API_KEY`.

The scripts send Last.fm at most 4 requests a second (it allows 5), and keep every answer in `data/interim/lastfm_cache.jsonl`, so nothing is ever asked twice.

## 3. Your settings

Copy the example and open the copy:

```bash
cp config.example.toml config.toml     # Windows: copy config.example.toml config.toml
```

Set at least your time zone, as a name from the [tz database](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones):

```toml
[home]
timezone = "Europe/London"
```

Every play's hour, weekday and date use it. Left empty, it's your computer's time zone, and the build says so. The other settings are optional and can wait until [chapter 6](6-make-it-yours.md), which explains each:

- `market`, your Spotify country for the website's song search;
- `[genres]`, genres your music has that the genre list lacks;
- `[not_music]`, tracks to leave out of your taste: white noise, audiobooks, recitations;
- `[pool]`, favourite artists whose scene you want more of;
- `[taste]`, genres you want more of, and a "less Western" dial;
- `[privacy]`, words the privacy check may let through.

`config.toml` is ignored by git. Only `config.example.toml`, with the defaults, is shared.

## 4. Optional: a Spotify app

Without one, everything works except two things on the website: Spotify's player for songs new to you (the cards link to a Spotify search instead), and saving songs to your Spotify. Songs from your own history play either way, because the export has their ids.

1. Log in at the [Spotify developer dashboard](https://developer.spotify.com/dashboard) and choose **Create app**.
2. Give it any name and description. As **Redirect URI**, enter exactly `http://127.0.0.1:8501`, tick **Web API**, and save. (Spotify doesn't accept `localhost` here, only `127.0.0.1`.)
3. In the app's **Settings**, copy the **Client ID**, and the **Client secret** under *View client secret*.
4. Save them in `secrets/spotify_client.txt`: the ID on the first line, the secret on the second. Or set `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`.
5. Check it with any song: `python -m recommender.spotify "Artist name" "Song title"`.

Two rules of Spotify's for apps in development mode, since February 2026: the app's owner needs Spotify Premium, and only users added under **User Management** in the app's settings may log in to it, five at most. Add your own Spotify account there before you connect the website to it ([chapter 5](5-the-website.md)).

## 5. For working on the code

To change the code and commit it, install the development tools and switch on the safety nets:

```bash
pip install -r requirements-dev.txt
git config core.hooksPath .githooks   # before every commit: the privacy checks of tests/test_privacy.py
nbstripout --install                  # notebook outputs are stripped as they are committed
pytest                                # all tests, on made-up data; they never touch yours
```

[docs/privacy.md](privacy.md) explains what each one protects.

Next: [3. Build the database](3-build-the-database.md).
