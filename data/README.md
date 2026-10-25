# data/

Your listening history lives here, on your computer only. Git ignores everything in this folder except this file, `external/musicbrainz_genres.txt` and the `.gitkeep` files that keep the empty folders in place.

| Folder | What goes in it | Made by |
|---|---|---|
| `raw/` | Spotify's exports, unzipped as they come: `Spotify Extended Streaming History/` and, if you asked for it, `Spotify Technical Log Information/`. Never edit them. | you ([chapter 1](../docs/1-get-your-data.md)) |
| `external/` | Reference files from elsewhere: `musicbrainz_genres.txt`, MusicBrainz's list of genre names, and `techlog_docs.json`, Spotify's descriptions of its technical log, read from the ReadMe in that export. | `scripts/extract_techlog_docs.py` |
| `interim/` | `lastfm_cache.jsonl` and `spotify_search_cache.jsonl`: every answer Last.fm and Spotify gave, so that nothing is asked twice. | `fetch_genres.py`, `fetch_candidates.py`, the website |
| `processed/` | `spotify.duckdb`, the database (bronze, silver, techlog and gold layers), plus `track_genres.csv` and `candidates.csv`. | the scripts in `scripts/` |
| `user/` | `recommender.duckdb` (your taste profile, the learned models, every list shown and **your ratings**), `ratings.csv` (a copy of the ratings), and the website's `website.json` and `wallpaper.jpg`. | the recommender and the website |

Everything in `interim/` and `processed/` can be rebuilt from `raw/`. Your ratings in `user/` can't, so back that folder up.

The raw exports and `spotify.duckdb` hold your IP addresses, your email address and your device ids, besides what you listened to and when. Don't upload them anywhere, and don't take them out of `.gitignore`.
