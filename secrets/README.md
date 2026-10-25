# secrets/

Your API keys and your Spotify login live here, on your computer only. Git ignores everything in this folder except this file. Never paste a key anywhere public, and never print one in a screenshot or an issue.

| File | What it holds | Where it comes from | Needed for |
|---|---|---|---|
| `lastfm_api_key.txt` | Your Last.fm API key, on one line. | [Create a Last.fm API account](https://www.last.fm/api/account/create), free. | Genres and the pool of songs new to you: required. |
| `spotify_client.txt` | Your Spotify app's Client ID on the first line and its Client secret on the second. | The [Spotify developer dashboard](https://developer.spotify.com/dashboard): steps in [chapter 2](../docs/2-set-up.md#4-optional-a-spotify-app). | The website's players for songs new to you, and saving to your Spotify: optional. |
| `spotify_account.json` | Your Spotify login: tokens that let the website save songs for you. | **Connect Spotify** on the website's Settings page. | Saving to Liked Songs and playlists: optional. |
| `spotify_login.json` | A random check code, while a login is under way (15 minutes at most). | The website. | Nothing else. |

Instead of the two key files, you can set the environment variables `LASTFM_API_KEY`, `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`.

**If a key gets out:** on Last.fm, create a new API account and use its key; on the Spotify dashboard, open your app's settings and make a new client secret. **Disconnect** on the Settings page deletes `spotify_account.json`, and [your Spotify account's Manage apps page](https://www.spotify.com/account/apps/) takes the website's access back on Spotify's side.
