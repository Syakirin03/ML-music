# 5. The website

The command line works, but rating songs is easier on cards with a player. `app.py` is a small website, built with [Streamlit](https://streamlit.io), that runs on your computer and nowhere else.

## Start it

```bash
streamlit run app.py
```

Then open **http://127.0.0.1:8501**. `localhost` works too, but Spotify sends you back to `127.0.0.1` after you connect your account, so use that. Stop the website with Ctrl+C in its terminal. Restart it after changing the code. After rebuilding the database or the pool, press **Reload history and pool** on the Settings page.

The first list takes a few seconds: it builds your taste profile if none is saved yet, and scores the pool.

## The five pages

**Recommendations.** Ten songs at a time, each a card with Spotify's player, the title and artist, the song's genres (the main one highlighted), your chance of liking it, and why it is there. The sidebar picks the kind of list (songs new to you, to rediscover, or like one you name) and its dials, and shows how many more ratings the learned models need.

- **Like, Neutral and Dislike only mark your choice.** The button fills in, a bar counts *Chosen 3 of 10*, and clicking it again takes it back. Nothing is saved yet and the list stays as it is, so a song that's playing keeps playing.
- **Once every song has a choice, or when you press Next songs,** the choices are saved together. The next list is built with them, and the page scrolls back to the top. *Next songs* with only some songs chosen sends those and skips the rest. Changing a setting sends the choices made so far first.
- **Neutral** means you heard the song and have no opinion: it leaves your lists and teaches nothing.
- Choices not sent yet live only in the open page, so closing the tab loses them.

**Your ratings.** Every rating, newest first, filtered by kind and by list, each with Undo, and all of them as a CSV file to download.

**Your data.** Exploratory analysis of your listening: where the data comes from, the cleaning steps with their counts, the medallion layers and their sizes, and charts. Hours per month, when in the week you listen, your main genres and top artists by year, how your plays end, and what your ratings say by genre. Every figure is read from the database when the page opens, and every chart has a *Show the numbers* table.

**How it works.** What the website is for, how a list is made, what is learned and what is set by hand, what the three supervised models learned (their weights, live), how they are tested, and how the recommender could get better.

**Settings.** The wallpaper (colour washes, plain, or a picture of your own, with a dimmer), your Spotify account, what a like also does on Spotify, and *Reload history and pool*.

## Spotify on the website

**The players.** Every card loads Spotify's embedded player, which plays a 30-second preview, or the whole song if you're logged in to Spotify in the same browser. Only one plays at a time. Songs from your history have their Spotify ids already. Songs new to you are looked up by title and artist, with your Spotify app from [chapter 2](2-set-up.md#4-optional-a-spotify-app). Every answer is cached in `data/interim/spotify_search_cache.jsonl`, and a match counts only with the same plain title and artist, so a cover isn't played instead. Without an app, the cards link to a Spotify search.

**Your account, if you want it.** **Connect Spotify** on the Settings page logs the website in to your account, so that it can save songs to your Liked Songs and add them to your playlists: from the cards, or every time you like a song (*When I like a song*). Settings can also make a private playlist for them.

- It asks for six permissions: to see and change your Liked Songs (`user-library-read`, `user-library-modify`), and to see and change your playlists (`playlist-read-private`, `playlist-read-collaborative`, `playlist-modify-private`, `playlist-modify-public`). It can't read your listening, your email or your payment details.
- The login is an [authorization-code flow](https://developer.spotify.com/documentation/web-api/tutorials/code-flow) with a random `state` value, checked when Spotify sends you back, so that a link made by someone else can't log the website in to *their* account.
- The tokens stay on your computer, in `secrets/spotify_account.json`, readable only by you. **Disconnect** deletes them, and your Spotify account's [Manage apps](https://www.spotify.com/account/apps/) page takes the access back on Spotify's side.
- If Spotify says you aren't registered, add your account under **User Management** in your app's settings on the developer dashboard.
- Songs you save reach the recommender with your next export, where Liked Songs count as 5 listens each.

## Only on your computer

`.streamlit/config.toml` binds the website to `localhost`, so no other device on your network can open it. It turns off Streamlit's usage statistics and hides the Deploy button. The font is served from `static/`, so the pages fetch nothing from the internet except Spotify's players. The website shows your listening and saves your ratings, so keep it that way: don't deploy it publicly.

## How it's built

For readers who want to build something like it:

- **Pages and navigation.** `app.py` is the frame: theme, logo, the end of a Spotify login, and `st.navigation(..., position="top")` over the pages in `app_pages/`.
- **State.** The list on screen lives in `st.session_state.page`, and the marked choices in `st.session_state.choices`. Buttons use `on_click` callbacks, which run before the page redraws.
- **Caching.** The history and the learned models are cached (`st.cache_resource`, and `st.cache_data` keyed by your ratings), so a rerun after a click is fast.
- **The look.** Colours and the Inter font come from `[theme]` in `.streamlit/config.toml`. Song cards are containers with a `key`, which Streamlit turns into a CSS class (`st-key-card-...`) that `app_pages/look.py` styles. Anything from the data that goes into HTML (song titles, artists, genres) is escaped first.
- **Charts.** [Altair](https://altair-viz.github.io), with tooltips, a colour-blind-checked palette, and the numbers of each chart in a table under it.
- **The player.** Each card holds a tiny page that loads Spotify's iFrame API and tells the other cards to pause over a `BroadcastChannel`.
- **Tests.** `streamlit.testing.v1.AppTest` runs the pages without a browser. Run it on a copy of your data: set `MUSIC_RECOMMENDER_USER_DIR` to a scratch folder holding a copy of `data/user/`, so no test rating is ever saved among yours.

## The demo on GitHub Pages

The real website can't go online: it shows your listening and saves your ratings. So the project has a second one, [a demo](https://syakirin03.github.io/ML-music/), in `site/`: the same pages, on a listener who doesn't exist.

- **The listener is made up.** `python scripts/build_demo_site.py` invents one: 14 artists with names like Synth Coast and Velvet Tram, about five years of daily listening whose favourites come and go, a Last.fm pool of a few thousand songs, and ratings of 14 lists by a rule of its own (it likes some genres more than others). Then it runs what you ran in chapters 3 and 4 on that listener, in a scratch folder: `build_database.py`, `build_gold.py`, the rediscover model, the website's lists and ratings, the model learned from them. What the pages show goes to `site/data/demo.js`. Rerun it after changing the recommender, so that the demo shows what the code does now.
- **Your browser ranks the songs.** GitHub Pages serves files, not Python, so `site/assets/engine.js` repeats the ranking of `scoring.py` and `recommend.py` (the score with your dials, the model learned from ratings, the cap per artist, variety) on the parts the script keeps for every song. With [Node.js](https://nodejs.org) installed, the script then checks that the page lists the same songs, in the same order and with the same reasons, as the Python code (`scripts/check_demo_engine.js`).
- **Visitors' clicks stay in their tab.** Like, Neutral and Dislike work as on the website, but the choices change no list, profile or model, and they're gone when the tab closes. The page's Content Security Policy forbids it to connect anywhere, send a form, or load anything from another site, and `tests/test_site.py` checks that the policy stays and that no script sends or stores anything.
- **Preview it** with `python -m http.server 8765 --bind 127.0.0.1 --directory site`, then open http://127.0.0.1:8765.
- **Publish your own copy** by turning on GitHub Pages once (the repository's Settings → Pages → Source: *GitHub Actions*). `.github/workflows/pages.yml` then runs the site's tests and publishes `site/` whenever it changes on `main`. Only ever publish the made-up listener: never put your own data in `site/`.

Next: [6. Make it yours](6-make-it-yours.md).
