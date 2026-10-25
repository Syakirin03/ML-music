# 1. Get your Spotify data

Everything in this project starts from data Spotify already keeps about you. Data protection laws such as the GDPR give you the right to a copy, and Spotify has a page for asking. This chapter covers which package to ask for, what's inside it, and where to put it.

**Ask for it today.** Spotify can take up to 30 days to prepare it, often much less. You can set up the rest ([chapter 2](2-set-up.md)) while you wait.

## Which package to ask for

Spotify offers three packages on its privacy page:

| Package | What it holds | For this project |
|---|---|---|
| **Extended streaming history** | Every play since you opened your account: when it ended, how long you listened, the track's Spotify id, how it started and ended, shuffle, offline | **Required.** This is the history everything learns from. |
| **Technical log information** | Events from the apps over the last months: songs you liked, playlist adds, what Spotify's daylist, DJ and Release Radar showed you | Optional. It adds your likes and playlist adds as signals, and the songs Spotify served you. |
| Account data | A year of plays without track ids or how they ended, plus your playlists, library and profile | Not used: without track ids and end reasons there's too little to learn from. |

## How to ask

1. Log in at [spotify.com](https://www.spotify.com) and open **Account → Privacy settings** ([direct link](https://www.spotify.com/account/privacy/)).
2. Under **Download your data**, tick **Extended streaming history**, and **Technical log information** if you want it too.
3. Press **Request data**, then confirm with the link Spotify emails you.
4. When the data is ready, Spotify emails a download link. It works only for a limited time, so download soon.

## What's inside

The download is a zip file per package. Unzipped, the streaming history looks like this:

```
Spotify Extended Streaming History/
├── ReadMeFirst_ExtendedStreamingHistory.pdf      Spotify's explanation of every field
├── Streaming_History_Audio_2019.json             one file per year or part of a year...
├── Streaming_History_Audio_2020_1.json
├── ...
└── Streaming_History_Video_2023.json             ...and video plays, if you have any
```

Each file is a JSON list with one record per play:

```json
{
  "ts": "2024-03-01T12:00:00Z",
  "platform": "android",
  "ms_played": 180000,
  "conn_country": "GB",
  "ip_addr": "192.0.2.1",
  "master_metadata_track_name": "First Song",
  "master_metadata_album_artist_name": "Test Band",
  "master_metadata_album_album_name": "First Album",
  "spotify_track_uri": "spotify:track:0000000000000000000001",
  "episode_name": null,
  "reason_start": "clickrow",
  "reason_end": "trackdone",
  "shuffle": false,
  "skipped": null,
  "offline": false,
  "offline_timestamp": null,
  "incognito_mode": false
}
```

Three things in it matter a lot later on:

- **`ts` is when the play stopped**, not when it started. The start is `ts` minus `ms_played`.
- **`reason_end` says how a play ended.** `trackdone` means it ran to the end; `fwdbtn` means you pressed next, and so on. It's the best signal of liking in the whole export, and it goes back further than the `skipped` flag, which in the author's export only appears from 2022.
- **`conn_country` and `ip_addr` come from your internet connection**, so travel, mobile networks and VPNs make them unreliable for telling your local time. That's why you'll set your time zone yourself in [chapter 2](2-set-up.md).

The technical log is a folder of many JSON files, one per kind of app event (`AddedToCollection`, `DJTrackServed`, ...), with its own ReadMe PDF and a `recipients.json`.

## Where to put it

Unzip both folders, as they are, into `data/raw/`:

```
data/raw/
├── Spotify Extended Streaming History/
└── Spotify Technical Log Information/       (optional)
```

These are the folder names the scripts look for by default. If yours differ, pass them with `--data-dir` and `--techlog-dir` (chapter 3). Never edit these files: the database keeps them exactly as they came, so that every later step can be rebuilt from them.

## Keep it private

These files hold more than your taste. They include the IP addresses you listened from, your device ids and, in the technical log, your email address. Every play also says when you were listening, which reveals your daily routine. Git ignores `data/` in this repository, so none of it can be committed by accident. Don't upload it anywhere, and don't take it out of `.gitignore`. [docs/privacy.md](privacy.md) has the full list.

Next: [2. Set up](2-set-up.md).
