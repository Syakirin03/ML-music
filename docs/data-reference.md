# Data reference

Every table of `data/processed/spotify.duckdb`, how they link, and what to know before training on them. The database describes itself too: every table and column has a comment.

```sql
SELECT table_name, column_name, comment FROM duckdb_columns() WHERE comment IS NOT NULL
```

## Bronze and silver: `build_database.py`

| Name | Type | Contents |
|---|---|---|
| `bronze.streams_raw` | table | Every streaming event exactly as exported: original field names and raw values, duplicates kept, plus `source_file` and `loaded_at` |
| `bronze.techlog_raw` | table | Every technical log record exactly as exported (minified JSON), duplicates kept, plus `event_type`, `source_file` and `loaded_at` |
| `bronze.techlog_recipients_raw` | table | `recipients.json` as exported |
| `silver.streams` | table | Every streaming event, music and podcasts, typed and renamed, exact duplicates removed |
| `silver.plays` | view | Music plays only, one row per play, in listening order, with `started_at`, local time, `session_id` and `session_position` |
| `silver.tracks` | table | One row per track in **either** export: `track_id`, name, artist, album, `play_count`, `avg_ms_played`. Tracks seen only in the technical log (recommended, queued, shuffled) have `play_count` 0 and no name |
| `silver.episodes` | table | One row per podcast episode in either export, the same way |
| `silver.devices` | table | The devices that wrote technical log records: manufacturer, model, type, operating system |
| `silver.log_event_types` | table | One row per event type, with Spotify's category and description, record count and date range |
| `silver.log_events` | table | Every technical log record once: `ts`, `received_at`, device, operating system and app version, country, `playback_id`, `stream_id` |
| `silver.log_references` | table | Every Spotify URI a log record mentions, also inside lists, with the field and position it came from |
| `silver.playbacks` | table | One row per playback id, with the track when an event names it |
| `silver.log_recipients` | table | The companies listed in `recipients.json` |
| `techlog.<EventType>` | one table per event type | Such as `techlog.AddedToCollection` or `techlog.DJTrackServed`: `event_id`, `ts` and the event's own fields as typed columns |

`build_database.py` replaces exactly these. Tables you create yourself in bronze or silver are kept, so you can store enrichment (embeddings, scores) in your own tables, keyed by `track_id`. Don't give them foreign keys to the tables above: those would block the rebuild. The `techlog` schema is rebuilt as a whole, so keep nothing of your own there.

## How the tables relate

These are foreign keys, which DuckDB enforces and database tools can draw:

| Column | References | Meaning |
|---|---|---|
| `silver.streams.track_uri` | `silver.tracks.track_uri` | the track played |
| `silver.streams.episode_uri` | `silver.episodes.episode_uri` | the episode played |
| `silver.log_events.event_type` | `silver.log_event_types` | the kind of record |
| `silver.log_events.device_id` | `silver.devices` | the device that logged it |
| `silver.log_events.playback_id` | `silver.playbacks` | the playback it belongs to |
| `silver.log_events.stream_id` | `silver.streams` | for `RawCoreStream` records, the same play in the streaming history |
| `silver.log_references.event_id` | `silver.log_events` | the record that mentions the URI |
| `silver.log_references.track_id`, `.episode_id` | `silver.tracks`, `silver.episodes` | what it mentions |
| `silver.playbacks.track_id` | `silver.tracks` | the playback's track |

Two more joins work the same way but can't be declared, because DuckDB views can't have constraints and foreign keys can't cross schemas:

- `techlog.<EventType>.event_id` = `silver.log_events.event_id`, one to one;
- `silver.plays.stream_id` = `silver.streams.stream_id`, and `silver.plays.track_id` = `silver.tracks.track_id`.

`stream_id`, `event_id`, `device_id` and `session_id` are row numbers that every build regenerates, and they can shift with a newer export. Only Spotify's ids (`track_id`, `episode_id`, the URIs) are stable keys to store.

## Gold: `build_gold.py`

| Table | One row per | Use it for |
|---|---|---|
| `gold.tracks` | track | item features: names, `genre`, `genres` (a list), `est_duration_ms`, `is_music` |
| `gold.track_genres` | track and genre | multi-hot genre features, and joins by genre |
| `gold.plays` | music play | the main training table: context, labels, the previous play in the session, and the history before the play |
| `gold.sessions` | listening session | sequence models: `track_ids` holds the session's tracks in order |
| `gold.track_transitions` | pair of tracks played one after the other | Markov-chain and co-occurrence baselines for next-track recommendation |
| `gold.feedback_events` | like, unlike, playlist add or remove, lyrics view | explicit feedback, for the period the technical log covers |
| `gold.exposures` | track served by a daylist, the DJ or Release Radar, per day | labelled candidates, including tracks never played, for training and evaluating a ranker |
| `gold.track_stats` | track | listening totals, skip rate, a time-decayed `affinity`, likes and playlist adds |
| `gold.artists` | artist | the same totals by artist, and the artist's genres |
| `gold.genres` | genre | the same totals by genre |
| `gold.genre_months` | month and main genre | how a taste changed over time |

`build_gold.py` replaces these eleven and leaves other gold tables alone, so you can keep your own model outputs there under other names. The tables fed by the technical log are empty without it.

## Things to know when training

- **Labels.** `end_type` is `finished`, `skipped` (you pressed next or back, picked another track, or stopped) or `interrupted` (the app closed, an error), which says nothing about the track. `label` is a ready-made skip-prediction target: 1 if finished, 0 if skipped within 30 seconds, NULL otherwise. `completion` is `ms_played` over the track's estimated length, the median of its plays that ran to the end. `listen` is 1 if finished, 0 if skipped early, otherwise the completion.
- **Leakage.** Only the `prev_*` and `*_prior_*` columns of `gold.plays` are point-in-time: they use plays that started before the row's play. The totals in `track_stats`, `artists`, `genres` and `genre_months` cover the whole history. Use them to describe items, never as features for predicting plays from the same period.
- **Splits.** `split` in `gold.plays` and `gold.sessions` assigns whole sessions in time order: the newest 10% of plays are `test`, the 10% before them `valid` (`TEST_SHARE`, `VALID_SHARE`). Skip rates drift over the years, so a test on the newest plays is the honest one.
- **Not music.** `is_music` (on tracks, plays and artists; `music_share` on sessions) is false for what `[not_music]` in `config.toml` marks: Spotify publishes white noise, audiobooks and recitations as music tracks. Filter on it, or model those apart.
- **Affinity** is a simple implicit rating: each play's `listen`, counting half as much for every 180 days of age (`AFFINITY_HALF_LIFE_DAYS`).
- **Exposures** cover only the technical log's months. `played_within_1d` is NULL when the history ends less than a day after the exposure. Mind position bias (`position`): what Spotify showed first was more likely to be played anyway.
- **Artists have no ids** in the exports, so they are told apart by name. Some appear under several spellings, such as a name in its own script and in Latin letters.

## Data notes

Streaming history:

- `ts` is when playback **stopped**, in UTC; `started_at` is `ts` minus `ms_played`.
- Local times (`started_at_local`, `local_hour`, `local_dow`, `local_date`) use the time zone of `config.toml` for every play, because `conn_country` comes from IP addresses and is often wrong.
- The export repeats some events verbatim. Bronze keeps them, silver keeps one copy, and `silver.plays` also collapses plays logged twice with the same end second, keeping the longer record.
- `offline_timestamp` is epoch milliseconds in older records and epoch seconds in newer ones. It is stored as a timestamp, and matches the playback's start by the device clock.
- Sessions break at pauses longer than 30 minutes (`SESSION_GAP_MINUTES`).
- `silver.tracks` also holds tracks that were never played: filter on `play_count > 0`, or use `gold.tracks.is_played`.

Technical log:

- It covers only the months before your request, and it can have gap days on which the streaming history has plays it doesn't.
- `ts` is when the app created the event (UTC); `received_at` is when Spotify received it, which can be days later for events logged offline.
- `RawCoreStream` is the log's record of each play. The streaming history is the same data with `ts` cut to the second, which is how `stream_id` is matched (same URI, same `ms_played`); the build prints how many match.
- Low-level playback events (downloads, audio route, lyrics) share a `playback_id` but name no track, so most can't be tied to a play. Matching them by time was only about 96% right, so the build doesn't.
- Column names are the export's field names without the `message_` prefix, except where the short name would clash with a `silver.log_events` column (`message_user_agent`, `message_device_id`).
- The build turns epoch numbers and ISO text in time fields into timestamps (0 and 1970-01-01 mean "not set" and become NULL), hex track ids into `spotify:track:` URIs, and every playback id into 32 lowercase hex characters (the export mixes hex and base64).
- `ts`, `playback_id` and `stream_id` exist both in the techlog tables and in `silver.log_events`: qualify them in joins (`a.ts`).

## DuckDB notes

- **One writer at a time.** Open notebooks and scripts with `read_only=True`, and close them before a build.
- **SQL.** Write `AS` before aliases such as `hours`, `minutes` or `days`, which otherwise parse as interval keywords. `least()` and `greatest()` skip NULLs.
- **Version 1.5.6** is pinned. Its limits shaped the design: foreign keys only within one schema (so every linked table is in silver), no constraints on views, and every table and index costs at least one 256 KB block (so the many techlog tables have no keys).
- **File size.** A rebuild writes the new tables before it frees the old ones, so the file can grow to about twice a fresh build, and later builds reuse the space. To shrink it, delete `spotify.duckdb` and build again, unless it holds tables of your own.
