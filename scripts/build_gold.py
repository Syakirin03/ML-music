"""Build the gold layer of spotify.duckdb: tables for training a music recommender.

    python scripts/build_gold.py [--db FILE] [--genres FILE]

Run it after build_database.py and fetch_genres.py. Every run rebuilds these
tables from silver, techlog and track_genres.csv, in one transaction:

    gold.tracks             one row per track: names, genres, estimated length, music flag
    gold.track_genres       one row per track and genre, with the genre's rank
    gold.plays              one row per music play: context, labels, and features known when it started
    gold.sessions           one row per listening session, with its tracks in order
    gold.track_transitions  one row per pair of tracks played one right after the other
    gold.feedback_events    likes, unlikes, playlist adds and removes, lyrics views (technical log)
    gold.exposures          tracks served by daylist, DJ and Release Radar, and whether you played them (technical log)
    gold.track_stats        one row per track: listening totals, skip rate, time-decayed affinity, feedback
    gold.artists            one row per artist: the same totals, and the artist's genres
    gold.genres             one row per genre: the same totals
    gold.genre_months       plays per month and main genre

Other tables in gold are left alone, so keep your own model outputs there
(embeddings, scores) under other names. Every table and column has a
description in the database.

The prev_* and *_prior_* columns of gold.plays only use plays that started
before the row's play, so they are safe as features for predicting it. The
totals in track_stats, artists, genres and genre_months cover the whole
history: as features for predicting plays they would leak the future.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

import duckdb

from build_database import CONFIG, HOME_TIMEZONE, SESSION_GAP_MINUTES, ident, literal

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "processed" / "spotify.duckdb"
DEFAULT_GENRES = ROOT / "data" / "processed" / "track_genres.csv"

# The last sessions, holding this share of all plays each, are the validation
# and then the test split of gold.plays and gold.sessions.
VALID_SHARE = 0.1
TEST_SHARE = 0.1

# In affinity scores a listen counts half as much for every this many days
# between it and the end of the listening history.
AFFINITY_HALF_LIFE_DAYS = 180

# Spotify counts a play as a stream from 30 seconds on. Moving on from a track
# sooner is a skip here.
SKIP_MS = 30_000

# reason_end values for plays the user ended: next, previous, another track
# picked, or stop.
USER_ENDED = ["fwdbtn", "backbtn", "endplay", "clickrow"]

# Tracks that aren't music to you ([not_music] in config.toml): those with one of these genres, and those
# without genres whose artist, title or album matches the pattern (lower case; empty: none).
NOT_MUSIC_GENRES = CONFIG["not_music"]["genres"]
NOT_MUSIC_PATTERN = CONFIG["not_music"]["pattern"]


class Table(NamedTuple):
    name: str
    key: str  # the columns that identify a row
    description: str
    sql: str
    columns: dict[str, str]
    # techlog table -> SELECT for the {branches} placeholder of sql; tables
    # missing from an export are left out.
    branches: dict[str, str] | None = None
    empty: str = ""  # SELECT with the branches' columns and no rows, for when none exist


TRACKS = Table(
    name="tracks",
    key="track_id",
    description=(
        "One row per track in silver.tracks, most played first: names, Last.fm genres (from track_genres.csv), "
        "estimated length and whether it is music. Static descriptions of the track, safe as model features."
    ),
    sql="""
WITH genres AS (
    SELECT track_id, genre, string_split(genres, '|') AS genres, genre_source
    FROM read_csv($genres_csv, header = true, nullstr = 'NA', columns = {
        'track_id': 'VARCHAR', 'track_name': 'VARCHAR', 'artist_name': 'VARCHAR', 'album_name': 'VARCHAR',
        'genre': 'VARCHAR', 'genres': 'VARCHAR', 'genre_source': 'VARCHAR'
    })
),
durations AS (
    SELECT track_id, CAST(median(ms_played) AS INTEGER) AS est_duration_ms
    FROM silver.plays
    WHERE reason_end = 'trackdone'
    GROUP BY track_id
)
SELECT
    t.track_id,
    t.track_name,
    t.artist_name,
    t.album_name,
    g.genre,
    g.genres,
    g.genre_source,
    coalesce(NOT (
        list_has_any(g.genres, CAST($not_music_genres AS VARCHAR[]))
        OR (g.genres IS NULL AND $not_music_pattern <> ''
            AND regexp_matches(lower(concat_ws(' ', t.artist_name, t.track_name, t.album_name)), $not_music_pattern))
    ), true) AS is_music,
    d.est_duration_ms,
    t.play_count > 0 AS is_played
FROM silver.tracks t
LEFT JOIN genres g USING (track_id)
LEFT JOIN durations d USING (track_id)
ORDER BY t.play_count DESC, t.track_id
""",
    columns={
        "track_id": "Spotify track id, the key of this table; the same as silver.tracks.track_id.",
        "track_name": "Most recent title, as in silver.tracks. NULL for tracks only seen in the technical log.",
        "artist_name": "Most recent artist, as in silver.tracks. The exports have no artist ids, so artists are told apart by name.",
        "album_name": "Most recent album, as in silver.tracks.",
        "genre": "Main genre from Last.fm: the first of genres. NULL if Last.fm had none.",
        "genres": "All the track's Last.fm genres, main one first (see fetch_genres.py). NULL if Last.fm had none.",
        "genre_source": "Whose Last.fm tags the genres come from: track, album, artist_tracks (the artist's other tracks) or artist.",
        "is_music": (
            "False for audio that Spotify publishes as music but that isn't music to you, by [not_music] in config.toml: "
            "one of its genres, or for tracks without genres, an artist, title or album matching its pattern. "
            "True for tracks without a title."
        ),
        "est_duration_ms": "Estimated length: the median ms_played of plays that ran to the end. NULL if none did.",
        "is_played": "The streaming history has a play of the track. False for tracks only seen in the technical log (recommended, queued, shuffled, ...).",
    },
)

TRACK_GENRES = Table(
    name="track_genres",
    key="track_id, genre",
    description="One row per track and genre, from gold.tracks.genres: for multi-hot genre features and joins by genre.",
    sql="""
SELECT track_id, unnest(genres) AS genre, generate_subscripts(genres, 1) AS genre_rank
FROM gold.tracks
WHERE genres IS NOT NULL
""",
    columns={
        "track_id": "The track; joins gold.tracks.",
        "genre": "One of the track's genres.",
        "genre_rank": "1 for the track's main genre, 2 for the next, and so on.",
    },
)

PLAYS = Table(
    name="plays",
    key="stream_id",
    description=(
        "One row per music play in silver.plays, in time order: the main training table. Context (time, device, "
        "shuffle), labels (end_type, finished, skipped_early, label), the previous play in the session, and the "
        "listening history before the play (*_prior_* columns, safe as features). split gives a time-ordered "
        "train/valid/test split by session."
    ),
    sql="""
WITH sessions AS (
    SELECT session_id, sum(count(*)) OVER (ORDER BY min(started_at), session_id) / sum(count(*)) OVER () AS share_through
    FROM silver.plays
    GROUP BY session_id
),
plays AS (
    SELECT
        p.stream_id, p.session_id, p.session_position, p.track_id, t.track_name, t.artist_name, t.genre, t.is_music,
        p.started_at, p.ended_at, p.started_at_local, p.local_hour, p.local_dow, p.ms_played,
        CASE WHEN t.est_duration_ms > 0 THEN least(p.ms_played / t.est_duration_ms, 1.0) END AS completion,  -- least() skips NULLs
        CASE
            WHEN p.reason_end = 'trackdone' THEN 'finished'
            WHEN list_contains($user_ended, p.reason_end) THEN 'skipped'
            ELSE 'interrupted'
        END AS end_type,
        p.reason_start, p.reason_end, p.shuffle, p.offline, p.incognito_mode, p.is_video, p.platform_family,
        CASE
            WHEN s.share_through <= 1 - $valid_share - $test_share THEN 'train'
            WHEN s.share_through <= 1 - $test_share THEN 'valid'
            ELSE 'test'
        END AS split
    FROM silver.plays p
    JOIN gold.tracks t USING (track_id)
    JOIN sessions s USING (session_id)
),
labelled AS (
    SELECT *, end_type = 'finished' AS finished, end_type = 'skipped' AND ms_played < $skip_ms AS skipped_early
    FROM plays
)
SELECT
    stream_id, session_id, session_position, track_id, track_name, artist_name, genre, is_music,
    started_at, ended_at, started_at_local,
    CAST(started_at_local AS DATE) AS local_date,
    CAST(local_hour AS INTEGER) AS local_hour,
    CAST(local_dow AS INTEGER) AS local_dow,
    local_dow >= 6 AS is_weekend,
    ms_played, completion, end_type, finished, skipped_early,
    CAST(CASE WHEN finished THEN 1 WHEN skipped_early THEN 0 END AS TINYINT) AS label,
    CASE
        WHEN finished THEN 1.0
        WHEN skipped_early THEN 0.0
        ELSE coalesce(completion, CASE WHEN ms_played >= $skip_ms THEN 0.5 ELSE 0.0 END)
    END AS listen,
    reason_start, reason_end, shuffle, offline, incognito_mode, is_video, platform_family,
    lag(track_id) OVER in_session AS prev_track_id,
    lag(end_type) OVER in_session AS prev_end_type,
    epoch(started_at - lag(ended_at) OVER in_session) AS seconds_since_prev,
    artist_name = lag(artist_name) OVER in_session AS same_artist_as_prev,
    row_number() OVER by_track - 1 AS track_prior_plays,
    coalesce(sum(CAST(finished AS INTEGER)) OVER track_before, 0) AS track_prior_finished,
    coalesce(sum(CAST(skipped_early AS INTEGER)) OVER track_before, 0) AS track_prior_skips,
    count(*) OVER (PARTITION BY track_id ORDER BY started_at RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND CURRENT ROW) - 1
        AS track_plays_prior_30d,
    epoch(started_at - lag(started_at) OVER by_track) / 86400 AS days_since_track_played,
    row_number() OVER by_artist - 1 AS artist_prior_plays,
    count(*) OVER (PARTITION BY artist_name ORDER BY started_at RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND CURRENT ROW) - 1
        AS artist_plays_prior_30d,
    CASE WHEN genre IS NOT NULL THEN
        count(*) OVER (PARTITION BY genre ORDER BY started_at RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND CURRENT ROW) - 1
    END AS genre_plays_prior_30d,
    split
FROM labelled
WINDOW
    in_session AS (PARTITION BY session_id ORDER BY session_position),
    by_track AS (PARTITION BY track_id ORDER BY started_at, stream_id),
    track_before AS (PARTITION BY track_id ORDER BY started_at, stream_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    by_artist AS (PARTITION BY artist_name ORDER BY started_at, stream_id)
ORDER BY started_at, stream_id
""",
    columns={
        "stream_id": "The play, the key of this table; the same as silver.plays.stream_id and silver.streams.stream_id.",
        "session_id": f"Listening session: plays with pauses of at most {SESSION_GAP_MINUTES} minutes. Joins gold.sessions.",
        "session_position": "Position of the play in its session, from 1.",
        "track_id": "The track; joins gold.tracks and gold.track_stats.",
        "track_name": "As in gold.tracks.",
        "artist_name": "As in gold.tracks; joins gold.artists.",
        "genre": "Main genre of the track, as in gold.tracks.",
        "is_music": "As in gold.tracks.",
        "started_at": "When the play started (the export's ts minus ms_played), UTC.",
        "ended_at": "When the play ended (the export's ts), UTC.",
        "started_at_local": f"started_at in {HOME_TIMEZONE} time.",
        "local_date": "Date of started_at_local.",
        "local_hour": "Hour of started_at_local, 0-23.",
        "local_dow": "ISO day of the week of started_at_local: 1 Monday ... 7 Sunday.",
        "is_weekend": "Saturday or Sunday.",
        "ms_played": "Milliseconds played.",
        "completion": "ms_played divided by the track's est_duration_ms, at most 1. NULL if the track's length is not known.",
        "end_type": (
            "How the play ended: finished (ran to the end), skipped (you pressed next or previous, picked another "
            "track or stopped), or interrupted (the app closed, the connection dropped, an error, ...), which says "
            "nothing about the track."
        ),
        "finished": "end_type is finished.",
        "skipped_early": f"Skipped within {SKIP_MS // 1000} seconds: end_type is skipped and ms_played < {SKIP_MS}.",
        "label": "Suggested target for skip prediction: 1 if finished, 0 if skipped_early, NULL otherwise (interrupted, or skipped later).",
        "listen": (
            f"How much of the track you heard, as an implicit rating: 1 if finished, 0 if skipped_early, otherwise "
            f"completion (0.5 if the length is not known and at least {SKIP_MS // 1000} seconds played, else 0)."
        ),
        "reason_start": "Why the track started: trackdone (the previous one ended), clickrow (you picked it), fwdbtn, backbtn, appload, ...",
        "reason_end": "Why the track ended, as exported. end_type groups the values.",
        "shuffle": "Shuffle was on.",
        "offline": "Played offline.",
        "incognito_mode": "Played in a private session.",
        "is_video": "A video version was played.",
        "platform_family": "android, ios, windows, web, ...",
        "prev_track_id": "Track of the previous play in the session. NULL for a session's first play.",
        "prev_end_type": "end_type of the previous play in the session.",
        "seconds_since_prev": (
            "Seconds from the end of the previous play in the session to the start of this one. Negative when they "
            "overlap: by under a second because the export rounds ts to the second, by a few seconds with crossfade, "
            "and now and then by minutes (two devices, or an export glitch)."
        ),
        "same_artist_as_prev": "Same artist as the previous play in the session.",
        "track_prior_plays": "Plays of this track that started before this one (0 the first time).",
        "track_prior_finished": "How many of those were finished.",
        "track_prior_skips": "How many of those were skipped_early.",
        "track_plays_prior_30d": "Plays of this track in the 30 days before this one.",
        "days_since_track_played": "Days since the previous play of this track started. NULL the first time.",
        "artist_prior_plays": "Plays of this artist that started before this one.",
        "artist_plays_prior_30d": "Plays of this artist in the 30 days before this one.",
        "genre_plays_prior_30d": "Plays of tracks with the same main genre in the 30 days before this one. NULL if the track has no genre.",
        "split": (
            f"train, valid or test, by session start: the last sessions holding {TEST_SHARE:.0%} of plays are test, "
            f"the {VALID_SHARE:.0%} before them valid. A session is never split."
        ),
    },
)

SESSIONS = Table(
    name="sessions",
    key="session_id",
    description=(
        "One row per listening session, with its tracks in play order (track_ids): sequences for session-based "
        "models (next-track prediction, item2vec), and session context."
    ),
    sql="""
WITH session_genres AS (
    SELECT session_id, genre, count(*) AS plays, min(session_position) AS first_position
    FROM gold.plays
    WHERE genre IS NOT NULL
    GROUP BY session_id, genre
),
main_genres AS (
    SELECT session_id, arg_max(genre, (plays, -first_position)) AS main_genre
    FROM session_genres
    GROUP BY session_id
)
SELECT
    p.session_id,
    min(p.started_at) AS started_at,
    max(p.ended_at) AS ended_at,
    min(p.started_at_local) AS started_at_local,
    CAST(min(p.started_at_local) AS DATE) AS local_date,
    CAST(hour(min(p.started_at_local)) AS INTEGER) AS local_hour,
    CAST(isodow(min(p.started_at_local)) AS INTEGER) AS local_dow,
    isodow(min(p.started_at_local)) >= 6 AS is_weekend,
    epoch(max(p.ended_at) - min(p.started_at)) / 60 AS duration_minutes,
    count(*) AS plays,
    count(DISTINCT p.track_id) AS distinct_tracks,
    count(DISTINCT p.artist_name) AS distinct_artists,
    count(*) FILTER (WHERE p.finished) AS finished_plays,
    count(*) FILTER (WHERE p.skipped_early) AS early_skips,
    avg(CAST(p.finished AS INTEGER)) AS finish_rate,
    avg(CAST(p.skipped_early AS INTEGER)) AS skip_rate,
    avg(CAST(p.shuffle AS INTEGER)) AS shuffle_rate,
    avg(CAST(p.is_music AS INTEGER)) AS music_share,
    arg_min(p.platform_family, p.session_position) AS platform_family,
    arg_min(p.reason_start, p.session_position) AS start_reason,
    any_value(m.main_genre) AS main_genre,
    list(p.track_id ORDER BY p.session_position) AS track_ids,
    any_value(p.split) AS split
FROM gold.plays p
LEFT JOIN main_genres m USING (session_id)
GROUP BY p.session_id
ORDER BY p.session_id
""",
    columns={
        "session_id": "The session, the key of this table; joins gold.plays.",
        "started_at": "Start of the session's first play, UTC.",
        "ended_at": "End of the session's last play, UTC.",
        "started_at_local": f"started_at in {HOME_TIMEZONE} time.",
        "local_date": "Date of started_at_local.",
        "local_hour": "Hour of started_at_local, 0-23.",
        "local_dow": "ISO day of the week of started_at_local: 1 Monday ... 7 Sunday.",
        "is_weekend": "Saturday or Sunday.",
        "duration_minutes": "From the start of the first play to the end of the last.",
        "plays": "Plays in the session.",
        "distinct_tracks": "Different tracks played.",
        "distinct_artists": "Different artists played.",
        "finished_plays": "Plays that were finished.",
        "early_skips": "Plays that were skipped_early.",
        "finish_rate": "finished_plays / plays.",
        "skip_rate": "early_skips / plays.",
        "shuffle_rate": "Share of the plays with shuffle on.",
        "music_share": "Share of the plays that are music (is_music).",
        "platform_family": "Platform of the first play.",
        "start_reason": "reason_start of the first play: how the session began (clickrow, appload, ...).",
        "main_genre": "The genre with the most plays in the session (ties: the one played first). NULL if no play has a genre.",
        "track_ids": "The session's tracks in play order, repeats included.",
        "split": "As in gold.plays.",
    },
)

TRACK_TRANSITIONS = Table(
    name="track_transitions",
    key="from_track_id, to_track_id",
    description=(
        "One row per pair of tracks played one right after the other in a session: counts for Markov-chain and "
        "co-occurrence baselines of next-track recommendation."
    ),
    sql="""
SELECT
    p.prev_track_id AS from_track_id,
    p.track_id AS to_track_id,
    any_value(f.artist_name = p.artist_name) AS same_artist,
    count(*) AS transitions,
    count(*) FILTER (WHERE p.reason_start = 'clickrow') AS picked,
    count(*) FILTER (WHERE p.finished) AS finished,
    count(*) FILTER (WHERE p.skipped_early) AS early_skips,
    min(p.started_at) AS first_at,
    max(p.started_at) AS last_at
FROM gold.plays p
JOIN gold.tracks f ON f.track_id = p.prev_track_id
GROUP BY p.prev_track_id, p.track_id
ORDER BY transitions DESC, from_track_id, to_track_id
""",
    columns={
        "from_track_id": "The earlier track; joins gold.tracks.",
        "to_track_id": "The track played right after it in the same session; joins gold.tracks.",
        "same_artist": "Both tracks are by the same artist.",
        "transitions": "Times to_track was played right after from_track.",
        "picked": "How many of those you started by picking the track (reason_start clickrow) instead of letting playback go on.",
        "finished": "How many of those plays of to_track were finished.",
        "early_skips": "How many of those plays of to_track were skipped_early.",
        "first_at": "Start of the first of those plays of to_track, UTC.",
        "last_at": "Start of the last of those plays of to_track, UTC.",
    },
)

FEEDBACK_EVENTS = Table(
    name="feedback_events",
    key="event_id",
    description=(
        "Things you did with tracks besides playing them, from the technical log (so only for the period it "
        "covers): likes and unlikes, playlist adds and removes, lyrics views. The log repeats some records within "
        "milliseconds; repeats of an event within a minute are dropped."
    ),
    sql="""
WITH events AS ({branches}),
spaced AS (
    SELECT *, ts - lag(ts) OVER (PARTITION BY event, track_id, playlist_uri ORDER BY ts, event_id) AS since_previous
    FROM events
)
SELECT event_id, ts, track_id, event, playlist_uri, source_event_type
FROM spaced
WHERE since_previous IS NULL OR since_previous >= INTERVAL 1 MINUTE
ORDER BY ts, event_id
""",
    # Every branch names its columns: UNION ALL takes the names from the first
    # one, and any of them can be missing from an export.
    branches={
        "AddedToCollection": """
    SELECT x.event_id, x.ts, r.track_id, 'like' AS event, CAST(NULL AS VARCHAR) AS playlist_uri, 'AddedToCollection' AS source_event_type
    FROM techlog.AddedToCollection x JOIN silver.log_references r USING (event_id)
    WHERE x."set" = 'collection' AND r.field = 'item_uri' AND r.track_id IS NOT NULL""",
        "RemovedFromCollection": """
    SELECT x.event_id, x.ts, r.track_id, 'unlike' AS event, CAST(NULL AS VARCHAR) AS playlist_uri, 'RemovedFromCollection' AS source_event_type
    FROM techlog.RemovedFromCollection x JOIN silver.log_references r USING (event_id)
    WHERE x."set" = 'collection' AND r.field = 'item_uri' AND r.track_id IS NOT NULL""",
        "AddedToPlaylist": """
    SELECT x.event_id, x.ts, r.track_id, 'playlist_add' AS event, x.playlist_uri, 'AddedToPlaylist' AS source_event_type
    FROM techlog.AddedToPlaylist x JOIN silver.log_references r USING (event_id)
    WHERE r.field = 'item_uri' AND r.track_id IS NOT NULL""",
        "RemovedFromPlaylist": """
    SELECT x.event_id, x.ts, r.track_id, 'playlist_remove' AS event, x.playlist_uri, 'RemovedFromPlaylist' AS source_event_type
    FROM techlog.RemovedFromPlaylist x JOIN silver.log_references r USING (event_id)
    WHERE r.field = 'item_uri' AND r.track_id IS NOT NULL""",
        "MinimumLyricsCharactersSeen": """
    SELECT x.event_id, x.ts, r.track_id, 'lyrics_view' AS event, CAST(NULL AS VARCHAR) AS playlist_uri,
           'MinimumLyricsCharactersSeen' AS source_event_type
    FROM techlog.MinimumLyricsCharactersSeen x JOIN silver.log_references r USING (event_id)
    WHERE r.field = 'track_uri' AND r.track_id IS NOT NULL""",
    },
    empty="""
    SELECT CAST(NULL AS BIGINT) AS event_id, CAST(NULL AS TIMESTAMPTZ) AS ts, CAST(NULL AS VARCHAR) AS track_id,
           CAST(NULL AS VARCHAR) AS event, CAST(NULL AS VARCHAR) AS playlist_uri, CAST(NULL AS VARCHAR) AS source_event_type
    WHERE false""",
    columns={
        "event_id": "The event's first log record, the key of this table; joins silver.log_events and techlog.<source_event_type>.",
        "ts": "When the client logged it, UTC.",
        "track_id": "The track; joins gold.tracks.",
        "event": (
            "like (added to Liked Songs), unlike (removed from Liked Songs), playlist_add, playlist_remove, or "
            "lyrics_view (140 characters of the track's lyrics were on screen)."
        ),
        "playlist_uri": "The playlist, for playlist_add and playlist_remove.",
        "source_event_type": "The techlog table the record is from.",
    },
)

EXPOSURES = Table(
    name="exposures",
    key="source, track_id, exposed_date",
    description=(
        "Tracks Spotify served you in a daylist, the DJ or Release Radar, from the technical log (so only for the "
        "period it covers), and whether you then played them: labelled candidates, including tracks you never "
        "played, for training and evaluating a ranker. One row per source, track and day."
    ),
    sql="""
WITH served AS ({branches}),
exposures AS (
    SELECT
        source,
        track_id,
        CAST(timezone($timezone, ts) AS DATE) AS exposed_date,
        min(ts) AS first_exposed_at,
        min(position) AS position,
        count(*) AS times_logged
    FROM served
    GROUP BY ALL
),
history AS (
    SELECT max(started_at) AS last_play FROM gold.plays
)
SELECT
    e.source,
    e.track_id,
    e.exposed_date,
    e.first_exposed_at,
    CAST(hour(timezone($timezone, e.first_exposed_at)) AS INTEGER) AS local_hour,
    e.position,
    e.times_logged,
    -- A play that started up to 5 minutes before the log record counts as a reaction to it.
    coalesce(bool_or(p.started_at < e.first_exposed_at - INTERVAL 5 MINUTE), false) AS heard_before,
    CASE WHEN h.last_play >= e.first_exposed_at + INTERVAL 1 DAY THEN coalesce(bool_or(
        p.started_at >= e.first_exposed_at - INTERVAL 5 MINUTE AND p.started_at < e.first_exposed_at + INTERVAL 1 DAY
    ), false) END AS played_within_1d,
    CASE WHEN h.last_play >= e.first_exposed_at + INTERVAL 1 DAY THEN coalesce(bool_or(
        p.finished AND p.started_at >= e.first_exposed_at - INTERVAL 5 MINUTE AND p.started_at < e.first_exposed_at + INTERVAL 1 DAY
    ), false) END AS finished_within_1d
FROM exposures e
CROSS JOIN history h
LEFT JOIN gold.plays p ON p.track_id = e.track_id
GROUP BY e.source, e.track_id, e.exposed_date, e.first_exposed_at, e.position, e.times_logged, h.last_play
ORDER BY e.first_exposed_at, e.source, e.position, e.track_id
""",
    branches={
        "DaylistGenerated": """
    SELECT 'daylist' AS source, x.ts, r.track_id, r.position
    FROM techlog.DaylistGenerated x JOIN silver.log_references r USING (event_id)
    WHERE r.field = 'tracks[]' AND r.track_id IS NOT NULL""",
        "DJTrackServed": """
    SELECT 'dj' AS source, x.ts, r.track_id, TRY_CAST(x.track_index AS INTEGER) AS position
    FROM techlog.DJTrackServed x JOIN silver.log_references r USING (event_id)
    WHERE r.field = 'track_uri' AND r.track_id IS NOT NULL""",
        "ReleaseRadarServedRecs": """
    SELECT 'release_radar' AS source, x.ts, r.track_id, r.position
    FROM techlog.ReleaseRadarServedRecs x JOIN silver.log_references r USING (event_id)
    WHERE r.field = 'track_uris[]' AND r.track_id IS NOT NULL""",
    },
    empty="""
    SELECT CAST(NULL AS VARCHAR) AS source, CAST(NULL AS TIMESTAMPTZ) AS ts, CAST(NULL AS VARCHAR) AS track_id,
           CAST(NULL AS INTEGER) AS position
    WHERE false""",
    columns={
        "source": "daylist, dj or release_radar.",
        "track_id": "The track served; joins gold.tracks. Many were never played, so they have no title or genres.",
        "exposed_date": f"The {HOME_TIMEZONE} date it was served. Repeated logs of a list on one day are merged.",
        "first_exposed_at": "When it was first served that day, UTC.",
        "local_hour": "Hour of first_exposed_at in local time, 0-23.",
        "position": "Its best (lowest) position in the served list that day, from 1; for dj, the DJ's track index.",
        "times_logged": "Log records merged into this row.",
        "heard_before": "The streaming history has a play of the track from before it was served.",
        "played_within_1d": "You played the track within a day after it was served. NULL if the streaming history ends sooner.",
        "finished_within_1d": "You played it to the end within a day after it was served. NULL if the streaming history ends sooner.",
    },
)

TRACK_STATS = Table(
    name="track_stats",
    key="track_id",
    description=(
        "One row per track in gold.tracks: listening totals over the whole history, a time-decayed affinity score "
        "of current taste, and feedback from the technical log. Totals include the future of any given play, so use "
        "gold.plays' *_prior_* columns as features for predicting plays."
    ),
    sql="""
WITH history AS (
    SELECT max(ended_at) AS ended_at FROM gold.plays
),
weighted AS (
    SELECT p.*, epoch(h.ended_at - p.started_at) / 86400 AS age_days
    FROM gold.plays p, history h
),
per_track AS (
    SELECT
        track_id,
        count(*) AS plays,
        count(*) FILTER (WHERE ms_played >= $skip_ms) AS streams,
        count(*) FILTER (WHERE finished) AS finished_plays,
        count(*) FILTER (WHERE skipped_early) AS early_skips,
        avg(completion) AS avg_completion,
        count(*) FILTER (WHERE reason_start = 'clickrow') AS picked_plays,
        count(*) FILTER (WHERE reason_start = 'backbtn') AS returned_to,
        sum(ms_played) AS total_ms_played,
        sum(listen) AS listens,
        sum(listen * pow(0.5, age_days / $half_life_days)) AS affinity,
        min(started_at) AS first_played_at,
        max(started_at) AS last_played_at,
        count(DISTINCT local_date) AS days_played,
        count(DISTINCT session_id) AS sessions,
        count(*) FILTER (WHERE age_days <= 30) AS plays_last_30d,
        count(*) FILTER (WHERE age_days <= 365) AS plays_last_365d
    FROM weighted
    GROUP BY track_id
),
feedback AS (
    SELECT
        track_id,
        arg_max(event, ts) FILTER (WHERE event IN ('like', 'unlike')) = 'like' AS liked_in_log,
        count(*) FILTER (WHERE event = 'playlist_add') AS playlist_adds,
        count(*) FILTER (WHERE event = 'playlist_remove') AS playlist_removes,
        count(*) FILTER (WHERE event = 'lyrics_view') AS lyrics_views
    FROM gold.feedback_events
    GROUP BY track_id
)
SELECT
    t.track_id,
    coalesce(s.plays, 0) AS plays,
    coalesce(s.streams, 0) AS streams,
    coalesce(s.finished_plays, 0) AS finished_plays,
    coalesce(s.early_skips, 0) AS early_skips,
    s.finished_plays / s.plays AS finish_rate,
    s.early_skips / s.plays AS skip_rate,
    s.avg_completion,
    coalesce(s.picked_plays, 0) AS picked_plays,
    coalesce(s.returned_to, 0) AS returned_to,
    coalesce(s.total_ms_played, 0) AS total_ms_played,
    coalesce(s.listens, 0) AS listens,
    coalesce(s.affinity, 0) AS affinity,
    s.first_played_at,
    s.last_played_at,
    epoch(h.ended_at - s.last_played_at) / 86400 AS days_since_played,
    coalesce(s.days_played, 0) AS days_played,
    coalesce(s.sessions, 0) AS sessions,
    coalesce(s.plays_last_30d, 0) AS plays_last_30d,
    coalesce(s.plays_last_365d, 0) AS plays_last_365d,
    coalesce(f.liked_in_log, false) AS liked_in_log,
    coalesce(f.playlist_adds, 0) AS playlist_adds,
    coalesce(f.playlist_removes, 0) AS playlist_removes,
    coalesce(f.lyrics_views, 0) AS lyrics_views
FROM gold.tracks t
CROSS JOIN history h
LEFT JOIN per_track s USING (track_id)
LEFT JOIN feedback f USING (track_id)
ORDER BY affinity DESC, t.track_id
""",
    columns={
        "track_id": "The track, the key of this table; joins gold.tracks.",
        "plays": "Plays in gold.plays.",
        "streams": f"Plays of at least {SKIP_MS // 1000} seconds, what Spotify counts as a stream.",
        "finished_plays": "Plays that were finished.",
        "early_skips": "Plays that were skipped_early.",
        "finish_rate": "finished_plays / plays. NULL if never played.",
        "skip_rate": "early_skips / plays. NULL if never played.",
        "avg_completion": "Average completion of the plays.",
        "picked_plays": "Plays you started by picking the track (reason_start clickrow).",
        "returned_to": "Plays started with the previous button: times you went back to the track.",
        "total_ms_played": "Milliseconds played in all.",
        "listens": "Plays weighted by how much of the track was heard: the sum of gold.plays.listen.",
        "affinity": (
            f"listens, with each play counting half as much for every {AFFINITY_HALF_LIFE_DAYS} days between it and "
            "the end of the listening history: how much you like the track lately."
        ),
        "first_played_at": "Start of the first play, UTC.",
        "last_played_at": "Start of the last play, UTC.",
        "days_since_played": "Days from last_played_at to the end of the listening history.",
        "days_played": "Local dates with a play.",
        "sessions": "Sessions with a play.",
        "plays_last_30d": "Plays in the last 30 days of the listening history.",
        "plays_last_365d": "Plays in the last 365 days of the listening history.",
        "liked_in_log": (
            "In Liked Songs by the technical log: its last like or unlike there is a like. Likes from before the "
            "period the log covers are not known."
        ),
        "playlist_adds": "Times added to a playlist (gold.feedback_events).",
        "playlist_removes": "Times removed from a playlist (gold.feedback_events).",
        "lyrics_views": "Plays with the lyrics on screen (gold.feedback_events).",
    },
)

ARTISTS = Table(
    name="artists",
    key="artist_name",
    description=(
        "One row per artist with a played track: genres, and the totals of gold.track_stats summed over the "
        "artist's tracks. The exports have no artist ids, so artists are told apart by name."
    ),
    sql="""
WITH genre_plays AS (
    SELECT t.artist_name, g.genre, sum(s.plays) AS plays, min(g.genre_rank) AS best_rank
    FROM gold.track_genres g
    JOIN gold.tracks t USING (track_id)
    JOIN gold.track_stats s USING (track_id)
    GROUP BY t.artist_name, g.genre
),
artist_genres AS (
    SELECT artist_name, list(genre ORDER BY plays DESC, best_rank, genre) AS genres
    FROM genre_plays
    GROUP BY artist_name
),
artist_days AS (
    SELECT artist_name, count(DISTINCT local_date) AS days_played
    FROM gold.plays
    GROUP BY artist_name
)
SELECT
    t.artist_name,
    g.genres[1] AS genre,
    g.genres[1:5] AS genres,
    count(*) FILTER (WHERE NOT t.is_music) * 2 <= count(*) AS is_music,
    count(*) AS tracks,
    sum(s.plays) AS plays,
    sum(s.streams) AS streams,
    sum(s.finished_plays) AS finished_plays,
    sum(s.early_skips) AS early_skips,
    sum(s.finished_plays) / sum(s.plays) AS finish_rate,
    sum(s.early_skips) / sum(s.plays) AS skip_rate,
    sum(s.total_ms_played) AS total_ms_played,
    sum(s.listens) AS listens,
    sum(s.affinity) AS affinity,
    min(s.first_played_at) AS first_played_at,
    max(s.last_played_at) AS last_played_at,
    d.days_played,
    count(*) FILTER (WHERE s.liked_in_log) AS liked_tracks_in_log
FROM gold.tracks t
JOIN gold.track_stats s USING (track_id)
JOIN artist_days d USING (artist_name)
LEFT JOIN artist_genres g USING (artist_name)
WHERE s.plays > 0
GROUP BY t.artist_name, g.genres, d.days_played
ORDER BY affinity DESC, t.artist_name
""",
    columns={
        "artist_name": "The artist, the key of this table; joins gold.tracks and gold.plays.",
        "genre": "The genre with the most plays among the artist's tracks.",
        "genres": "Up to 5 genres of the artist's tracks, most played first.",
        "is_music": "At least half of the artist's tracks are music (is_music).",
        "tracks": "The artist's played tracks.",
        "plays": "Plays of the artist's tracks.",
        "streams": "As in gold.track_stats, summed.",
        "finished_plays": "As in gold.track_stats, summed.",
        "early_skips": "As in gold.track_stats, summed.",
        "finish_rate": "finished_plays / plays.",
        "skip_rate": "early_skips / plays.",
        "total_ms_played": "As in gold.track_stats, summed.",
        "listens": "As in gold.track_stats, summed.",
        "affinity": "As in gold.track_stats, summed: how much you like the artist lately.",
        "first_played_at": "Start of the first play of any of the artist's tracks, UTC.",
        "last_played_at": "Start of the last play of any of the artist's tracks, UTC.",
        "days_played": "Local dates with a play of the artist.",
        "liked_tracks_in_log": "The artist's tracks with liked_in_log.",
    },
)

GENRES = Table(
    name="genres",
    key="genre",
    description="One row per genre in gold.track_genres, with the totals of gold.track_stats over its tracks.",
    sql="""
SELECT
    g.genre,
    count(*) AS tracks,
    count(*) FILTER (WHERE g.genre_rank = 1) AS main_tracks,
    count(DISTINCT t.artist_name) AS artists,
    sum(s.plays) AS plays,
    coalesce(sum(s.plays) FILTER (WHERE g.genre_rank = 1), 0) AS main_plays,
    coalesce(sum(s.plays) FILTER (WHERE g.genre_rank = 1), 0) / (SELECT count(*) FROM gold.plays) AS share_of_plays,
    sum(s.total_ms_played) AS total_ms_played,
    sum(s.finished_plays) / sum(s.plays) AS finish_rate,
    sum(s.early_skips) / sum(s.plays) AS skip_rate,
    sum(s.affinity) AS affinity,
    min(s.first_played_at) AS first_played_at,
    max(s.last_played_at) AS last_played_at
FROM gold.track_genres g
JOIN gold.tracks t USING (track_id)
JOIN gold.track_stats s USING (track_id)
GROUP BY g.genre
ORDER BY plays DESC, g.genre
""",
    columns={
        "genre": "The genre, the key of this table; joins gold.track_genres.",
        "tracks": "Tracks with the genre, as their main genre or not.",
        "main_tracks": "Tracks with the genre as their main genre.",
        "artists": "Artists of those tracks.",
        "plays": "Plays of tracks with the genre. A play counts for each of its track's genres.",
        "main_plays": "Plays of tracks with the genre as their main genre. Each play counts for one genre only.",
        "share_of_plays": "main_plays / all plays in gold.plays.",
        "total_ms_played": "Milliseconds played of tracks with the genre.",
        "finish_rate": "Share of the plays of tracks with the genre that were finished.",
        "skip_rate": "Share of the plays of tracks with the genre that were skipped_early.",
        "affinity": "affinity of gold.track_stats summed over the genre's tracks: how much you like the genre lately.",
        "first_played_at": "Start of the first play of a track with the genre, UTC.",
        "last_played_at": "Start of the last play of a track with the genre, UTC.",
    },
)

GENRE_MONTHS = Table(
    name="genre_months",
    key="month, genre",
    description="Plays per month and main genre: how your taste changed over time. Each play counts for its track's main genre only.",
    sql="""
SELECT month, genre, plays, tracks, total_ms_played, plays / sum(plays) OVER (PARTITION BY month) AS share_of_plays, finish_rate
FROM (
    SELECT
        CAST(date_trunc('month', local_date) AS DATE) AS month,
        genre,
        count(*) AS plays,
        count(DISTINCT track_id) AS tracks,
        sum(ms_played) AS total_ms_played,
        avg(CAST(finished AS INTEGER)) AS finish_rate
    FROM gold.plays
    GROUP BY 1, 2
)
ORDER BY month, plays DESC, genre
""",
    columns={
        "month": "First day of the month, in local time.",
        "genre": "Main genre. NULL for plays of tracks without a genre.",
        "plays": "Plays that month of tracks with this main genre.",
        "tracks": "Different tracks among them.",
        "total_ms_played": "Milliseconds played.",
        "share_of_plays": "plays / all plays that month.",
        "finish_rate": "Share of the plays that were finished.",
    },
)

# In build order: each table is built from silver, techlog and the tables before it.
TABLES = (
    TRACKS,
    TRACK_GENRES,
    PLAYS,
    SESSIONS,
    TRACK_TRANSITIONS,
    FEEDBACK_EVENTS,
    EXPOSURES,
    TRACK_STATS,
    ARTISTS,
    GENRES,
    GENRE_MONTHS,
)


def table_sql(con: duckdb.DuckDBPyConnection, table: Table) -> str:
    if table.branches is None:
        return table.sql
    found = {row[0] for row in con.execute("SELECT table_name FROM duckdb_tables() WHERE schema_name = 'techlog'").fetchall()}
    branches = [sql for name, sql in table.branches.items() if name in found]
    missing = [name for name in table.branches if name not in found]
    if missing:
        print(f"Note: no techlog.{', techlog.'.join(missing)} in this export, so gold.{table.name} has nothing from them")
    return table.sql.replace("{branches}", "\n    UNION ALL".join(branches) or table.empty)


def build(con: duckdb.DuckDBPyConnection, genres_csv: Path) -> None:
    params = {
        "genres_csv": genres_csv.as_posix(),
        "not_music_genres": NOT_MUSIC_GENRES,
        "not_music_pattern": NOT_MUSIC_PATTERN,
        "user_ended": USER_ENDED,
        "skip_ms": SKIP_MS,
        "valid_share": VALID_SHARE,
        "test_share": TEST_SHARE,
        "half_life_days": AFFINITY_HALF_LIFE_DAYS,
        "timezone": HOME_TIMEZONE,
    }
    con.execute("CREATE SCHEMA IF NOT EXISTS gold")
    for table in TABLES:
        sql = table_sql(con, table)
        used = {name: value for name, value in params.items() if re.search(rf"\${name}\b", sql)}
        con.execute(f"CREATE OR REPLACE TABLE gold.{table.name} AS {sql}", used)

        duplicates = con.execute(f"SELECT count(*) - count(DISTINCT ({table.key})) FROM gold.{table.name}").fetchone()[0]
        if duplicates:
            raise RuntimeError(f"gold.{table.name} has {duplicates:,} rows with the same {table.key} as another row")
        columns = [row[0] for row in con.execute(f"DESCRIBE gold.{table.name}").fetchall()]
        if set(columns) != set(table.columns):
            raise RuntimeError(
                f"gold.{table.name}: columns without a description: {sorted(set(columns) - set(table.columns))}; "
                f"descriptions without a column: {sorted(set(table.columns) - set(columns))}"
            )
        con.execute(f"COMMENT ON TABLE gold.{table.name} IS {literal(table.description)}")
        for column, text in table.columns.items():
            con.execute(f"COMMENT ON COLUMN gold.{table.name}.{ident(column)} IS {literal(text)}")

    plays, silver_plays, session_plays = con.execute(
        "SELECT (SELECT count(*) FROM gold.plays), (SELECT count(*) FROM silver.plays), (SELECT sum(plays) FROM gold.sessions)"
    ).fetchone()
    if not plays == silver_plays == session_plays:
        raise RuntimeError(f"gold.plays has {plays:,} plays, silver.plays {silver_plays:,}, gold.sessions {session_plays:,}")


def report(con: duckdb.DuckDBPyConnection, db_path: Path, genres_csv: Path) -> None:
    print(f"Gold layer built in {db_path}:")
    for table in TABLES:
        rows = con.execute(f"SELECT count(*) FROM gold.{table.name}").fetchone()[0]
        print(f"  gold.{table.name:<19}{rows:>9,}")
    for split, first, last, plays in con.execute(
        "SELECT split, min(local_date), max(local_date), count(*) FROM gold.plays GROUP BY split ORDER BY min(started_at)"
    ).fetchall():
        print(f"  {split:<5} split: {plays:>7,} plays, {first} to {last}")
    not_in_csv = con.execute(
        "SELECT count(*) FROM gold.tracks WHERE is_played AND track_id NOT IN (SELECT track_id FROM read_csv($csv, header = true))",
        {"csv": genres_csv.as_posix()},
    ).fetchone()[0]
    if not_in_csv:
        print(f"Note: {not_in_csv:,} played tracks are not in {genres_csv.name}, so they have no genres yet (run fetch_genres.py)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="DuckDB file built by build_database.py")
    parser.add_argument("--genres", type=Path, default=DEFAULT_GENRES, help="CSV written by fetch_genres.py")
    args = parser.parse_args()
    if not args.db.exists():
        sys.exit(f"Database not found: {args.db} (run build_database.py)")
    if not args.genres.exists():
        sys.exit(f"Genres not found: {args.genres} (run fetch_genres.py)")

    con = duckdb.connect(str(args.db))
    try:
        con.begin()
        try:
            build(con, args.genres)
            con.commit()
        except Exception:
            con.rollback()
            raise
        report(con, args.db.resolve(), args.genres)
    finally:
        con.close()


if __name__ == "__main__":
    main()
