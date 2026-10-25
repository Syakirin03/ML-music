"""Build a DuckDB database from Spotify's Extended Streaming History and Technical Log exports.

    python scripts/build_database.py [--data-dir DIR] [--techlog-dir DIR] [--db FILE]

Every run rebuilds these objects from the exports, in one transaction:

    bronze.streams_raw             table  every streaming event exactly as exported (duplicates kept), plus source file and load time
    bronze.techlog_raw             table  every technical log record exactly as exported (duplicates kept), plus event type, source file and load time
    bronze.techlog_recipients_raw  table  recipients.json from the technical log export
    silver.streams                 table  every streaming event (music and podcasts), typed, exact duplicates removed
    silver.plays                   view   music plays with start time, local time and listening sessions
    silver.tracks                  table  one row per track in either export: latest metadata, play count, average ms_played
    silver.episodes                table  one row per podcast episode in either export
    silver.devices                 table  one row per device that wrote technical log records
    silver.log_event_types         table  one row per technical log event type, with Spotify's description
    silver.log_events              table  every technical log record once: when, which device and app, which playback and stream
    silver.log_references          table  every Spotify URI a log record mentions, linked to tracks and episodes
    silver.playbacks               table  one row per playback id in the technical log
    silver.log_recipients          table  companies listed in recipients.json
    techlog.<EventType>            tables one per technical log event type: its own fields as typed columns

Silver and techlog are built from bronze, not from the JSON files. Silver
tables are linked by foreign keys; DuckDB only supports those within one
schema, so techlog tables join silver.log_events on event_id without a
declared key. The techlog schema is rebuilt as a whole. Tables you create in
bronze or silver are left alone, so keep enrichment data (audio features,
genres, embeddings, ...) in your own tables keyed by track_id. Do not give them
FOREIGN KEY constraints to the tables above: those would block the rebuild.
"""

import argparse
import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
# The project folder, for recommender.config: your settings in config.toml.
sys.path.insert(0, str(ROOT))
from recommender.config import CONFIG

DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "Spotify Extended Streaming History"
DEFAULT_TECHLOG_DIR = ROOT / "data" / "raw" / "Spotify Technical Log Information"
DEFAULT_DOCS = ROOT / "data" / "external" / "techlog_docs.json"
DEFAULT_DB = ROOT / "data" / "processed" / "spotify.duckdb"

# conn_country comes from IP addresses and is often wrong (travel, a VPN, a mobile network), so every
# play's local time uses one time zone: yours, from config.toml, else your computer's.
HOME_TIMEZONE = CONFIG["home"]["timezone"] or duckdb.sql("SELECT current_setting('TimeZone')").fetchone()[0]

# A pause longer than this between two plays starts a new listening session.
SESSION_GAP_MINUTES = 30

# Fields in the export and the types they are read as.
SOURCE_COLUMNS = {
    "ts": "VARCHAR",
    "platform": "VARCHAR",
    "ms_played": "BIGINT",
    "conn_country": "VARCHAR",
    "ip_addr": "VARCHAR",
    "master_metadata_track_name": "VARCHAR",
    "master_metadata_album_artist_name": "VARCHAR",
    "master_metadata_album_album_name": "VARCHAR",
    "spotify_track_uri": "VARCHAR",
    "episode_name": "VARCHAR",
    "episode_show_name": "VARCHAR",
    "spotify_episode_uri": "VARCHAR",
    "audiobook_title": "VARCHAR",
    "audiobook_uri": "VARCHAR",
    "audiobook_chapter_uri": "VARCHAR",
    "audiobook_chapter_title": "VARCHAR",
    "reason_start": "VARCHAR",
    "reason_end": "VARCHAR",
    "shuffle": "BOOLEAN",
    "skipped": "BOOLEAN",
    "offline": "BOOLEAN",
    "offline_timestamp": "BIGINT",
    "incognito_mode": "BOOLEAN",
}

# Silver tables, children before parents, so each can be dropped while the
# ones it references still exist. The techlog schema is dropped as a whole.
MANAGED_SILVER_TABLES = (
    "silver.log_references",
    "silver.log_events",
    "silver.playbacks",
    "silver.log_event_types",
    "silver.devices",
    "silver.log_recipients",
    "silver.streams",
    "silver.tracks",
    "silver.episodes",
)

BRONZE_SQL = """
CREATE OR REPLACE TABLE bronze.streams_raw AS
SELECT * EXCLUDE (filename), parse_filename(filename) AS source_file, now() AS loaded_at
FROM read_json($files, format = 'array', columns = {columns}, filename = true)
"""

STAGE_SQL = """
CREATE OR REPLACE TEMP TABLE staged AS
SELECT
    ts::TIMESTAMPTZ AS ts,
    ms_played,
    CASE
        WHEN spotify_track_uri IS NOT NULL THEN 'track'
        WHEN spotify_episode_uri IS NOT NULL THEN 'episode'
        WHEN coalesce(audiobook_chapter_uri, audiobook_uri) IS NOT NULL THEN 'audiobook'
        ELSE 'unknown'
    END AS content_type,
    spotify_track_uri AS track_uri,
    master_metadata_track_name AS track_name,
    master_metadata_album_artist_name AS artist_name,
    master_metadata_album_album_name AS album_name,
    spotify_episode_uri AS episode_uri,
    episode_name,
    episode_show_name AS show_name,
    audiobook_uri,
    audiobook_title,
    audiobook_chapter_uri,
    audiobook_chapter_title,
    reason_start,
    reason_end,
    shuffle,
    skipped,
    offline,
    -- Epoch seconds from late 2022 on, epoch milliseconds before that.
    CASE
        WHEN offline_timestamp >= 100000000000 THEN to_timestamp(offline_timestamp / 1000.0)
        WHEN offline_timestamp > 0 THEN to_timestamp(offline_timestamp)
    END AS offline_timestamp,
    incognito_mode,
    platform,
    CASE
        WHEN lower(platform) LIKE 'android%' THEN 'android'
        WHEN lower(platform) LIKE 'ios%' THEN 'ios'
        WHEN lower(platform) LIKE 'windows%' THEN 'windows'
        WHEN lower(platform) LIKE 'web_player%' THEN 'web'
        WHEN lower(platform) LIKE 'os x%' OR lower(platform) LIKE 'osx%' OR lower(platform) LIKE 'mac%' THEN 'macos'
        WHEN lower(platform) LIKE 'linux%' THEN 'linux'
        WHEN platform IS NULL OR lower(platform) = 'unknown' THEN 'unknown'
        ELSE 'other'
    END AS platform_family,
    conn_country,
    ip_addr,
    source_file LIKE 'Streaming_History_Video%' AS is_video,
    source_file
FROM bronze.streams_raw
"""

STREAMS_DDL = """
CREATE OR REPLACE TABLE silver.streams (
    stream_id               BIGINT PRIMARY KEY,
    ts                      TIMESTAMPTZ NOT NULL,
    ms_played               INTEGER NOT NULL,
    content_type            VARCHAR NOT NULL,
    track_uri               VARCHAR REFERENCES silver.tracks (track_uri),
    track_name              VARCHAR,
    artist_name             VARCHAR,
    album_name              VARCHAR,
    episode_uri             VARCHAR REFERENCES silver.episodes (episode_uri),
    episode_name            VARCHAR,
    show_name               VARCHAR,
    audiobook_uri           VARCHAR,
    audiobook_title         VARCHAR,
    audiobook_chapter_uri   VARCHAR,
    audiobook_chapter_title VARCHAR,
    reason_start            VARCHAR,
    reason_end              VARCHAR,
    shuffle                 BOOLEAN,
    skipped                 BOOLEAN,
    offline                 BOOLEAN,
    offline_timestamp       TIMESTAMPTZ,
    incognito_mode          BOOLEAN,
    platform                VARCHAR,
    platform_family         VARCHAR NOT NULL,
    conn_country            VARCHAR,
    ip_addr                 VARCHAR,
    is_video                BOOLEAN NOT NULL,
    source_file             VARCHAR NOT NULL
)
"""

STREAMS_STAGE_SQL = """
CREATE OR REPLACE TEMP TABLE streams_staged AS
SELECT row_number() OVER (ORDER BY {event_key}) AS stream_id, *
FROM (
    SELECT * FROM staged
    -- The export repeats some events verbatim; keep one copy.
    QUALIFY row_number() OVER (PARTITION BY {event_key} ORDER BY source_file) = 1
)
"""

INSERT_STREAMS_SQL = """
INSERT INTO silver.streams BY NAME
SELECT * FROM streams_staged
ORDER BY stream_id
"""

PLAYS_SQL = """
CREATE OR REPLACE VIEW silver.plays AS
WITH music AS (
    SELECT *, ts - to_milliseconds(ms_played) AS started_at
    FROM silver.streams
    WHERE content_type = 'track'
    -- One play is sometimes logged twice with the same end second; keep the longer record.
    QUALIFY row_number() OVER (PARTITION BY track_uri, ts ORDER BY ms_played DESC, stream_id) = 1
),
gaps AS (
    SELECT *, started_at - lag(ts) OVER (ORDER BY started_at, stream_id) AS gap
    FROM music
),
sessions AS (
    SELECT *,
        CAST(sum((gap IS NULL OR gap > INTERVAL {gap_minutes} MINUTE)::INTEGER)
            OVER (ORDER BY started_at, stream_id) AS INTEGER) AS session_id
    FROM gaps
)
SELECT
    stream_id,
    session_id,
    row_number() OVER (PARTITION BY session_id ORDER BY started_at, stream_id) AS session_position,
    split_part(track_uri, ':', 3) AS track_id,
    track_name,
    artist_name,
    album_name,
    started_at,
    ts AS ended_at,
    timezone('{timezone}', started_at) AS started_at_local,
    hour(timezone('{timezone}', started_at)) AS local_hour,
    isodow(timezone('{timezone}', started_at)) AS local_dow,
    ms_played,
    skipped,
    reason_start,
    reason_end,
    shuffle,
    offline,
    incognito_mode,
    is_video,
    platform_family,
    conn_country
FROM sessions
ORDER BY started_at, stream_id
"""

# ---- Technical log ----

# Fields about where and when a log record was written. silver.log_events keeps
# them as columns; the rest of a record is the event's own fields (message_*).
# timestamp_utc repeats context_time as text.
LOG_ENVELOPE_FIELDS = (
    "context_time",
    "context_receiver_service_timestamp",
    "context_device_manufacturer",
    "context_device_model",
    "context_device_type",
    "context_os_name",
    "context_os_version",
    "context_application_version",
    "context_conn_country",
    "context_user_agent",
    "timestamp_utc",
)

# Columns of silver.log_events. In the techlog tables an event field loses its
# message_ prefix, unless the short name would clash with one of these.
LOG_EVENT_COLUMNS = (
    "event_id", "event_type", "ts", "received_at", "device_id", "os_version", "app_version",
    "conn_country", "user_agent", "playback_id", "stream_id", "source_file",
)

# Fields that hold a track as a 32-character hex id instead of a spotify:track: URI.
TRACK_GID_FIELDS = {"TrackNotPlayed": ("message_play_track", "message_display_track")}

# Columns through which events with a playback id name the track being played.
PLAYBACK_TRACK_COLUMNS = ("track_uri", "entity_uri", "track_id", "track", "play_track")

ZERO_ID = "0" * 32
BASE62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def playback_id_sql(x: str) -> str:
    """Playback ids are 32 hex characters in some event types and base64 of the same 16 bytes in others."""
    return (
        f"CASE WHEN regexp_full_match({x}, '[0-9a-fA-F]{{32}}') THEN nullif(lower({x}), '{ZERO_ID}')"
        f" WHEN regexp_full_match({x}, '[A-Za-z0-9+/]{{22}}==') THEN nullif(lower(hex(from_base64({x}))), '{ZERO_ID}')"
        f" ELSE nullif({x}, '') END"
    )


def track_gid_sql(x: str) -> str:
    """spotify:track: URI for a 32-character hex track id: a track id is the same 128-bit number in base 62."""
    number = f"(('0x' || left({x}, 16))::UBIGINT::UHUGEINT << 64 | ('0x' || right({x}, 16))::UBIGINT::UHUGEINT)"
    powers = ", ".join(str(62**k) for k in range(21, -1, -1))
    digits = f"list_transform([{powers}]::UHUGEINT[], lambda p: substr('{BASE62}', ({number} // p % 62)::INTEGER + 1, 1))"
    return f"CASE WHEN regexp_full_match({x}, '[0-9a-f]{{32}}') THEN 'spotify:track:' || array_to_string({digits}, '') ELSE {x} END"


def iso_timestamp_sql(x: str) -> str:
    """Timestamp from ISO 8601 text, which here also comes with nanoseconds or a trailing [Region/City]."""
    return rf"TRY_CAST(regexp_replace(regexp_replace({x}, '\[[^\]]*\]$', ''), '(\.\d{{6}})\d+', '\1') AS TIMESTAMPTZ)"


def timestamp_sql(kind: str, x: str) -> str:
    # Spotify writes 0 (or 1970-01-01) for times that are not set.
    if kind == "iso":
        return f"nullif({iso_timestamp_sql(x)}, TIMESTAMPTZ '1970-01-01 00:00:00+00')"
    scale = 1000 if kind == "epoch_ms" else 1_000_000
    return f"make_timestamptz(CAST(nullif({x}, 0) AS BIGINT) * {scale})"


LOG_BRONZE_DDL = """
CREATE OR REPLACE TABLE bronze.techlog_raw (
    event_type  VARCHAR NOT NULL,
    record      JSON NOT NULL,
    source_file VARCHAR NOT NULL,
    loaded_at   TIMESTAMPTZ NOT NULL
);
CREATE OR REPLACE TABLE bronze.techlog_recipients_raw (
    document    JSON NOT NULL,
    source_file VARCHAR NOT NULL,
    loaded_at   TIMESTAMPTZ NOT NULL
);
"""

LOG_BRONZE_SQL = r"""
INSERT INTO bronze.techlog_raw
SELECT
    -- Big event types are split over files such as Download.json and Download_1.json.
    regexp_replace(parse_filename(filename, true), '_\d+$', '') AS event_type,
    json(json) AS record,
    parse_filename(filename) AS source_file,
    now() AS loaded_at
FROM read_json_objects($files, format = 'array', filename = true)
"""

LOG_RECIPIENTS_BRONZE_SQL = """
INSERT INTO bronze.techlog_recipients_raw
SELECT json(json) AS document, parse_filename(filename) AS source_file, now() AS loaded_at
FROM read_json_objects($file, filename = true)
"""

LOG_STAGE_SQL = """
CREATE OR REPLACE TEMP TABLE log_staged AS
WITH unique_records AS (
    SELECT * FROM bronze.techlog_raw
    -- The export repeats some records verbatim, also across the files of one event type; keep one copy.
    QUALIFY row_number() OVER (PARTITION BY event_type, record ORDER BY source_file) = 1
),
typed AS (
    SELECT
        event_type,
        make_timestamptz((record ->> 'context_time')::BIGINT * 1000) AS ts,
        make_timestamptz((record ->> 'context_receiver_service_timestamp')::BIGINT * 1000) AS received_at,
        record ->> 'context_device_manufacturer' AS manufacturer,
        record ->> 'context_device_model' AS model,
        record ->> 'context_device_type' AS device_type,
        record ->> 'context_os_name' AS os_name,
        record ->> 'context_os_version' AS os_version,
        record ->> 'context_application_version' AS app_version,
        record ->> 'context_conn_country' AS conn_country,
        record ->> 'context_user_agent' AS user_agent,
        {playback_id} AS playback_id,
        record ->> 'timestamp_utc' AS timestamp_utc,
        json_merge_patch(record, '{envelope}') AS message,
        source_file,
        record
    FROM unique_records
)
SELECT row_number() OVER (ORDER BY ts, event_type, record) AS event_id, * EXCLUDE (record)
FROM typed
"""

# Value ranges of event fields named like times, to find the ones holding Unix
# epoch numbers or ISO 8601 text.
LOG_TIME_STATS_SQL = r"""
SELECT
    s.event_type,
    f.key,
    count(*) FILTER (WHERE f.type IN ('UBIGINT', 'BIGINT', 'DOUBLE')) AS numbers,
    count(*) FILTER (WHERE f.type IN ('UBIGINT', 'BIGINT', 'DOUBLE') AND TRY_CAST(f.value AS DOUBLE) <> 0) AS nonzero,
    count(*) FILTER (WHERE f.type IN ('UBIGINT', 'BIGINT', 'DOUBLE') AND TRY_CAST(f.value AS DOUBLE) BETWEEN 1e12 AND 4e12) AS epoch_ms,
    count(*) FILTER (WHERE f.type IN ('UBIGINT', 'BIGINT', 'DOUBLE') AND TRY_CAST(f.value AS DOUBLE) BETWEEN 1e9 AND 4e9) AS epoch_s,
    count(*) FILTER (WHERE f.type = 'VARCHAR') AS strings,
    count(*) FILTER (WHERE f.type = 'VARCHAR' AND {iso} IS NOT NULL) AS iso
FROM log_staged s, json_each(s.message) f
WHERE f.key ILIKE '%time%' OR f.key ILIKE '%\_at' ESCAPE '\'
GROUP BY ALL
"""

LOG_REFS_SQL = r"""
CREATE OR REPLACE TEMP TABLE log_refs AS
WITH found AS (
    SELECT s.event_id, s.event_type, f.key AS field, substr(t.fullkey, 2) AS path, t.atom ->> '$' AS uri
    FROM log_staged s, json_each(s.message) f, json_tree(f.value) t
    WHERE t.type = 'VARCHAR' AND regexp_matches(t.atom ->> '$', '^spotify:[a-z][a-z0-9-]*:')
    {gid_references}
)
SELECT
    found.event_id,
    c.column_name || regexp_replace(found.path, '\[\d+\]', '[]', 'g') AS field,
    CAST(nullif(regexp_extract(found.path, '.*\[(\d+)\]', 1), '') AS INTEGER) + 1 AS position,
    found.uri,
    split_part(found.uri, ':', 2) AS entity_type,
    CASE WHEN regexp_full_match(found.uri, 'spotify:track:[0-9A-Za-z]{{22}}') THEN split_part(found.uri, ':', 3) END AS track_id,
    CASE WHEN regexp_full_match(found.uri, 'spotify:episode:[0-9A-Za-z]{{22}}') THEN split_part(found.uri, ':', 3) END AS episode_id
FROM found
JOIN log_columns c ON c.event_type = found.event_type AND c.field = found.field
"""

STREAM_LINKS_SQL = """
CREATE OR REPLACE TEMP TABLE stream_links AS
WITH logged AS (
    SELECT *, row_number() OVER (PARTITION BY uri, ms_played, end_second ORDER BY event_id) AS nth
    FROM (
        SELECT
            event_id,
            message ->> 'message_content_uri' AS uri,
            (message ->> 'message_ms_played')::BIGINT AS ms_played,
            date_trunc('second', ts) AS end_second
        FROM log_staged
        WHERE event_type = 'RawCoreStream'
    )
),
streamed AS (
    SELECT *, row_number() OVER (PARTITION BY uri, ms_played, end_second ORDER BY stream_id) AS nth
    FROM (
        SELECT stream_id, coalesce(track_uri, episode_uri, audiobook_chapter_uri) AS uri, ms_played, ts AS end_second
        FROM silver.streams
    )
)
-- The streaming history is this log with ts cut to the second: same URI, same
-- ms_played. Identical plays within one second pair up in order.
SELECT logged.event_id, streamed.stream_id
FROM logged JOIN streamed USING (uri, ms_played, end_second, nth)
"""

TRACKS_SQL = """
CREATE OR REPLACE TABLE silver.tracks (
    track_id      VARCHAR PRIMARY KEY,
    track_uri     VARCHAR NOT NULL UNIQUE,
    track_name    VARCHAR,
    artist_name   VARCHAR,
    album_name    VARCHAR,
    play_count    INTEGER NOT NULL,
    avg_ms_played INTEGER
);
INSERT INTO silver.tracks BY NAME
SELECT split_part(track_uri, ':', 3) AS track_id, track_uri, track_name, artist_name, album_name, 0 AS play_count
FROM streams_staged
WHERE content_type = 'track'
-- Track and album names sometimes change between plays; keep the most recent.
QUALIFY row_number() OVER (PARTITION BY track_uri ORDER BY ts DESC, stream_id DESC) = 1;
-- Tracks the technical log mentions (recommended, queued, shuffled, ...) that were never played.
INSERT INTO silver.tracks BY NAME
SELECT DISTINCT track_id, 'spotify:track:' || track_id AS track_uri, 0 AS play_count
FROM log_refs
WHERE track_id IS NOT NULL AND track_id NOT IN (SELECT track_id FROM silver.tracks);
"""

EPISODES_SQL = """
CREATE OR REPLACE TABLE silver.episodes (
    episode_id    VARCHAR PRIMARY KEY,
    episode_uri   VARCHAR NOT NULL UNIQUE,
    episode_name  VARCHAR,
    show_name     VARCHAR,
    play_count    INTEGER NOT NULL,
    avg_ms_played INTEGER
);
INSERT INTO silver.episodes BY NAME
SELECT split_part(episode_uri, ':', 3) AS episode_id, episode_uri, episode_name, show_name, 0 AS play_count
FROM streams_staged
WHERE content_type = 'episode'
QUALIFY row_number() OVER (PARTITION BY episode_uri ORDER BY ts DESC, stream_id DESC) = 1;
INSERT INTO silver.episodes BY NAME
SELECT DISTINCT episode_id, 'spotify:episode:' || episode_id AS episode_uri, 0 AS play_count
FROM log_refs
WHERE episode_id IS NOT NULL AND episode_id NOT IN (SELECT episode_id FROM silver.episodes);
"""

# silver.streams references tracks and episodes, so their play statistics are
# filled in once the streams and silver.plays exist.
PLAY_COUNTS_SQL = """
UPDATE silver.tracks t
SET play_count = p.play_count, avg_ms_played = p.avg_ms_played
FROM (
    SELECT track_id, count(*) AS play_count, CAST(round(avg(ms_played)) AS INTEGER) AS avg_ms_played
    FROM silver.plays
    GROUP BY track_id
) p
WHERE p.track_id = t.track_id;
UPDATE silver.episodes e
SET play_count = s.play_count, avg_ms_played = s.avg_ms_played
FROM (
    SELECT episode_uri, count(*) AS play_count, CAST(round(avg(ms_played)) AS INTEGER) AS avg_ms_played
    FROM silver.streams
    WHERE content_type = 'episode'
    GROUP BY episode_uri
) s
WHERE s.episode_uri = e.episode_uri;
"""

DEVICES_SQL = """
CREATE OR REPLACE TABLE silver.devices (
    device_id     INTEGER PRIMARY KEY,
    manufacturer  VARCHAR,
    model         VARCHAR,
    device_type   VARCHAR,
    os_name       VARCHAR,
    first_seen_at TIMESTAMPTZ NOT NULL,
    last_seen_at  TIMESTAMPTZ NOT NULL
);
INSERT INTO silver.devices BY NAME
SELECT
    row_number() OVER (ORDER BY min(ts), manufacturer, model, device_type, os_name) AS device_id,
    manufacturer, model, device_type, os_name, min(ts) AS first_seen_at, max(ts) AS last_seen_at
FROM log_staged
WHERE coalesce(manufacturer, model, device_type, os_name) IS NOT NULL
GROUP BY manufacturer, model, device_type, os_name;
"""

LOG_EVENT_TYPES_SQL = """
CREATE OR REPLACE TABLE silver.log_event_types (
    event_type     VARCHAR PRIMARY KEY,
    category       VARCHAR,
    description    VARCHAR,
    event_count    INTEGER NOT NULL,
    first_event_at TIMESTAMPTZ NOT NULL,
    last_event_at  TIMESTAMPTZ NOT NULL
);
INSERT INTO silver.log_event_types BY NAME
SELECT s.event_type, d.category, d.description, count(*) AS event_count, min(s.ts) AS first_event_at, max(s.ts) AS last_event_at
FROM log_staged s
LEFT JOIN log_docs d USING (event_type)
GROUP BY ALL;
"""

PLAYBACKS_SQL = """
CREATE OR REPLACE TABLE silver.playbacks (
    playback_id    VARCHAR PRIMARY KEY,
    track_id       VARCHAR REFERENCES silver.tracks (track_id),
    first_event_at TIMESTAMPTZ NOT NULL,
    last_event_at  TIMESTAMPTZ NOT NULL,
    event_count    INTEGER NOT NULL
);
INSERT INTO silver.playbacks BY NAME
SELECT s.playback_id, named.track_id, min(s.ts) AS first_event_at, max(s.ts) AS last_event_at, count(*) AS event_count
FROM log_staged s
LEFT JOIN (
    -- A playback gets a track only when the events that name one agree.
    SELECT e.playback_id, any_value(r.track_id) AS track_id
    FROM log_refs r
    JOIN log_staged e USING (event_id)
    WHERE e.playback_id IS NOT NULL AND r.track_id IS NOT NULL AND r.field IN ({track_columns})
    GROUP BY e.playback_id
    HAVING count(DISTINCT r.track_id) = 1
) named USING (playback_id)
WHERE s.playback_id IS NOT NULL
GROUP BY ALL;
"""

LOG_EVENTS_SQL = """
CREATE OR REPLACE TABLE silver.log_events (
    event_id     BIGINT PRIMARY KEY,
    event_type   VARCHAR NOT NULL REFERENCES silver.log_event_types (event_type),
    ts           TIMESTAMPTZ NOT NULL,
    received_at  TIMESTAMPTZ,
    device_id    INTEGER REFERENCES silver.devices (device_id),
    os_version   VARCHAR,
    app_version  VARCHAR,
    conn_country VARCHAR,
    user_agent   VARCHAR,
    playback_id  VARCHAR REFERENCES silver.playbacks (playback_id),
    stream_id    BIGINT UNIQUE REFERENCES silver.streams (stream_id),
    source_file  VARCHAR NOT NULL
);
INSERT INTO silver.log_events BY NAME
SELECT
    s.event_id, s.event_type, s.ts, s.received_at, d.device_id, s.os_version, s.app_version,
    s.conn_country, s.user_agent, s.playback_id, l.stream_id, s.source_file
FROM log_staged s
LEFT JOIN silver.devices d
    ON d.manufacturer IS NOT DISTINCT FROM s.manufacturer
    AND d.model IS NOT DISTINCT FROM s.model
    AND d.device_type IS NOT DISTINCT FROM s.device_type
    AND d.os_name IS NOT DISTINCT FROM s.os_name
LEFT JOIN stream_links l USING (event_id)
ORDER BY s.event_id;
"""

LOG_REFERENCES_SQL = """
CREATE OR REPLACE TABLE silver.log_references (
    event_id    BIGINT NOT NULL REFERENCES silver.log_events (event_id),
    field       VARCHAR NOT NULL,
    position    INTEGER,
    uri         VARCHAR NOT NULL,
    entity_type VARCHAR NOT NULL,
    track_id    VARCHAR REFERENCES silver.tracks (track_id),
    episode_id  VARCHAR REFERENCES silver.episodes (episode_id)
);
INSERT INTO silver.log_references BY NAME
SELECT * FROM log_refs
ORDER BY event_id, field, position;
"""

LOG_RECIPIENTS_SQL = """
CREATE OR REPLACE TABLE silver.log_recipients (
    recipient_group VARCHAR NOT NULL,
    recipient       VARCHAR NOT NULL,
    PRIMARY KEY (recipient_group, recipient)
);
INSERT INTO silver.log_recipients
SELECT DISTINCT grp.value ->> 'name', member.value ->> '$'
FROM bronze.techlog_recipients_raw r,
    json_each(r.document, '$.recipients') grp,
    json_each(grp.value, '$.members') member;
"""

COMMENTS = {
    "TABLE bronze.streams_raw": "Every event exactly as exported: original field names and values, duplicates kept.",
    "COLUMN bronze.streams_raw.source_file": "Export file the event was read from.",
    "COLUMN bronze.streams_raw.loaded_at": "When the build that loaded this row ran.",
    "TABLE bronze.techlog_raw": "Every technical log record exactly as exported (as minified JSON), duplicates kept.",
    "COLUMN bronze.techlog_raw.event_type": "Event type, from the file name: Download.json and Download_1.json both hold Download records.",
    "COLUMN bronze.techlog_raw.record": "The record with its original field names and values.",
    "COLUMN bronze.techlog_raw.source_file": "Export file the record was read from.",
    "COLUMN bronze.techlog_raw.loaded_at": "When the build that loaded this row ran.",
    "TABLE bronze.techlog_recipients_raw": "recipients.json from the technical log export, as exported.",
    "COLUMN bronze.techlog_recipients_raw.loaded_at": "When the build that loaded this row ran.",
    "TABLE silver.streams": "Every event in the Spotify export (music, podcasts, audiobooks), typed, with exact duplicates removed.",
    "COLUMN silver.streams.stream_id": "Row id in time order. Stable for a given export; may shift when a newer export adds events.",
    "COLUMN silver.streams.ts": "When playback STOPPED (UTC). Playback started at ts - ms_played.",
    "COLUMN silver.streams.ms_played": "Milliseconds actually played; paused time is not counted.",
    "COLUMN silver.streams.content_type": "track, episode, audiobook or unknown.",
    "COLUMN silver.streams.artist_name": "From the export field master_metadata_album_artist_name.",
    "COLUMN silver.streams.offline_timestamp": "Roughly when playback started, by the device clock. Set for nearly every event since late 2022, and for older offline plays.",
    "COLUMN silver.streams.platform_family": "android, ios, windows, macos, linux, web, other or unknown, derived from platform.",
    "COLUMN silver.streams.is_video": "The event came from a Streaming_History_Video file.",
    "COLUMN silver.streams.source_file": "Export file the event was read from.",
    "VIEW silver.plays": (
        "Music plays only, one row per play: same-second duplicate logs collapsed, start and end times, "
        f"local time in {HOME_TIMEZONE}, and sessions split at pauses over {SESSION_GAP_MINUTES} minutes."
    ),
    "TABLE silver.tracks": (
        "One row per track in either export. Played tracks have their most recent name, artist and album; "
        "tracks only seen in the technical log (recommended, queued, shuffled, ...) have play_count 0 and no metadata."
    ),
    "COLUMN silver.tracks.track_id": "Spotify track id, as used by the Spotify Web API.",
    "COLUMN silver.tracks.play_count": "Plays in silver.plays. 0 for tracks only seen in the technical log.",
    "COLUMN silver.tracks.avg_ms_played": "Average milliseconds played per play of this track, over silver.plays (skips included). NULL if never played.",
    "TABLE silver.episodes": "One row per podcast episode in either export. Episodes only seen in the technical log have play_count 0 and no metadata.",
    "COLUMN silver.episodes.play_count": "Streams of this episode in silver.streams. 0 for episodes only seen in the technical log.",
    "COLUMN silver.episodes.avg_ms_played": "Average milliseconds played per stream of this episode. NULL if never played.",
    "TABLE silver.devices": "One row per device that wrote technical log records. Join silver.log_events on device_id.",
    "COLUMN silver.devices.device_id": "Row id in order of first appearance. Stable for a given export; may shift when a newer export adds devices.",
    "COLUMN silver.devices.device_type": "mobile, tablet, desktop, speaker, ...",
    "COLUMN silver.devices.first_seen_at": "First technical log record from this device.",
    "COLUMN silver.devices.last_seen_at": "Last technical log record from this device.",
    "TABLE silver.log_event_types": (
        "One row per technical log event type in the export, with Spotify's category and description from the ReadMe. "
        "The table techlog.<event_type> has each type's own fields."
    ),
    "COLUMN silver.log_event_types.category": "Spotify's data category: USAGE_DATA, USER_DATA, PAYMENT_DATA, ...",
    "TABLE silver.log_events": (
        "Every technical log record once (exact duplicates removed): when, which device and app, and the playback "
        "and stream it belongs to. techlog.<event_type> has the event's own fields, joined on event_id."
    ),
    "COLUMN silver.log_events.event_id": "Row id in time order. Stable for a given export; may shift when a newer export adds records.",
    "COLUMN silver.log_events.ts": "When the client created the event (context_time, UTC).",
    "COLUMN silver.log_events.received_at": "When Spotify's servers received the event. Can be days after ts for events logged offline.",
    "COLUMN silver.log_events.device_id": "Device that logged the event. NULL for web pages and server-side events.",
    "COLUMN silver.log_events.os_version": "Operating system version on the device.",
    "COLUMN silver.log_events.app_version": "Spotify app version.",
    "COLUMN silver.log_events.conn_country": "Country the connection came from (ISO 3166-1 alpha-2), from IP geolocation.",
    "COLUMN silver.log_events.user_agent": "Browser user agent, for events from web pages and the desktop app's web view.",
    "COLUMN silver.log_events.playback_id": "Playback the event belongs to, as 32 lowercase hex characters (the export mixes hex and base64).",
    "COLUMN silver.log_events.stream_id": "RawCoreStream records only: the same play in silver.streams (same URI and ms_played, ts in the same second).",
    "COLUMN silver.log_events.source_file": "Export file the record was read from.",
    "TABLE silver.log_references": (
        "One row per Spotify URI in a technical log record, also inside lists: "
        "which events mention which tracks, episodes, playlists, albums, artists, ..."
    ),
    "COLUMN silver.log_references.field": "Column of techlog.<event_type> holding the URI; [] marks a list, as in tracks[].",
    "COLUMN silver.log_references.position": "1-based position in the list; NULL for a single value.",
    "COLUMN silver.log_references.entity_type": "Second part of the URI: track, episode, playlist, album, artist, show, ...",
    "COLUMN silver.log_references.track_id": "The track, when the URI is spotify:track:<id>.",
    "COLUMN silver.log_references.episode_id": "The episode, when the URI is spotify:episode:<id>.",
    "TABLE silver.playbacks": (
        "One row per playback id in the technical log. The low-level events about one playback "
        "(file choice, downloads, audio route, lyrics, errors) share it."
    ),
    "COLUMN silver.playbacks.track_id": "The track played, when events of the playback name it (lyrics, errors, TrackNotPlayed, ...) and agree.",
    "COLUMN silver.playbacks.event_count": "Technical log records with this playback id.",
    "TABLE silver.log_recipients": "Companies listed in recipients.json as possible recipients of the technical log data.",
}

# Comments for the columns every techlog table starts with, and notes for converted columns.
LOG_TABLE_COLUMN_COMMENTS = {
    "event_id": "Row in silver.log_events; join on event_id for the device, app version and country.",
    "ts": "When the client created the event (UTC).",
    "stream_id": "The same play in silver.streams, when the streaming history has it.",
}
LOG_CONVERSION_NOTES = {
    "epoch_ms": "Converted from Unix epoch milliseconds; 0 reads as NULL.",
    "epoch_s": "Converted from Unix epoch seconds; 0 reads as NULL.",
    "iso": "Converted from ISO 8601 text.",
    "gid": "Hex track ids are converted to spotify:track: URIs.",
    "playback": "As 32 lowercase hex characters (the export mixes hex and base64); joins silver.playbacks.",
}


def ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def find_export_files(data_dir: Path) -> list[str]:
    files = sorted(data_dir.glob("Streaming_History_*.json"))
    if not files:
        sys.exit(f"No Streaming_History_*.json files found in {data_dir}")
    return [f.as_posix() for f in files]


def find_techlog_files(log_dir: Path) -> tuple[list[str], str | None]:
    if not log_dir.is_dir():
        print(f"Note: no technical log folder at {log_dir}; the log tables will be empty")
        return [], None
    events = sorted(p.as_posix() for p in log_dir.glob("*.json") if p.name.lower() != "recipients.json")
    recipients = log_dir / "recipients.json"
    return events, recipients.as_posix() if recipients.exists() else None


def load_docs(path: Path) -> dict:
    if not path.exists():
        print(f"Note: {path.name} not found, so the log tables get no descriptions (run extract_techlog_docs.py)")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["events"]


def check_fields(con: duckdb.DuckDBPyConnection, files: list[str]) -> None:
    found = {
        row[0]
        for row in con.execute(
            "SELECT DISTINCT unnest(json_keys(json)) FROM read_json_objects($files, format = 'array')",
            {"files": files},
        ).fetchall()
    }
    unknown = sorted(found - SOURCE_COLUMNS.keys())
    missing = sorted(SOURCE_COLUMNS.keys() - found)
    if unknown:
        print(f"WARNING: fields in the export that are not stored: {', '.join(unknown)}")
    if missing:
        print(f"Note: fields absent from the export (stored as NULL): {', '.join(missing)}")


def column_name(field: str, taken: set[str]) -> str:
    if field == "message_playback_id":
        return "playback_id"
    short = field.removeprefix("message_")
    return field if short in LOG_EVENT_COLUMNS or short in taken else short


def transform_type(structure):
    """json_transform type for a json_group_structure entry: always-null fields stay JSON."""
    if isinstance(structure, dict):
        return {key: transform_type(value) for key, value in structure.items()}
    if isinstance(structure, list):
        return [transform_type(structure[0]) if structure else "JSON"]
    return {"NULL": "JSON", "UBIGINT": "BIGINT"}.get(structure, structure)


def plan_log_tables(con: duckdb.DuckDBPyConnection) -> dict[str, list[dict]]:
    """Columns of each techlog table, from the JSON structure and value ranges of the event's fields."""
    time_kinds = {}
    stats = con.execute(LOG_TIME_STATS_SQL.format(iso=iso_timestamp_sql("(f.value ->> '$')"))).fetchall()
    for event_type, field, numbers, nonzero, epoch_ms, epoch_s, strings, iso in stats:
        if numbers and not strings and nonzero and epoch_ms == nonzero:
            time_kinds[event_type, field] = "epoch_ms"
        elif numbers and not strings and nonzero and epoch_s == nonzero:
            time_kinds[event_type, field] = "epoch_s"
        elif strings and not numbers and iso == strings:
            time_kinds[event_type, field] = "iso"

    plans = {}
    structures = con.execute(
        "SELECT event_type, json_group_structure(message) FROM log_staged GROUP BY event_type ORDER BY event_type"
    ).fetchall()
    for event_type, structure in structures:
        columns, taken = [], set(LOG_TABLE_COLUMN_COMMENTS)
        for field, type_ in json.loads(structure).items():
            name = column_name(field, taken)
            taken.add(name)
            if field == "message_playback_id":
                kind = "playback"
            elif field in TRACK_GID_FIELDS.get(event_type, ()):
                kind = "gid"
            else:
                kind = time_kinds.get((event_type, field))
            columns.append({"field": field, "name": name, "type": transform_type(type_), "kind": kind})
        plans[event_type] = columns

    con.execute("CREATE OR REPLACE TEMP TABLE log_columns (event_type VARCHAR, field VARCHAR, column_name VARCHAR)")
    rows = [(event_type, c["field"], c["name"]) for event_type, columns in plans.items() for c in columns]
    if rows:
        con.executemany("INSERT INTO log_columns VALUES (?, ?, ?)", rows)
    return plans


def gid_references_sql() -> str:
    parts = []
    for event_type, fields in TRACK_GID_FIELDS.items():
        for field in fields:
            value = f"(message ->> {literal(field)})"
            parts.append(
                f"UNION ALL SELECT event_id, event_type, {literal(field)}, '', {track_gid_sql(value)} FROM log_staged"
                f" WHERE event_type = {literal(event_type)} AND regexp_full_match({value}, '[0-9a-f]{{32}}')"
            )
    return "\n    ".join(parts)


def create_log_table(con: duckdb.DuckDBPyConnection, event_type: str, columns: list[dict], doc: dict) -> None:
    leading = ["event_id", "ts"] + (["stream_id"] if event_type == "RawCoreStream" else [])
    select = list(leading)
    for c in columns:
        source = f"m.{ident(c['field'])}"
        if c["kind"] == "playback":
            expr = "playback_id"
        elif c["kind"] == "gid":
            expr = track_gid_sql(source)
        elif c["kind"]:
            expr = timestamp_sql(c["kind"], source)
        else:
            expr = source
        select.append(f"{expr} AS {ident(c['name'])}")
    structure = {c["field"]: c["type"] for c in columns if c["kind"] != "playback"}
    typed = f", json_transform(s.message, {literal(json.dumps(structure))}) AS m" if structure else ""
    table = f"techlog.{ident(event_type)}"
    con.execute(f"""
        CREATE TABLE {table} AS
        SELECT {", ".join(select)}
        FROM (
            SELECT s.event_id, s.ts, l.stream_id, s.playback_id{typed}
            FROM log_staged s
            LEFT JOIN stream_links l USING (event_id)
            WHERE s.event_type = {literal(event_type)}
        )
        ORDER BY event_id
    """)

    statements = []
    if doc:
        description = "{description} ({category})".format(**doc)
        statements.append(f"COMMENT ON TABLE {table} IS {literal(description)}")
    for name in leading:
        statements.append(f"COMMENT ON COLUMN {table}.{ident(name)} IS {literal(LOG_TABLE_COLUMN_COMMENTS[name])}")
    for c in columns:
        text = " ".join(t for t in (doc.get("fields", {}).get(c["field"]), LOG_CONVERSION_NOTES.get(c["kind"])) if t)
        if text:
            statements.append(f"COMMENT ON COLUMN {table}.{ident(c['name'])} IS {literal(text)}")
    con.execute(";\n".join(statements))


def check_log_tables(con: duckdb.DuckDBPyConnection, plans: dict[str, list[dict]]) -> list[str]:
    """Fields whose values did not fit the type their table gives them (json_transform reads those as NULL)."""
    problems = []
    for event_type, columns in plans.items():
        plain = [c for c in columns if c["kind"] is None]
        if not plain:
            continue
        present = ", ".join(
            "count(*) FILTER (WHERE json_type(message, {}) <> 'NULL')".format(literal("$." + json.dumps(c["field"])))
            for c in plain
        )
        typed = ", ".join("count({})".format(ident(c["name"])) for c in plain)
        before = con.execute(f"SELECT {present} FROM log_staged WHERE event_type = ?", [event_type]).fetchone()
        after = con.execute(f"SELECT {typed} FROM techlog.{ident(event_type)}").fetchone()
        for c, n_before, n_after in zip(plain, before, after):
            if n_after != n_before:
                problems.append(
                    f"techlog.{event_type}.{c['name']}: {n_before - n_after:,} of {n_before:,} values "
                    f"do not fit {json.dumps(c['type'])} and read as NULL"
                )
    return problems


def build(
    con: duckdb.DuckDBPyConnection, files: list[str], log_files: list[str], recipients_file: str | None, docs: dict
) -> list[str]:
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    con.execute("CREATE SCHEMA IF NOT EXISTS silver")
    con.execute("DROP SCHEMA IF EXISTS techlog CASCADE")
    con.execute("DROP VIEW IF EXISTS silver.plays")
    for table in MANAGED_SILVER_TABLES:
        con.execute(f"DROP TABLE IF EXISTS {table}")

    columns = "{" + ", ".join(f"'{name}': '{type_}'" for name, type_ in SOURCE_COLUMNS.items()) + "}"
    con.execute(BRONZE_SQL.format(columns=columns), {"files": files})
    con.execute(LOG_BRONZE_DDL)
    if log_files:
        con.execute(LOG_BRONZE_SQL, {"files": log_files})
    if recipients_file:
        con.execute(LOG_RECIPIENTS_BRONZE_SQL, {"file": recipients_file})

    # Staging: temp tables the silver tables are filled from.
    con.execute(STAGE_SQL)
    staged_columns = [row[0] for row in con.execute("DESCRIBE staged").fetchall()]
    event_key = ", ".join(c for c in staged_columns if c != "source_file")
    con.execute(STREAMS_STAGE_SQL.format(event_key=event_key))
    envelope = json.dumps({field: None for field in LOG_ENVELOPE_FIELDS})
    con.execute(LOG_STAGE_SQL.format(playback_id=playback_id_sql("(record ->> 'message_playback_id')"), envelope=envelope))
    plans = plan_log_tables(con)
    con.execute(LOG_REFS_SQL.format(gid_references=gid_references_sql()))
    con.execute("CREATE OR REPLACE TEMP TABLE log_docs (event_type VARCHAR, category VARCHAR, description VARCHAR)")
    if docs:
        con.executemany("INSERT INTO log_docs VALUES (?, ?, ?)", [(k, v["category"], v["description"]) for k, v in docs.items()])

    # Referenced tables before the tables that reference them.
    con.execute(TRACKS_SQL)
    con.execute(EPISODES_SQL)
    con.execute(STREAMS_DDL)
    con.execute(INSERT_STREAMS_SQL)
    con.execute(PLAYS_SQL.format(timezone=HOME_TIMEZONE, gap_minutes=SESSION_GAP_MINUTES))
    con.execute(PLAY_COUNTS_SQL)
    con.execute(STREAM_LINKS_SQL)
    con.execute(DEVICES_SQL)
    con.execute(LOG_EVENT_TYPES_SQL)
    con.execute(PLAYBACKS_SQL.format(track_columns=", ".join(map(literal, PLAYBACK_TRACK_COLUMNS))))
    con.execute(LOG_EVENTS_SQL)
    con.execute(LOG_REFERENCES_SQL)
    con.execute(LOG_RECIPIENTS_SQL)

    con.execute("CREATE SCHEMA techlog")
    for event_type, columns in plans.items():
        create_log_table(con, event_type, columns, docs.get(event_type, {}))

    for target, text in COMMENTS.items():
        con.execute(f"COMMENT ON {target} IS {literal(text)}")
    return check_log_tables(con, plans)


def report(con: duckdb.DuckDBPyConnection, db_path: Path, docs: dict, problems: list[str]) -> None:
    def scalar(sql: str):
        return con.execute(sql).fetchone()[0]

    def line(name: str, count: int, note: str = "") -> None:
        print(f"  {name:<23}{count:>9,}" + (f"  ({note})" if note else ""))

    files, raw, first, last = con.execute(
        "SELECT count(DISTINCT source_file), count(*), min(ts)::TIMESTAMPTZ::DATE, max(ts)::TIMESTAMPTZ::DATE"
        " FROM bronze.streams_raw"
    ).fetchone()
    streams = scalar("SELECT count(*) FROM silver.streams")
    music = scalar("SELECT count(*) FROM silver.streams WHERE content_type = 'track'")
    plays = scalar("SELECT count(*) FROM silver.plays")
    tracks, played = con.execute("SELECT count(*), count(*) FILTER (WHERE play_count > 0) FROM silver.tracks").fetchone()
    episodes, streamed = con.execute("SELECT count(*), count(*) FILTER (WHERE play_count > 0) FROM silver.episodes").fetchone()
    print(f"Loaded {files} streaming history files ({first} to {last})")
    line("bronze.streams_raw", raw, "as exported")
    line("silver.streams", streams, f"{raw - streams:,} exact duplicates dropped")
    line("silver.plays", plays, f"music only; {music - plays:,} same-second duplicate logs collapsed")
    line("silver.tracks", tracks, f"{played:,} played, {tracks - played:,} only in the technical log")
    line("silver.episodes", episodes, f"{streamed:,} played, {episodes - streamed:,} only in the technical log")

    log_raw = scalar("SELECT count(*) FROM bronze.techlog_raw")
    if log_raw:
        log_files = scalar("SELECT count(DISTINCT source_file) FROM bronze.techlog_raw")
        events, types = con.execute("SELECT count(*), count(DISTINCT event_type) FROM silver.log_events").fetchone()
        first, last = con.execute("SELECT min(ts)::DATE, max(ts)::DATE FROM silver.log_events").fetchone()
        refs, track_refs, episode_refs = con.execute(
            "SELECT count(*), count(track_id), count(episode_id) FROM silver.log_references"
        ).fetchone()
        playbacks, with_track = con.execute("SELECT count(*), count(track_id) FROM silver.playbacks").fetchone()
        tables = scalar("SELECT count(*) FROM duckdb_tables() WHERE schema_name = 'techlog'")
        print(f"Loaded {log_files} technical log files: {types} event types ({first} to {last})")
        line("bronze.techlog_raw", log_raw, "as exported")
        line("silver.log_events", events, f"{log_raw - events:,} exact duplicates dropped")
        line("silver.log_references", refs, f"{track_refs:,} to tracks, {episode_refs:,} to episodes")
        line("silver.playbacks", playbacks, f"{with_track:,} with a known track")
        line("silver.devices", scalar("SELECT count(*) FROM silver.devices"))
        line("silver.log_recipients", scalar("SELECT count(*) FROM silver.log_recipients"))
        line("techlog tables", tables, "one per event type")
        logged, linked, later = con.execute(
            "SELECT count(*), count(stream_id), count(*) FILTER (WHERE stream_id IS NULL AND ts >= (SELECT max(ts) FROM silver.streams))"
            " FROM silver.log_events WHERE event_type = 'RawCoreStream'"
        ).fetchone()
        print(
            f"  RawCoreStream: {linked:,} of {logged:,} plays linked to silver.streams; "
            f"{later:,} of the other {logged - linked:,} are after the streaming history ends"
        )

        if docs:
            undocumented = [r[0] for r in con.execute("SELECT event_type FROM silver.log_event_types WHERE description IS NULL ORDER BY 1").fetchall()]
            if undocumented:
                print(f"Note: techlog_docs.json does not describe {', '.join(undocumented)}")
        mismatched = scalar("SELECT count(*) FROM log_staged WHERE TRY_CAST(timestamp_utc AS TIMESTAMPTZ) IS DISTINCT FROM ts AND timestamp_utc IS NOT NULL")
        if mismatched:
            print(f"WARNING: {mismatched:,} log records have a timestamp_utc that differs from context_time (ts uses context_time)")
        unmapped = [r[0] for r in con.execute(
            r"SELECT DISTINCT f.key FROM log_staged, json_each(message) f WHERE f.key LIKE 'context\_%' ESCAPE '\' ORDER BY 1"
        ).fetchall()]
        if unmapped:
            print(f"WARNING: context fields not mapped to silver.log_events columns (kept in the techlog tables): {', '.join(unmapped)}")

    unknown = scalar("SELECT count(*) FROM silver.streams WHERE content_type = 'unknown'")
    if unknown:
        print(f"WARNING: {unknown:,} events have no track, episode or audiobook URI")
    bad_ids = scalar("SELECT count(*) FROM silver.tracks WHERE NOT regexp_full_match(track_id, '[0-9A-Za-z]{22}')")
    if bad_ids:
        print(f"WARNING: {bad_ids:,} track URIs are not in the spotify:track:<id> form")
    for problem in problems:
        print(f"WARNING: {problem}")
    print(f"Database: {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="folder with the streaming history JSON files")
    parser.add_argument("--techlog-dir", type=Path, default=DEFAULT_TECHLOG_DIR, help="folder with the technical log JSON files")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="DuckDB file to create or update")
    args = parser.parse_args()

    try:
        duckdb.sql(f"SELECT timezone({literal(HOME_TIMEZONE)}, TIMESTAMPTZ '2020-01-01 00:00:00+00')").fetchone()
    except duckdb.Error:
        sys.exit(f"Unknown time zone {HOME_TIMEZONE!r} in config.toml: use a name such as Europe/London or Asia/Tokyo")
    if not CONFIG["home"]["timezone"]:
        print(f"Note: local times use your computer's time zone, {HOME_TIMEZONE}; set timezone in config.toml to choose one")
    files = find_export_files(args.data_dir)
    log_files, recipients_file = find_techlog_files(args.techlog_dir)
    docs = load_docs(DEFAULT_DOCS)
    con = duckdb.connect(str(args.db))
    try:
        check_fields(con, files)
        con.begin()
        try:
            problems = build(con, files, log_files, recipients_file, docs)
            con.commit()
        except Exception:
            con.rollback()
            raise
        report(con, args.db.resolve(), docs, problems)
    finally:
        con.close()


if __name__ == "__main__":
    main()
