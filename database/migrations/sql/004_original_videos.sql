CREATE TABLE IF NOT EXISTS original_videos (
    id           SERIAL      PRIMARY KEY,
    filename     TEXT        NOT NULL UNIQUE,
    full_path    TEXT        NOT NULL,
    date_added   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    duration_sec REAL
);
