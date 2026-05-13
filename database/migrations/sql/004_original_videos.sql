CREATE TABLE IF NOT EXISTS original_videos (
    id           INTEGER PRIMARY KEY,
    filename     TEXT    NOT NULL,
    full_path    TEXT    NOT NULL UNIQUE,
    date_added   TEXT    NOT NULL DEFAULT (datetime('now')),
    duration_sec REAL
);
