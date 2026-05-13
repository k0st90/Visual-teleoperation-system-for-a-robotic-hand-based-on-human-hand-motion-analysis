CREATE TABLE IF NOT EXISTS usage_sessions (
    id          INTEGER PRIMARY KEY,
    started_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    ended_at    TEXT,
    hand_id     INTEGER REFERENCES hands(id) ON DELETE SET NULL,
    min_cutoff  REAL,
    beta        REAL
);
