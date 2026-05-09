CREATE TABLE IF NOT EXISTS usage_sessions (
    id               SERIAL      PRIMARY KEY,
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at         TIMESTAMPTZ,
    hand_id          INTEGER     REFERENCES hands(id) ON DELETE SET NULL,
    min_cutoff       REAL,
    beta             REAL,
    frames_processed INTEGER
);
