CREATE TABLE IF NOT EXISTS models (
    id              INTEGER PRIMARY KEY,
    hand_id         INTEGER NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    run_id          TEXT    NOT NULL,
    checkpoint_path TEXT    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
