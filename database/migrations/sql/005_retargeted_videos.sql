CREATE TABLE IF NOT EXISTS retargeted_videos (
    id               INTEGER PRIMARY KEY,
    original_id      INTEGER REFERENCES original_videos(id) ON DELETE SET NULL,
    hand_id          INTEGER REFERENCES hands(id) ON DELETE SET NULL,
    filename         TEXT    NOT NULL,
    full_path        TEXT    NOT NULL UNIQUE,
    date_created     TEXT    NOT NULL DEFAULT (datetime('now')),
    min_cutoff       REAL,
    beta             REAL,
    cam_distance     REAL,
    cam_yaw          REAL,
    cam_pitch        REAL,
    model_id         INTEGER REFERENCES models(id) ON DELETE SET NULL
);
