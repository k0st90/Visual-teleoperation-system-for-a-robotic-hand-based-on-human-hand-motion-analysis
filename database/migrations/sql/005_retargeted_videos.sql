CREATE TABLE IF NOT EXISTS retargeted_videos (
    id               SERIAL      PRIMARY KEY,
    original_id      INTEGER     REFERENCES original_videos(id) ON DELETE SET NULL,
    hand_id          INTEGER     REFERENCES hands(id) ON DELETE SET NULL,
    filename         TEXT        NOT NULL UNIQUE,
    full_path        TEXT        NOT NULL,
    date_created     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    duration_sec     REAL,
    min_cutoff       REAL,
    beta             REAL,
    cam_distance     REAL,
    cam_yaw          REAL,
    cam_pitch        REAL,
    model_id         INTEGER     REFERENCES models(id) ON DELETE SET NULL
);
