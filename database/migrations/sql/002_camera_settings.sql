CREATE TABLE IF NOT EXISTS camera_settings (
    hand_id      INTEGER PRIMARY KEY REFERENCES hands(id) ON DELETE CASCADE,
    cam_distance REAL    NOT NULL DEFAULT 0.7,
    cam_yaw      REAL    NOT NULL DEFAULT 45.0,
    cam_pitch    REAL    NOT NULL DEFAULT -30.0,
    updated_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);
