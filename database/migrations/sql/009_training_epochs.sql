CREATE TABLE IF NOT EXISTS training_epochs (
    id             INTEGER PRIMARY KEY,
    run_id         TEXT    NOT NULL,
    epoch          INTEGER NOT NULL,
    train_loss     REAL,
    val_loss       REAL,
    links_vec_loss REAL,
    joint_pos_loss REAL,
    lr             REAL,
    epoch_time_sec REAL,
    is_best        INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);
