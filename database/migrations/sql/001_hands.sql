CREATE TABLE IF NOT EXISTS hands (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL UNIQUE,
    yml_path    TEXT    NOT NULL,
    assets_path TEXT    NOT NULL,
    added_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);
