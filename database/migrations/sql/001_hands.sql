CREATE TABLE IF NOT EXISTS hands (
    id          SERIAL      PRIMARY KEY,
    name        TEXT        NOT NULL UNIQUE,
    yml_path    TEXT        NOT NULL,
    assets_path TEXT        NOT NULL,
    added_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
