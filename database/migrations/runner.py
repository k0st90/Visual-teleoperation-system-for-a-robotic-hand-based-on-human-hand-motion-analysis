"""
Migration runner.
Tracks applied migrations in schema_migrations table.
Applies new .sql files from migrations/sql/ in order.
"""

import os
from database.connection import get_connection

SQL_DIR = os.path.join(os.path.dirname(__file__), "sql")

_INIT = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    id         SERIAL      PRIMARY KEY,
    filename   TEXT        NOT NULL UNIQUE,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def run():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_INIT)
            cur.execute("SELECT filename FROM schema_migrations")
            applied = {row[0] for row in cur.fetchall()}

        files = sorted(f for f in os.listdir(SQL_DIR) if f.endswith(".sql"))
        pending = [f for f in files if f not in applied]

        for filename in pending:
            path = os.path.join(SQL_DIR, filename)
            sql  = open(path, encoding="utf-8").read()
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (filename,)
                )
            conn.commit()
            print(f"  [migration] applied: {filename}")

        if not pending:
            print("  [migration] all up to date")
