import os
from database.connection import get_connection, get_cursor


def get_or_create(name: str, yml_path: str, assets_path: str) -> int:
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                INSERT INTO hands (name, yml_path, assets_path)
                VALUES (%s, %s, %s)
                ON CONFLICT (name) DO UPDATE
                    SET yml_path    = EXCLUDED.yml_path,
                        assets_path = EXCLUDED.assets_path
                RETURNING id
            """, (name, yml_path, assets_path))
            hand_id = cur.fetchone()["id"]
        conn.commit()
    return hand_id


def get_all() -> list:
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("SELECT id, name, yml_path, assets_path, added_at FROM hands ORDER BY name")
            return cur.fetchall()


def get_by_name(name: str) -> dict | None:
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                "SELECT id, name, yml_path, assets_path FROM hands WHERE name = %s",
                (name,)
            )
            return cur.fetchone()


def update_paths(name: str, yml_path: str, assets_path: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE hands SET yml_path = %s, assets_path = %s WHERE name = %s",
                (yml_path, assets_path, name)
            )
        conn.commit()


def delete(name: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM hands WHERE name = %s", (name,))
        conn.commit()


def validate_paths(name: str) -> tuple[bool, str]:
    row = get_by_name(name)
    if not row:
        return False, "Руку не знайдено в БД"
    if not os.path.isfile(row["yml_path"]):
        return False, f"Конфіг не знайдено: {row['yml_path']}"
    if not os.path.isdir(row["assets_path"]):
        return False, f"Assets не знайдено: {row['assets_path']}"
    return True, ""
