import os
from database.connection import get_connection, get_cursor
from database.repositories.hands import get_or_create as get_or_create_hand


def add_original(filename: str, full_path: str, duration_sec: float = None) -> int:
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                INSERT INTO original_videos (filename, full_path, duration_sec)
                VALUES (%s, %s, %s)
                ON CONFLICT (filename) DO UPDATE
                    SET full_path = EXCLUDED.full_path
                RETURNING id
            """, (filename, full_path, duration_sec))
            vid_id = cur.fetchone()["id"]
        conn.commit()
    return vid_id


def get_original_by_filename(filename: str) -> dict | None:
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                "SELECT * FROM original_videos WHERE filename = %s",
                (filename,)
            )
            return cur.fetchone()


def delete_original(filename: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM original_videos WHERE filename = %s", (filename,))
        conn.commit()


def add_retargeted(filename: str, full_path: str,
                   original_filename: str = None, hand_name: str = None,
                   min_cutoff: float = None, beta: float = None,
                   cam_distance: float = None, cam_yaw: float = None,
                   cam_pitch: float = None, model_id: int = None) -> int:
    original_id = None
    if original_filename:
        row = get_original_by_filename(original_filename)
        if row:
            original_id = row["id"]

    hand_id = None
    if hand_name:
        try:
            hand_id = get_or_create_hand(hand_name, yml_path="", assets_path="")
        except Exception:
            pass

    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                INSERT INTO retargeted_videos
                    (filename, full_path, original_id, hand_id,
                     min_cutoff, beta, cam_distance, cam_yaw, cam_pitch, model_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (filename) DO UPDATE
                    SET full_path = EXCLUDED.full_path
                RETURNING id
            """, (filename, full_path, original_id, hand_id,
                  min_cutoff, beta, cam_distance, cam_yaw, cam_pitch, model_id))
            vid_id = cur.fetchone()["id"]
        conn.commit()
    return vid_id


def delete_retargeted(filename: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM retargeted_videos WHERE filename = %s", (filename,))
        conn.commit()


def get_all_originals() -> list:
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("SELECT * FROM original_videos ORDER BY date_added DESC")
            return cur.fetchall()


def get_all_retargeted() -> list:
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                SELECT rv.*, h.name as hand_name, ov.filename as source_filename
                FROM retargeted_videos rv
                LEFT JOIN hands h ON h.id = rv.hand_id
                LEFT JOIN original_videos ov ON ov.id = rv.original_id
                ORDER BY rv.date_created DESC
            """)
            return cur.fetchall()
