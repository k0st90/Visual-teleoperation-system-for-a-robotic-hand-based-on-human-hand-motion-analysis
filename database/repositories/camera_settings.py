from database.connection import get_connection, get_cursor
from database.repositories.hands import get_by_name


def get(hand_name: str) -> dict | None:
    with get_connection() as conn:
        cur = get_cursor(conn)
        cur.execute("""
            SELECT cs.cam_distance, cs.cam_yaw, cs.cam_pitch
            FROM camera_settings cs
            JOIN hands h ON h.id = cs.hand_id
            WHERE h.name = ?
        """, (hand_name,))
        return cur.fetchone()


def save(hand_name: str, dist: float, yaw: float, pitch: float):
    row = get_by_name(hand_name)
    if not row:
        return
    hand_id = row["id"]
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO camera_settings (hand_id, cam_distance, cam_yaw, cam_pitch, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT (hand_id) DO UPDATE
                SET cam_distance = EXCLUDED.cam_distance,
                    cam_yaw      = EXCLUDED.cam_yaw,
                    cam_pitch    = EXCLUDED.cam_pitch,
                    updated_at   = datetime('now')
        """, (hand_id, dist, yaw, pitch))
        conn.commit()
