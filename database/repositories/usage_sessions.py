from database.connection import get_connection, get_cursor
from database.repositories.hands import get_by_name


def start(hand_name: str, min_cutoff: float, beta: float) -> int:
    hand_row = get_by_name(hand_name)
    hand_id  = hand_row["id"] if hand_row else None
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                INSERT INTO usage_sessions (hand_id, min_cutoff, beta)
                VALUES (%s, %s, %s) RETURNING id
            """, (hand_id, min_cutoff, beta))
            session_id = cur.fetchone()["id"]
        conn.commit()
    return session_id


def finish(session_id: int):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE usage_sessions
                SET ended_at = NOW()
                WHERE id = %s
            """, (session_id,))
        conn.commit()


def get_all() -> list:
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                SELECT us.*, h.name as hand_name
                FROM usage_sessions us
                LEFT JOIN hands h ON h.id = us.hand_id
                ORDER BY us.started_at DESC
            """)
            return cur.fetchall()
