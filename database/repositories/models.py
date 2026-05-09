from database.connection import get_connection, get_cursor
from database.repositories.hands import get_by_name as get_hand_by_name


def save(hand_name: str, run_id: str, checkpoint_path: str) -> int:
    row = get_hand_by_name(hand_name)
    if row is None:
        raise ValueError(f"Hand '{hand_name}' not found in DB")
    hand_id = row["id"]
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                INSERT INTO models (hand_id, run_id, checkpoint_path)
                VALUES (%s, %s, %s) RETURNING id
            """, (hand_id, run_id, checkpoint_path))
            model_id = cur.fetchone()["id"]
        conn.commit()
    return model_id


def get_latest(hand_name: str) -> dict | None:
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                SELECT m.id, m.run_id, m.checkpoint_path, m.created_at
                FROM models m
                JOIN hands h ON h.id = m.hand_id
                WHERE h.name = %s
                ORDER BY m.created_at DESC
                LIMIT 1
            """, (hand_name,))
            return cur.fetchone()


def get_all_for_hand(hand_name: str) -> list:
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                SELECT m.id, m.run_id, m.checkpoint_path, m.created_at
                FROM models m
                JOIN hands h ON h.id = m.hand_id
                WHERE h.name = %s
                ORDER BY m.created_at DESC
            """, (hand_name,))
            return cur.fetchall()
