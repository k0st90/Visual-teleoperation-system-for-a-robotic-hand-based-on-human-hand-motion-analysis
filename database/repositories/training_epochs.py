from database.connection import get_connection, get_cursor


def save(run_id: str, epoch: int, train_loss: float, val_loss: float,
         links_vec_loss: float = None, joint_pos_loss: float = None,
         lr: float = None, epoch_time_sec: float = None,
         is_best: bool = False):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO training_epochs
                    (run_id, epoch, train_loss, val_loss, links_vec_loss,
                     joint_pos_loss, lr, epoch_time_sec, is_best)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (run_id, epoch, train_loss, val_loss, links_vec_loss,
                  joint_pos_loss, lr, epoch_time_sec, is_best))
        conn.commit()


def get_for_run(run_id: str) -> list:
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                SELECT * FROM training_epochs
                WHERE run_id = %s
                ORDER BY epoch
            """, (run_id,))
            return cur.fetchall()
