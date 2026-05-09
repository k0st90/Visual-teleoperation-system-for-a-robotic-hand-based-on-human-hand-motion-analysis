import psycopg2
from psycopg2.extras import RealDictCursor

DSN = "host=localhost port=5432 dbname=hand_teleop user=teleop password=teleop123"


def get_connection():
    return psycopg2.connect(DSN)


def get_cursor(conn):
    return conn.cursor(cursor_factory=RealDictCursor)
