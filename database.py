import sqlite3

def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


def insert_user(name):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name) VALUES (?)", (name,))
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return user_id


def get_user_name(user_id):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM users WHERE id=?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else "Unknown"