#!/usr/bin/env python3
"""
init_db.py
==========
Standalone script to initialise the SQLite database.
Run this once before starting the app:
    python init_db.py
"""
import os
import sqlite3

DATABASE = os.path.join(os.path.dirname(__file__), "database", "fakenews.db")

def init():
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
    with sqlite3.connect(DATABASE) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS queries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_address  TEXT    NOT NULL,
                input_type  TEXT    NOT NULL,
                input_text  TEXT    NOT NULL,
                prediction  TEXT    NOT NULL,
                confidence  REAL    NOT NULL,
                explanation TEXT,
                model_used  TEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS admin_sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                token      TEXT UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
    print(f"✅ Database initialised at: {DATABASE}")

if __name__ == "__main__":
    init()
