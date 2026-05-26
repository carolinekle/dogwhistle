import sqlite3

DB_PATH = "data/whistle_results.db"

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_conn()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY,
    content TEXT,
    dog_whistle TEXT,
    ingroup TEXT,
    model TEXT,  -- 'openai' or 'llama_guard'
    flagged BOOLEAN,
    categories TEXT,
    category_scores TEXT,
    source TEXT
    )
    """)
    conn.commit()
    conn.close()
    print(f"DB initialized at {DB_PATH}")

if __name__ == "__main__":
    init_db()