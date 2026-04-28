import pandas as pd
import sqlite3
import os

def load_to_warehouse():
    # Auto-detect folder where THIS script lives
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "papers_clean.csv")
    db_path  = os.path.join(base_dir, "papers_warehouse.db")

    print(f"Looking for file at: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            link            TEXT,
            title           TEXT,
            authors         TEXT,
            description     TEXT,
            category        TEXT,
            primary_subject TEXT,
            subjects        TEXT,
            date            TEXT,
            link_of_paper   TEXT,
            link_of_pdf     TEXT,
            clean_text      TEXT
        )
    """)
    conn.commit()

    df.to_sql("papers", conn, if_exists="replace", index=False)

    total = cursor.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    print(f"Total papers in warehouse: {total}")

    conn.close()
    print(f"✅ Warehouse ready → {db_path}")

if __name__ == "__main__":
    load_to_warehouse()