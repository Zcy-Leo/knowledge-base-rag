import sqlite3

conn = sqlite3.connect("my_local_database/chroma.sqlite3")
cursor = conn.cursor()

tables_to_check = ["embeddings", "embedding_metadata", "embedding_fulltext_search_content"]

for table in tables_to_check:
    print(f"\n--- Table: {table} ---")
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    cursor.execute(f"SELECT * FROM {table} LIMIT 2")
    rows = cursor.fetchall()
    for i, row in enumerate(rows):
        print(f"  Sample row {i}: {row}")
        print(f"    Types: {[type(x).__name__ for x in row]}")

conn.close()
