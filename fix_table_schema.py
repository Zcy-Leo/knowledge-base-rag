import sqlite3

conn = sqlite3.connect('my_local_database/chroma.sqlite3')
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS doc_hash_store")

cursor.execute('''
    CREATE TABLE doc_hash_store (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT NOT NULL,
        chunk_hashes TEXT NOT NULL,
        last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        version_number INTEGER NOT NULL DEFAULT 1,
        UNIQUE(doc_id, version_number)
    )
''')

cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_doc_hash_store_doc_id ON doc_hash_store(doc_id)
''')

conn.commit()
conn.close()
print('Fixed doc_hash_store table schema')