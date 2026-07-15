import sqlite3

conn = sqlite3.connect('my_local_database/livevectorlake.db')
c = conn.cursor()

c.execute('SELECT COUNT(*) FROM chunk_versions WHERE status="active"')
print(f"Active rows: {c.fetchone()[0]}")

c.execute('SELECT MIN(valid_from), MAX(valid_from) FROM chunk_versions')
print(f"Time range: {c.fetchone()}")

c.execute('SELECT valid_from, doc_id, content FROM chunk_versions LIMIT 3')
for row in c.fetchall():
    print(f"  {row[0]} | {row[1][:20]} | {row[2][:50]}")

test_timestamp = "2026-07-12T00:00:00"
c.execute('''
    SELECT COUNT(*) FROM chunk_versions 
    WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
    AND status = "active"
''', (test_timestamp, test_timestamp))
print(f"\nRows with timestamp {test_timestamp}: {c.fetchone()[0]}")

conn.close()