import sqlite3

conn = sqlite3.connect('my_local_database/livevectorlake.db')
c = conn.cursor()

test_timestamp = "2026-07-12T00:00:00"
c.execute('''
    SELECT cv.chunk_id, cv.doc_id, cv.position, cv.content, cv.embedding,
           cv.valid_from, cv.valid_to, cv.version_number
    FROM chunk_versions cv
    WHERE cv.valid_from <= ? AND (cv.valid_to IS NULL OR cv.valid_to > ?)
    AND cv.status = 'active'
    ORDER BY cv.doc_id, cv.position
    LIMIT 5
''', (test_timestamp, test_timestamp))

rows = c.fetchall()
print(f"Total rows: {len(rows)}")
print(f"First row embedding is None: {rows[0][4] is None}")

has_embedding = sum(1 for row in rows if row[4] is not None)
print(f"Rows with embedding: {has_embedding}/{len(rows)}")

if rows:
    doc_ids = set()
    for row in rows:
        doc_ids.add(row[1])
    print(f"\nUnique doc_ids: {len(doc_ids)}")
    for doc_id in list(doc_ids)[:3]:
        print(f"  - {doc_id}")

conn.close()