import sqlite3
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("Debugging Temporal Query")
print("=" * 60)

conn = sqlite3.connect('my_local_database/chroma.sqlite3')
cursor = conn.cursor()

print("\n--- Checking chunk_versions table ---")
cursor.execute('SELECT COUNT(*) FROM chunk_versions')
print(f"Total rows: {cursor.fetchone()[0]}")

cursor.execute('SELECT COUNT(*) FROM chunk_versions WHERE status = "active"')
print(f"Active rows: {cursor.fetchone()[0]}")

cursor.execute('SELECT MIN(valid_from), MAX(valid_from) FROM chunk_versions')
min_time, max_time = cursor.fetchone()
print(f"Time range: {min_time} to {max_time}")

cursor.execute('SELECT valid_from, valid_to, status FROM chunk_versions LIMIT 5')
print("\nSample rows:")
for row in cursor.fetchall():
    print(f"  valid_from={row[0]}, valid_to={row[1]}, status={row[2]}")

print("\n--- Testing SQL query directly ---")
test_timestamp = "2026-07-12T00:00:00"
print(f"Testing with timestamp: {test_timestamp}")

cursor.execute('''
    SELECT cv.chunk_id, cv.doc_id, cv.position, cv.content,
           cv.valid_from, cv.valid_to, cv.version_number
    FROM chunk_versions cv
    WHERE cv.valid_from <= ? AND (cv.valid_to IS NULL OR cv.valid_to > ?)
    AND cv.status = 'active'
    ORDER BY cv.doc_id, cv.position
    LIMIT 3
''', (test_timestamp, test_timestamp))

rows = cursor.fetchall()
print(f"Rows returned: {len(rows)}")
for row in rows:
    print(f"  doc_id={row[1][:20]}, valid_from={row[4]}")

conn.close()

print("\n--- Testing FAISSSearch temporal_query ---")
from faiss_search import FAISSSearch

searcher = FAISSSearch()
searcher.initialize()

test_query = "financial"
print(f"\nTest query: '{test_query}'")

results = searcher.temporal_query(test_query, timestamp=None, k=3)
print(f"Results with None timestamp: {len(results)}")

results = searcher.temporal_query(test_query, timestamp="2026-07-12T00:00:00", k=3)
print(f"Results with 2026-07-12: {len(results)}")

results = searcher.temporal_query(test_query, timestamp="2026-07-11T20:38:10", k=3)
print(f"Results with exact time: {len(results)}")

print("\n" + "=" * 60)
print("Debug completed!")
print("=" * 60)