import sqlite3

conn = sqlite3.connect('my_local_database/chroma.sqlite3')
cursor = conn.cursor()

# Check embeddings count
cursor.execute("SELECT COUNT(*) FROM embeddings")
result = cursor.fetchone()
print(f"Embeddings count: {result[0]}")

# Check embedding_metadata count
cursor.execute("SELECT COUNT(*) FROM embedding_metadata")
result = cursor.fetchone()
print(f"Embedding metadata count: {result[0]}")

# Check embedding_fulltext_search_content count
cursor.execute("SELECT COUNT(*) FROM embedding_fulltext_search_content")
result = cursor.fetchone()
print(f"Fulltext content count: {result[0]}")

conn.close()