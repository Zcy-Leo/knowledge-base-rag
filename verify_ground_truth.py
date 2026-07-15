import os
import json
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "my_local_database", "chroma.sqlite3")
GT_PATH = os.path.join(BASE_DIR, "experiment_results", "ground_truth.json")

def get_doc_content(doc_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        int_id = int(doc_id)
        cursor.execute("SELECT c0 FROM embedding_fulltext_search_content WHERE id = ?", (int_id,))
        row = cursor.fetchone()
        content = row[0] if row else ""
        
        cursor.execute("SELECT key, string_value FROM embedding_metadata WHERE id = ?", (int_id,))
        rows = cursor.fetchall()
        metadata = {r[0]: r[1] for r in rows}
        
        return content, metadata
    finally:
        conn.close()

with open(GT_PATH, 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)

for query, gt in ground_truth.items():
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"Type: {gt['type']}, Difficulty: {gt['difficulty']}")
    print(f"Relevant IDs: {gt['relevant_ids']}")
    print(f"\n--- Relevant Documents ---")
    for doc_id in gt['relevant_ids']:
        content, metadata = get_doc_content(doc_id)
        title = metadata.get('title', '')[:80]
        source_file = metadata.get('source_file', '')
        print(f"\nID: {doc_id}")
        print(f"Source: {source_file}")
        print(f"Title: {title}")
        print(f"Content (first 200 chars): {content[:200]}")