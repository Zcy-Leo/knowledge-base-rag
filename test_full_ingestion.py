import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Full JSON Ingestion Test")
print("=" * 70)

# Step 1: Load JSON
print("\nStep 1: Loading JSON file...")
json_path = "knowledge_json_output/knowledge_2023-half-year-analyses-by-segment_20260712_122135.json"

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

entries_data = data.get("entries", [])
source = data.get("source_file", "")
print(f"  Loaded {len(entries_data)} entries from {source}")

# Step 2: Create KnowledgeBase
print("\nStep 2: Creating KnowledgeBase...")
from knowledge_schema import KnowledgeBase, KnowledgeEntry

kb = KnowledgeBase(source_file=source)
for item in entries_data:
    entry = KnowledgeEntry(**item)
    if not entry.metadata:
        entry.metadata = {}
    entry.metadata["company"] = "NA"
    entry.metadata["topic"] = "Other"
    kb.add(entry)
print(f"  Created KnowledgeBase with {len(kb.entries)} entries")

# Step 3: Load embedding model
print("\nStep 3: Loading embedding model...")
import time
t_start = time.time()

try:
    from app_v2 import load_embeddings, get_db
    embeddings = load_embeddings()
    print(f"  Embedding model loaded in {time.time()-t_start:.2f}s")
except Exception as e:
    print(f"  ❌ Failed to load embeddings: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test DBWriter
print("\nStep 4: Testing DBWriter...")
try:
    from db_writer import DBWriter
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_local_database", "chroma.sqlite3")
    
    writer = DBWriter(DB_PATH)
    texts = [e.to_chroma_text() for e in kb.entries]
    metas = [e.to_chroma_metadata() for e in kb.entries]
    
    count = writer.add_documents(texts, metas, embeddings)
    print(f"  ✅ Successfully ingested {count} documents")
except Exception as e:
    print(f"  ❌ DBWriter failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Verify database
print("\nStep 5: Verifying database...")
try:
    from db_reader import DBReader
    reader = DBReader(DB_PATH)
    total = reader.count_documents()
    print(f"  ✅ Total documents in DB: {total}")
except Exception as e:
    print(f"  ❌ DBReader failed: {e}")

print("\n" + "=" * 70)
print("All tests passed!")
