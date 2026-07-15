import sys
import os
import json
import uuid
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Incremental Update Test - Re-upload Same File")
print("=" * 70)

# Create test JSON data
test_data = {
    "entries": [
        {
            "id": str(uuid.uuid4()),
            "type": "general",
            "title": "Test Document",
            "content": "This is a test document.\n\nIt has two paragraphs.\n\nAnd a third one.",
            "source_file": "knowledge_test-doc.json",
            "source_page": 1,
            "keywords": ["test"],
            "created_at": "2026-07-13T10:00:00",
            "metadata": {"company": "NA", "topic": "Other"}
        }
    ],
    "source_file": "knowledge_test-doc.json"
}

# Step 1: Create test file
print("\nStep 1: Creating test file...")
test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_incremental.json")
with open(test_file_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2)

# Step 2: Count documents before
from db_reader import DBReader
from app_v2 import DB_PATH

reader = DBReader(DB_PATH)
before_count = reader.count_documents()
print(f"\nStep 2: Documents before ingest: {before_count}")

# Step 3: First ingest
print("\nStep 3: First ingest (should show New Chunks)...")
from app_v2 import ingest_entries_to_db, get_embedding_model
from knowledge_schema import KnowledgeEntry

embeddings = get_embedding_model()

with open(test_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

entries = []
for item in data["entries"]:
    entry = KnowledgeEntry(**item)
    if not entry.metadata:
        entry.metadata = {}
    entries.append(entry)

_, count1, result1 = ingest_entries_to_db(entries, embeddings, data["source_file"])
print(f"   First ingest result: count={count1}")
print(f"   Incremental: {result1}")

# Step 4: Count documents after first ingest
after_count1 = reader.count_documents()
print(f"\nStep 4: Documents after first ingest: {after_count1}")
print(f"   Added: {after_count1 - before_count} documents")

# Step 5: Second ingest (same file)
print("\nStep 5: Second ingest (same file - should show unchanged)...")
_, count2, result2 = ingest_entries_to_db(entries, embeddings, data["source_file"])
print(f"   Second ingest result: count={count2}")
print(f"   Incremental: {result2}")

# Step 6: Count documents after second ingest
after_count2 = reader.count_documents()
print(f"\nStep 6: Documents after second ingest: {after_count2}")
print(f"   Added in second ingest: {after_count2 - after_count1} documents")

# Step 7: Verify
print("\nStep 7: Verification:")
if count2 == 0:
    print("   ✅ CORRECT: No documents were added in second ingest (all unchanged)")
else:
    print(f"   ❌ INCORRECT: {count2} documents were added in second ingest")

if after_count2 == after_count1:
    print("   ✅ CORRECT: Total document count unchanged after re-upload")
else:
    print(f"   ❌ INCORRECT: Document count changed from {after_count1} to {after_count2}")

# Step 8: Cleanup
print("\nStep 8: Cleaning up...")
if os.path.exists(test_file_path):
    os.remove(test_file_path)

print("\n" + "=" * 70)
print("Incremental update test completed!")
print("=" * 70)
