import sys
import os
import json
import uuid
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Incremental Update v2 Test - Entry-level tracking")
print("=" * 70)

# Create test JSON data with 3 entries
test_data_v1 = {
    "entries": [
        {
            "id": "entry-1",
            "type": "general",
            "title": "Entry 1",
            "content": "This is entry 1 content.",
            "source_file": "knowledge_test-v2.json",
            "source_page": 1,
            "keywords": ["test"],
            "created_at": "2026-07-13T10:00:00",
            "metadata": {"company": "NA", "topic": "Other"}
        },
        {
            "id": "entry-2",
            "type": "general",
            "title": "Entry 2",
            "content": "This is entry 2 content.",
            "source_file": "knowledge_test-v2.json",
            "source_page": 2,
            "keywords": ["test"],
            "created_at": "2026-07-13T10:00:00",
            "metadata": {"company": "NA", "topic": "Other"}
        },
        {
            "id": "entry-3",
            "type": "general",
            "title": "Entry 3",
            "content": "This is entry 3 content.",
            "source_file": "knowledge_test-v2.json",
            "source_page": 3,
            "keywords": ["test"],
            "created_at": "2026-07-13T10:00:00",
            "metadata": {"company": "NA", "topic": "Other"}
        }
    ],
    "source_file": "knowledge_test-v2.json"
}

# Test data v2 with 2 entries (entry-2 deleted)
test_data_v2 = {
    "entries": [
        {
            "id": "entry-1",
            "type": "general",
            "title": "Entry 1",
            "content": "This is entry 1 content.",
            "source_file": "knowledge_test-v2.json",
            "source_page": 1,
            "keywords": ["test"],
            "created_at": "2026-07-13T10:00:00",
            "metadata": {"company": "NA", "topic": "Other"}
        },
        {
            "id": "entry-3",
            "type": "general",
            "title": "Entry 3",
            "content": "This is entry 3 content.",
            "source_file": "knowledge_test-v2.json",
            "source_page": 3,
            "keywords": ["test"],
            "created_at": "2026-07-13T10:00:00",
            "metadata": {"company": "NA", "topic": "Other"}
        }
    ],
    "source_file": "knowledge_test-v2.json"
}

# Step 1: Count documents before
from db_reader import DBReader
from app_v2 import DB_PATH

reader = DBReader(DB_PATH)
before_count = reader.count_documents()
print(f"\nStep 1: Documents before ingest: {before_count}")

# Step 2: First ingest (v1 - 3 entries)
print("\nStep 2: First ingest (v1 - 3 entries)...")
from app_v2 import ingest_entries_to_db, get_embedding_model
from knowledge_schema import KnowledgeEntry

embeddings = get_embedding_model()

entries_v1 = []
for item in test_data_v1["entries"]:
    entry = KnowledgeEntry(**item)
    entries_v1.append(entry)

_, count1, result1 = ingest_entries_to_db(entries_v1, embeddings, test_data_v1["source_file"])
print(f"   First ingest result: count={count1}")
print(f"   Incremental: {result1}")

after_count1 = reader.count_documents()
print(f"   Documents after first ingest: {after_count1}")

# Step 3: Second ingest (v2 - 2 entries, entry-2 deleted)
print("\nStep 3: Second ingest (v2 - 2 entries, entry-2 deleted)...")
entries_v2 = []
for item in test_data_v2["entries"]:
    entry = KnowledgeEntry(**item)
    entries_v2.append(entry)

_, count2, result2 = ingest_entries_to_db(entries_v2, embeddings, test_data_v2["source_file"])
print(f"   Second ingest result: count={count2}")
print(f"   Incremental: {result2}")

after_count2 = reader.count_documents()
print(f"   Documents after second ingest: {after_count2}")

# Step 4: Verify
print("\nStep 4: Verification:")
if result2.get("deleted", 0) == 1:
    print("   ✅ CORRECT: 1 entry detected as deleted")
else:
    print(f"   ❌ INCORRECT: Expected 1 deleted, got {result2.get('deleted', 0)}")

if result2.get("unchanged", 0) == 2:
    print("   ✅ CORRECT: 2 entries detected as unchanged")
else:
    print(f"   ❌ INCORRECT: Expected 2 unchanged, got {result2.get('unchanged', 0)}")

if result2.get("version_number", 0) == 2:
    print("   ✅ CORRECT: Version number is 2")
else:
    print(f"   ❌ INCORRECT: Expected version 2, got {result2.get('version_number', 0)}")

expected_count = after_count1 - 1
if after_count2 == expected_count:
    print(f"   ✅ CORRECT: Document count decreased by 1 (from {after_count1} to {after_count2})")
else:
    print(f"   ❌ INCORRECT: Expected {expected_count} documents, got {after_count2}")

# Step 5: Third ingest (same as v2 - should show all unchanged)
print("\nStep 5: Third ingest (same as v2 - no changes)...")
_, count3, result3 = ingest_entries_to_db(entries_v2, embeddings, test_data_v2["source_file"])
print(f"   Third ingest result: count={count3}")
print(f"   Incremental: {result3}")

after_count3 = reader.count_documents()
print(f"   Documents after third ingest: {after_count3}")

print("\nStep 6: Verification (third ingest):")
if result3.get("new", 0) == 0 and result3.get("modified", 0) == 0 and result3.get("deleted", 0) == 0:
    print("   ✅ CORRECT: No changes detected")
else:
    print(f"   ❌ INCORRECT: Expected no changes")

if after_count3 == after_count2:
    print("   ✅ CORRECT: Document count unchanged")
else:
    print(f"   ❌ INCORRECT: Document count changed")

if result3.get("version_number", 0) == 3:
    print("   ✅ CORRECT: Version number is 3")
else:
    print(f"   ❌ INCORRECT: Expected version 3, got {result3.get('version_number', 0)}")

print("\n" + "=" * 70)
print("Incremental update v2 test completed!")
print("=" * 70)
