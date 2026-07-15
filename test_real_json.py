import sys
import traceback
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Testing with REAL JSON files from upload")
print("=" * 70)

json_files = []
upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
if os.path.exists(upload_dir):
    for f in os.listdir(upload_dir):
        if f.endswith('.json'):
            json_files.append(os.path.join(upload_dir, f))
            print(f"Found JSON: {f}")

if not json_files:
    print("No JSON files found in uploads directory!")
    print("Please copy your JSON files to the uploads directory and try again.")
    sys.exit(1)

test_file = json_files[0]
print(f"\nTesting with: {os.path.basename(test_file)}")

# Step 1: Read and parse JSON
try:
    print("\n1. Reading JSON file...")
    with open(test_file, 'r', encoding='utf-8') as f:
        raw_data = f.read()
        print(f"   File size: {len(raw_data)} bytes")
    
    print("   Parsing JSON...")
    data = json.loads(raw_data)
    print(f"   JSON parsed successfully, keys: {list(data.keys())}")
    
    entries_data = data.get('entries', data.get('data', []))
    if isinstance(entries_data, dict):
        entries_data = list(entries_data.values())
    print(f"   Found {len(entries_data)} entries")
    
    if entries_data:
        print(f"   First entry keys: {list(entries_data[0].keys())}")
        
except Exception as e:
    print(f"   ❌ JSON parsing failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: Create KnowledgeEntry objects
try:
    print("\n2. Creating KnowledgeEntry objects...")
    from knowledge_schema import KnowledgeEntry
    
    entries = []
    for item in entries_data:
        entry = KnowledgeEntry(**item)
        if not entry.metadata:
            entry.metadata = {}
        entry.metadata["company"] = "NA"
        entry.metadata["topic"] = "Test"
        entries.append(entry)
    
    print(f"   ✅ Created {len(entries)} KnowledgeEntry objects")
    
except Exception as e:
    print(f"   ❌ KnowledgeEntry creation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 3: Test ingest_entries_to_db
try:
    print("\n3. Calling ingest_entries_to_db...")
    import time
    t_start = time.time()
    
    from app_v2 import ingest_entries_to_db, get_embedding_model
    
    print("   Loading embedding model...")
    embeddings = get_embedding_model()
    print(f"   Embedding model loaded in {time.time()-t_start:.2f}s")
    
    print("   Calling ingest_entries_to_db...")
    _, count, incremental_result = ingest_entries_to_db(entries, embeddings, os.path.basename(test_file))
    
    print(f"   ✅ ingest_entries_to_db succeeded!")
    print(f"   Count: {count}")
    print(f"   Incremental result: {incremental_result}")
    print(f"   Total time: {time.time()-t_start:.2f}s")
    
except Exception as e:
    print(f"   ❌ ingest_entries_to_db failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test completed")
print("=" * 70)
