import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_schema import KnowledgeBase, KnowledgeEntry

# Test 1: Check JSON file structure
print("=" * 60)
print("Test 1: JSON File Structure")
print("=" * 60)

json_dir = "knowledge_json_output"
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

if not json_files:
    print("❌ No JSON files found in knowledge_json_output/")
    sys.exit(1)

# Load the first JSON file
json_path = os.path.join(json_dir, json_files[0])
print(f"Loading: {json_path}")

try:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Schema version: {data.get('schema_version')}")
    print(f"Source file: {data.get('source_file')}")
    print(f"Total entries: {data.get('total_entries')}")
    
    entries = data.get('entries', [])
    if entries:
        first_entry = entries[0]
        print(f"\nFirst entry keys: {list(first_entry.keys())}")
        
        # Check if all required fields exist
        required_fields = ['id', 'type', 'title', 'content', 'source_file']
        missing_fields = [f for f in required_fields if f not in first_entry]
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
        else:
            print("✅ All required fields present")
            
        # Test KnowledgeEntry construction
        print("\nTest 2: KnowledgeEntry Construction")
        print("=" * 60)
        try:
            entry = KnowledgeEntry(**first_entry)
            print("✅ KnowledgeEntry constructed successfully")
            print(f"  type: {entry.type}")
            print(f"  title: {entry.title[:50]}...")
            print(f"  content length: {len(entry.content)}")
        except Exception as e:
            print(f"❌ KnowledgeEntry construction failed: {e}")
            
    else:
        print("❌ No entries in JSON file")
        
except Exception as e:
    print(f"❌ Failed to load JSON: {e}")

# Test 3: Simulate the full ingestion flow
print("\nTest 3: Full Ingestion Flow")
print("=" * 60)

try:
    kb = KnowledgeBase.load_json(json_path)
    print(f"✅ Loaded {len(kb.entries)} entries into KnowledgeBase")
    
    # Check if entries can be converted to Chroma format
    for i, entry in enumerate(kb.entries[:3]):
        text = entry.to_chroma_text()
        meta = entry.to_chroma_metadata()
        print(f"\nEntry {i+1}:")
        print(f"  text length: {len(text)}")
        print(f"  metadata keys: {list(meta.keys())}")
        
except Exception as e:
    print(f"❌ Full flow failed: {e}")

print("\n" + "=" * 60)
print("Test completed!")
