import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time

print("[TEST] Starting test...", flush=True)

# Test 1: Load embedding model
print("[TEST] Test 1: Loading embedding model...", flush=True)
start = time.time()
from app_v2 import load_embeddings
embeddings = load_embeddings()
print(f"[TEST] Embedding model loaded in {time.time()-start:.2f}s", flush=True)

# Test 2: Load sample JSON
print("[TEST] Test 2: Loading sample JSON...", flush=True)
json_path = "knowledge_json_output/knowledge_2022-financial-statements-p11.pdf_20260712_122134.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
entries_data = data.get("entries", [])
print(f"[TEST] Loaded {len(entries_data)} entries", flush=True)

# Test 3: Check marker raw format
print("[TEST] Test 3: Checking marker raw format...", flush=True)
is_marker_raw = False
if len(entries_data) == 1:
    first_entry = entries_data[0]
    content = first_entry.get("content", "")
    if content.startswith("[") and content.endswith("]"):
        try:
            json.loads(content)
            is_marker_raw = True
            print(f"[TEST] Detected marker raw format, content length: {len(content)}", flush=True)
        except:
            pass

# Test 4: Parse marker raw format
if is_marker_raw:
    print("[TEST] Test 4: Parsing marker raw format...", flush=True)
    marker_elements = json.loads(content)
    print(f"[TEST] Parsed {len(marker_elements)} marker elements", flush=True)
    
    entries_data = []
    for element in marker_elements:
        elem_type = element.get("type", "")
        elem_text = element.get("text", "")
        
        if elem_text and isinstance(elem_text, str):
            entries_data.append({
                "id": str(hash(elem_text)),
                "type": "general",
                "title": f"{elem_type}: {elem_text[:50]}..." if len(elem_text) > 50 else f"{elem_type}: {elem_text}",
                "content": elem_text,
                "source_file": data.get("source_file", "test"),
                "source_page": 0,
                "keywords": [],
                "created_at": "2026-07-12T00:00:00",
                "metadata": {"element_type": elem_type}
            })
    print(f"[TEST] Done: {len(entries_data)} entries from marker raw format", flush=True)

# Test 5: Create KnowledgeEntry objects
print("[TEST] Test 5: Creating KnowledgeEntry objects...", flush=True)
from knowledge_schema import KnowledgeBase, KnowledgeEntry
kb = KnowledgeBase(source_file="test")
for item in entries_data:
    entry = KnowledgeEntry(**item)
    if not entry.metadata:
        entry.metadata = {}
    entry.metadata["company"] = "NA"
    entry.metadata["topic"] = "Financial"
    kb.add(entry)
print(f"[TEST] Created {len(kb.entries)} KnowledgeEntry objects", flush=True)

# Test 6: Ingest to database
print("[TEST] Test 6: Ingesting to database...", flush=True)
start = time.time()
from app_v2 import ingest_entries_to_db
_, count, incremental_result = ingest_entries_to_db(kb.entries, embeddings, "test")
print(f"[TEST] Ingested {count} entries in {time.time()-start:.2f}s", flush=True)

print("[TEST] All tests completed successfully!", flush=True)