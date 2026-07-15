import json
import os
import sys
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_schema import KnowledgeBase, KnowledgeEntry

# Test: Parse marker raw format JSON
print("=" * 60)
print("Test: Marker Raw Format JSON Parsing")
print("=" * 60)

json_path = "knowledge_json_output/knowledge_2022-financial-statements-p11.pdf_20260712_122134.json"
print(f"Loading: {json_path}")

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

entries_data = data.get("entries", [])
source = data.get("source_file", "")

print(f"Original entries count: {len(entries_data)}")

# Check if this is marker raw format
is_marker_raw = False
if len(entries_data) == 1:
    first_entry = entries_data[0]
    content = first_entry.get("content", "")
    if content.startswith("[") and content.endswith("]"):
        try:
            json.loads(content)
            is_marker_raw = True
        except:
            pass

print(f"Is marker raw format: {is_marker_raw}")

if is_marker_raw:
    marker_elements = json.loads(content)
    print(f"Marker elements count: {len(marker_elements)}")
    
    # Parse marker elements
    parsed_entries = []
    for element in marker_elements:
        elem_type = element.get("type", "")
        elem_text = element.get("text", "")
        
        if elem_text and isinstance(elem_text, str):
            entry_dict = {
                "id": str(uuid.uuid4()),
                "type": "general",
                "title": f"{elem_type}: {elem_text[:50]}..." if len(elem_text) > 50 else f"{elem_type}: {elem_text}",
                "content": elem_text,
                "source_file": source,
                "source_page": 0,
                "keywords": [],
                "created_at": datetime.now().isoformat(),
                "metadata": {"element_type": elem_type}
            }
            parsed_entries.append(entry_dict)
    
    print(f"\nParsed entries: {len(parsed_entries)}")
    
    # Test KnowledgeEntry construction
    kb = KnowledgeBase(source_file=source)
    for entry_dict in parsed_entries[:3]:
        entry = KnowledgeEntry(**entry_dict)
        kb.add(entry)
        print(f"  - {entry.title[:60]}")
    
    print(f"\n✅ Successfully parsed {len(kb.entries)} entries!")
    
    # Test Chroma conversion
    texts = [e.to_chroma_text() for e in kb.entries]
    metas = [e.to_chroma_metadata() for e in kb.entries]
    print(f"\n✅ Chroma conversion: {len(texts)} texts, {len(metas)} metadatas")

else:
    print("❌ Not marker raw format")

print("\n" + "=" * 60)
print("Test completed!")
