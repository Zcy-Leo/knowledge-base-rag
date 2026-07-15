import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Testing JSON Upload Validation Logic")
print("=" * 70)

# Find JSON files in common locations
search_paths = [
    os.path.dirname(os.path.abspath(__file__)),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads"),
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~/Downloads"),
]

json_files = []
for path in search_paths:
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith('.json') and not f.startswith('.'):
                full_path = os.path.join(path, f)
                if os.path.isfile(full_path):
                    json_files.append((f, full_path))
                    print(f"Found: {full_path}")

if not json_files:
    print("No JSON files found! Creating test JSON...")
    test_data = {
        "entries": [
            {
                "id": "test_entry_1",
                "type": "general",
                "title": "Test Document",
                "content": "This is a test document with content.",
                "source_file": "test.json",
                "source_page": 1,
                "keywords": [],
                "created_at": "2026-07-13T10:00:00",
                "metadata": {"company": "NA", "topic": "Financial"}
            }
        ],
        "source_file": "test.json"
    }
    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_upload.json")
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
    json_files = [("test_upload.json", test_path)]
    print(f"Created test JSON: {test_path}")

# Import detection functions
print("\nImporting detection functions...")
try:
    from app_v2 import detect_company_from_filename, detect_topic_from_filename, get_all_companies
    print("✅ Functions imported successfully")
except Exception as e:
    print(f"❌ Failed to import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test each JSON file
for file_name, file_path in json_files[:3]:
    print(f"\n{'='*50}")
    print(f"Testing file: {file_name}")
    print(f"{'='*50}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        entries_data = data.get("entries", [])
        source = data.get("source_file", file_name)
        
        print(f"Entries count: {len(entries_data)}")
        print(f"Source: {source}")
        
        if entries_data:
            first_entry = entries_data[0]
            print(f"First entry keys: {list(first_entry.keys())}")
            first_meta = first_entry.get("metadata", {})
            print(f"First entry metadata: {first_meta}")
            
            detected_company = first_meta.get("company", "")
            detected_topic = first_meta.get("topic", "")
            print(f"Detected from metadata - Company: '{detected_company}', Topic: '{detected_topic}'")
        
        # Test filename detection
        filename_company = detect_company_from_filename(source)
        filename_topic = detect_topic_from_filename(source)
        print(f"Detected from filename - Company: '{filename_company}', Topic: '{filename_topic}'")
        
        # Check if valid
        all_companies_list = get_all_companies()
        all_topics_list = ["TechDocs", "Books", "Academic", "Government", "Financial", "Office", "Email", "Images", "Data", "Demo", "Other"]
        
        final_company = detected_company if detected_company else filename_company
        final_topic = detected_topic if detected_topic else filename_topic
        
        is_valid = final_company != "" and final_topic != ""
        print(f"\nFinal Company: '{final_company}'")
        print(f"Final Topic: '{final_topic}'")
        print(f"Is valid: {is_valid}")
        
        if not is_valid:
            print("\n⚠️ NOT VALID - Suggested fixes:")
            if not final_company:
                print(f"   - Add 'company' to JSON metadata, or select from dropdown: {all_companies_list}")
            if not final_topic:
                print(f"   - Add 'topic' to JSON metadata, or select from dropdown: {all_topics_list}")
        
    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("Validation test completed")
print("=" * 70)
