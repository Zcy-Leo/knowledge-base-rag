import sys
import os
import json
import uuid
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Full End-to-End Ingest Flow Test")
print("=" * 70)

# Create test JSON data
test_data = {
    "entries": [
        {
            "id": str(uuid.uuid4()),
            "type": "general",
            "title": "Financial Report 2022",
            "content": "This is a financial report document for the year 2022.\n\nIt contains important financial data and analysis.\n\nRevenue increased by 15% compared to last year.",
            "source_file": "knowledge_2022-financial-report.json",
            "source_page": 1,
            "keywords": ["financial", "report", "2022"],
            "created_at": "2026-07-13T10:00:00",
            "metadata": {"company": "NA", "topic": "Financial"}
        },
        {
            "id": str(uuid.uuid4()),
            "type": "general",
            "title": "Market Analysis Q2",
            "content": "Market analysis for the second quarter shows positive trends.\n\nMarket share increased by 5 percentage points.",
            "source_file": "knowledge_2022-financial-report.json",
            "source_page": 2,
            "keywords": ["market", "analysis", "Q2"],
            "created_at": "2026-07-13T10:00:00",
            "metadata": {"company": "NA", "topic": "Financial"}
        }
    ],
    "source_file": "knowledge_2022-financial-report.json"
}

print("Step 1: Creating test JSON file...")
test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_full_ingest.json")
with open(test_file_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2)
print(f"   Created: {test_file_path}")

# Step 2: Parse JSON like Streamlit upload
print("\nStep 2: Parsing JSON...")
with open(test_file_path, 'r', encoding='utf-8') as f:
    file_data = f.read().encode('utf-8')

data = json.loads(file_data.decode("utf-8"))
entries_data = data.get("entries", [])
source = data.get("source_file", "test.json")

print(f"   Entries: {len(entries_data)}")
print(f"   Source: {source}")

# Step 3: Auto-detect company/topic
print("\nStep 3: Auto-detecting Company/Topic...")
from app_v2 import detect_company_from_filename, detect_topic_from_filename

detected_company = ""
detected_topic = ""
if entries_data:
    first_meta = entries_data[0].get("metadata", {})
    detected_company = first_meta.get("company", "")
    detected_topic = first_meta.get("topic", "")

print(f"   From metadata - Company: '{detected_company}', Topic: '{detected_topic}'")

filename_company = detect_company_from_filename(source)
filename_topic = detect_topic_from_filename(source)
print(f"   From filename - Company: '{filename_company}', Topic: '{filename_topic}'")

# Step 4: Create KnowledgeEntry objects
print("\nStep 4: Creating KnowledgeEntry objects...")
from knowledge_schema import KnowledgeEntry

entries = []
for item in entries_data:
    entry = KnowledgeEntry(**item)
    if not entry.metadata:
        entry.metadata = {}
    entry.metadata["company"] = detected_company if detected_company else "NA"
    entry.metadata["topic"] = detected_topic if detected_topic else "Other"
    entries.append(entry)

print(f"   Created {len(entries)} KnowledgeEntry objects")

# Step 5: Call ingest_entries_to_db
print("\nStep 5: Calling ingest_entries_to_db...")
start_time = time.time()

from app_v2 import ingest_entries_to_db, get_embedding_model

print("   Loading embedding model...")
embeddings = get_embedding_model()
print(f"   Model loaded in {time.time()-start_time:.2f}s")

print("   Calling ingest_entries_to_db...")
_, count, incremental_result = ingest_entries_to_db(entries, embeddings, source)
total_time = time.time() - start_time

print(f"   ✅ SUCCESS! Count: {count}")
print(f"   Time taken: {total_time:.2f}s")

# Step 6: Check incremental result
print("\nStep 6: Incremental Update Result:")
if incremental_result:
    print(f"   New Chunks: {incremental_result.get('new', 0)}")
    print(f"   Modified: {incremental_result.get('modified', 0)}")
    print(f"   Deleted: {incremental_result.get('deleted', 0)}")
    print(f"   Unchanged: {incremental_result.get('unchanged', 0)}")
    if "embeddings_computed" in incremental_result:
        computed = incremental_result.get("embeddings_computed", 0)
        reused = incremental_result.get("embeddings_reused", 0)
        total = computed + reused
        if total > 0:
            savings = ((total - computed) / total) * 100
            print(f"   Embedding savings: {savings:.1f}% ({reused}/{total} chunks reused)")
    if "version_number" in incremental_result:
        print(f"   Document version: {incremental_result['version_number']}")
else:
    print("   ❌ No incremental result returned")

# Step 7: Verify database
print("\nStep 7: Verifying database...")
from db_reader import DBReader
from app_v2 import DB_PATH

reader = DBReader(DB_PATH)
total_docs = reader.count_documents()
print(f"   Total documents in DB: {total_docs}")

# Step 8: Cleanup
print("\nStep 8: Cleaning up...")
if os.path.exists(test_file_path):
    os.remove(test_file_path)
    print(f"   Removed test file: {test_file_path}")

print("\n" + "=" * 70)
print("Full ingest flow test completed successfully!")
print("=" * 70)
