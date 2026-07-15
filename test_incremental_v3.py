import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "my_local_database", "chroma.sqlite3")
LVL_DB_PATH = os.path.join(BASE_DIR, "my_local_database", "livevectorlake.db")

print("======================================================================")
print("Incremental Update v3 Test - Comprehensive Boundary Case Testing")
print("======================================================================")

def delete_test_docs():
    import sqlite3
    conn = sqlite3.connect(LVL_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DROP TABLE IF EXISTS entry_versions")
        cursor.execute("DROP TABLE IF EXISTS doc_metadata")
        conn.commit()
        print("Cleared existing test documents from livevectorlake.db")
    except sqlite3.OperationalError:
        print("No existing test documents to clear")
    conn.close()

delete_test_docs()

reader = None
try:
    from db_reader import DBReader
    reader = DBReader(DB_PATH)
    total_before = reader.count_documents()
    print(f"\nStep 0: Database connected: {total_before} documents")
except Exception as e:
    print(f"Database connection error: {e}")
    sys.exit(1)

def create_test_entry_dicts(entries_data):
    entry_dicts = []
    for i, data in enumerate(entries_data):
        text = f"{data['title']}\n\n{data['content']}"
        entry_dicts.append({
            'id': data['id'],
            'text': text,
            'metadata': {'company': 'TestCo', 'topic': 'test'}
        })
    return entry_dicts

def run_test(test_name, entry_dicts, source_label, expected_new=0, expected_modified=0, expected_deleted=0, expected_unchanged=0):
    print(f"\n--- {test_name} ---")
    
    from incremental_ingestor import IncrementalIngestor
    ingestor = IncrementalIngestor()
    
    print(f"  doc_id={source_label}, entries={len(entry_dicts)}")
    
    result = ingestor.ingest_entries(entry_dicts, source_label)
    print(f"  Incremental result: {result}")
    
    actual_new = result.get('new', 0)
    actual_modified = result.get('modified', 0)
    actual_deleted = result.get('deleted', 0)
    actual_unchanged = result.get('unchanged', 0)
    version = result.get('version_number', 0)
    
    success = True
    if actual_new != expected_new:
        print(f"  ❌ NEW: Expected {expected_new}, got {actual_new}")
        success = False
    if actual_modified != expected_modified:
        print(f"  ❌ MODIFIED: Expected {expected_modified}, got {actual_modified}")
        success = False
    if actual_deleted != expected_deleted:
        print(f"  ❌ DELETED: Expected {expected_deleted}, got {actual_deleted}")
        success = False
    if actual_unchanged != expected_unchanged:
        print(f"  ❌ UNCHANGED: Expected {expected_unchanged}, got {actual_unchanged}")
        success = False
    
    if success:
        print(f"  ✅ Test PASSED (version {version})")
    else:
        print(f"  ❌ Test FAILED")
    
    return success, result

test_results = []

entries_v1 = [
    {'id': 'test-entry-1', 'content': 'This is the first test entry', 'title': 'Entry 1'},
    {'id': 'test-entry-2', 'content': 'This is the second test entry', 'title': 'Entry 2'},
    {'id': 'test-entry-3', 'content': 'This is the third test entry', 'title': 'Entry 3'},
]

entries_v2_delete = [
    {'id': 'test-entry-1', 'content': 'This is the first test entry', 'title': 'Entry 1'},
    {'id': 'test-entry-3', 'content': 'This is the third test entry', 'title': 'Entry 3'},
]

entries_v3_modify = [
    {'id': 'test-entry-1', 'content': 'This is the FIRST test entry (modified)', 'title': 'Entry 1'},
    {'id': 'test-entry-3', 'content': 'This is the third test entry', 'title': 'Entry 3'},
]

entries_v4_add = [
    {'id': 'test-entry-1', 'content': 'This is the FIRST test entry (modified)', 'title': 'Entry 1'},
    {'id': 'test-entry-3', 'content': 'This is the third test entry', 'title': 'Entry 3'},
    {'id': 'test-entry-4', 'content': 'This is the fourth test entry (new)', 'title': 'Entry 4'},
]

entries_v5_full_modify = [
    {'id': 'test-entry-1', 'content': 'Completely different content for entry 1', 'title': 'Entry 1 Modified'},
    {'id': 'test-entry-3', 'content': 'Third entry also changed', 'title': 'Entry 3 Modified'},
    {'id': 'test-entry-4', 'content': 'Fourth entry updated too', 'title': 'Entry 4 Modified'},
]

entries_v6_empty = []

print("\n\n=== Test Case 1: Initial Upload (v1 - 3 entries) ===")
entry_dicts_v1 = create_test_entry_dicts(entries_v1)
success, result = run_test("Initial Upload", entry_dicts_v1, "test_initial", 
                          expected_new=3, expected_modified=0, expected_deleted=0, expected_unchanged=0)
test_results.append(("Initial Upload", success))

print("\n\n=== Test Case 2: Delete 1 entry (v2 - 2 entries) ===")
entry_dicts_v2 = create_test_entry_dicts(entries_v2_delete)
success, result = run_test("Delete Entry", entry_dicts_v2, "test_initial",
                          expected_new=0, expected_modified=0, expected_deleted=1, expected_unchanged=2)
test_results.append(("Delete Entry", success))

print("\n\n=== Test Case 3: Modify 1 entry (v3 - 2 entries) ===")
entry_dicts_v3 = create_test_entry_dicts(entries_v3_modify)
success, result = run_test("Modify Entry", entry_dicts_v3, "test_initial",
                          expected_new=0, expected_modified=1, expected_deleted=0, expected_unchanged=1)
test_results.append(("Modify Entry", success))

print("\n\n=== Test Case 4: Add 1 entry (v4 - 3 entries) ===")
entry_dicts_v4 = create_test_entry_dicts(entries_v4_add)
success, result = run_test("Add Entry", entry_dicts_v4, "test_initial",
                          expected_new=1, expected_modified=0, expected_deleted=0, expected_unchanged=2)
test_results.append(("Add Entry", success))

print("\n\n=== Test Case 5: Modify all entries (v5 - 3 entries) ===")
entry_dicts_v5 = create_test_entry_dicts(entries_v5_full_modify)
success, result = run_test("Modify All", entry_dicts_v5, "test_initial",
                          expected_new=0, expected_modified=3, expected_deleted=0, expected_unchanged=0)
test_results.append(("Modify All", success))

print("\n\n=== Test Case 6: Upload empty document (v6 - 0 entries) ===")
entry_dicts_v6 = create_test_entry_dicts(entries_v6_empty)
success, result = run_test("Empty Document", entry_dicts_v6, "test_empty",
                          expected_new=0, expected_modified=0, expected_deleted=0, expected_unchanged=0)
test_results.append(("Empty Document", success))

print("\n\n=== Test Case 7: Upload same document twice (no changes) ===")
entry_dicts_v5_again = create_test_entry_dicts(entries_v5_full_modify)
success, result = run_test("No Changes", entry_dicts_v5_again, "test_initial",
                          expected_new=0, expected_modified=0, expected_deleted=0, expected_unchanged=3)
test_results.append(("No Changes", success))

print("\n\n=== Test Case 8: Complex scenario - Add + Modify + Delete simultaneously ===")
entries_complex = [
    {'id': 'test-entry-1', 'content': 'Entry 1 has been modified', 'title': 'Entry 1 Modified'},
    {'id': 'test-entry-5', 'content': 'This is a completely new entry', 'title': 'Entry 5 New'},
]
entry_dicts_complex = create_test_entry_dicts(entries_complex)
success, result = run_test("Complex (Add+Modify+Delete)", entry_dicts_complex, "test_initial",
                          expected_new=1, expected_modified=1, expected_deleted=2, expected_unchanged=0)
test_results.append(("Complex (Add+Modify+Delete)", success))

print("\n\n=== Test Case 9: Re-upload same document after deletion (recovery) ===")
entry_dicts_recover = create_test_entry_dicts(entries_v1)
success, result = run_test("Recovery", entry_dicts_recover, "test_recovery",
                          expected_new=3, expected_modified=0, expected_deleted=0, expected_unchanged=0)
test_results.append(("Recovery", success))

print("\n\n=== Test Case 10: Upload with different source label ===")
entry_dicts_diff_source = create_test_entry_dicts(entries_v1)
success, result = run_test("Different Source Label", entry_dicts_diff_source, "test_different_source",
                          expected_new=3, expected_modified=0, expected_deleted=0, expected_unchanged=0)
test_results.append(("Different Source Label", success))

print("\n\n======================================================================")
print("Test Summary")
print("======================================================================")

passed = sum(1 for _, s in test_results if s)
failed = len(test_results) - passed

print(f"\nTotal tests: {len(test_results)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

for name, success in test_results:
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"  {status}: {name}")

if failed == 0:
    print("\n🎉 All tests passed!")
else:
    print(f"\n⚠️ {failed} tests failed!")

print("\n======================================================================")
print("Incremental update v3 test completed!")
print("======================================================================")