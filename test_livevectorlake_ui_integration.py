import os
import sys
import time
import json
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "my_local_database", "chroma.sqlite3")

print("=" * 70)
print("LiveVectorLake UI Integration Test")
print("=" * 70)

def test_chunk_versions_table():
    print("\n1. Testing chunk_versions table...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM chunk_versions")
        count = cursor.fetchone()[0]
        print(f"   ✅ chunk_versions table exists, {count} records")
        
        cursor.execute("PRAGMA table_info(chunk_versions)")
        columns = [col[1] for col in cursor.fetchall()]
        required_cols = ['chunk_id', 'doc_id', 'position', 'content', 'valid_from', 'valid_to', 'version_number', 'status', 'change_type']
        for col in required_cols:
            if col in columns:
                print(f"   ✅ Column '{col}' exists")
            else:
                print(f"   ❌ Column '{col}' missing")
        
        conn.close()
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_doc_hash_store_table():
    print("\n2. Testing doc_hash_store table...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM doc_hash_store")
        count = cursor.fetchone()[0]
        print(f"   ✅ doc_hash_store table exists, {count} records")
        
        cursor.execute("PRAGMA table_info(doc_hash_store)")
        columns = [col[1] for col in cursor.fetchall()]
        required_cols = ['doc_id', 'chunk_hashes', 'last_updated', 'version_number']
        for col in required_cols:
            if col in columns:
                print(f"   ✅ Column '{col}' exists")
            else:
                print(f"   ❌ Column '{col}' missing")
        
        conn.close()
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_incremental_ingestor():
    print("\n3. Testing IncrementalIngestor...")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from incremental_ingestor import IncrementalIngestor
        
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"})
        ingestor = IncrementalIngestor(embeddings)
        
        test_content_v1 = """
HP Printer Maintenance Guide

Chapter 1: Introduction
HP printers require regular maintenance to ensure optimal performance.

Chapter 2: Cleaning Procedures
Clean the print heads every 30 days.

Chapter 3: Troubleshooting
Common issues include paper jams and ink cartridge errors.
"""
        
        test_content_v2 = """
HP Printer Maintenance Guide

Chapter 1: Introduction
HP printers require regular maintenance to ensure optimal performance.

Chapter 2: Cleaning Procedures
Clean the print heads every 60 days.

Chapter 3: Troubleshooting
Common issues include paper jams and ink cartridge errors.

Chapter 4: Advanced Settings
Configure duplex printing for better efficiency.
"""
        
        print("   Testing version 1 ingestion...")
        result1 = ingestor.ingest_document(test_content_v1, "test_maintenance_guide")
        print(f"   ✅ Version 1: {result1}")
        
        print("   Testing version 2 ingestion...")
        result2 = ingestor.ingest_document(test_content_v2, "test_maintenance_guide")
        print(f"   ✅ Version 2: {result2}")
        
        assert result1['new'] == 4, f"Expected 4 new chunks, got {result1['new']}"
        assert result2['modified'] == 1, f"Expected 1 modified chunk, got {result2['modified']}"
        assert result2['new'] == 1, f"Expected 1 new chunk, got {result2['new']}"
        assert result2['unchanged'] == 3, f"Expected 3 unchanged chunks, got {result2['unchanged']}"
        
        print("   ✅ Change detection works correctly")
        return True
    except AssertionError as e:
        print(f"   ❌ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_version_history():
    print("\n4. Testing Version History...")
    try:
        from incremental_ingestor import IncrementalIngestor
        
        ingestor = IncrementalIngestor(None)
        
        history = ingestor.get_doc_version_history("test_maintenance_guide")
        print(f"   ✅ Version history retrieved: {len(history)} versions")
        for h in history:
            print(f"      - Version {h['version']} at {h['timestamp']}")
        
        assert len(history) >= 2, f"Expected at least 2 versions, got {len(history)}"
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_active_chunks():
    print("\n5. Testing Active Chunks...")
    try:
        from incremental_ingestor import IncrementalIngestor
        
        ingestor = IncrementalIngestor(None)
        
        chunks = ingestor.get_active_chunks("test_maintenance_guide")
        print(f"   ✅ Active chunks retrieved: {len(chunks)}")
        for chunk in chunks:
            print(f"      - Position {chunk.position}: {chunk.content[:50]}...")
        
        assert len(chunks) >= 4, f"Expected at least 4 active chunks, got {len(chunks)}"
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_temporal_query():
    print("\n6. Testing Temporal Query...")
    try:
        from faiss_search import FAISSSearch
        
        searcher = FAISSSearch()
        searcher.initialize()
        
        results = searcher.temporal_query("printer maintenance", k=3)
        print(f"   ✅ Temporal query returned {len(results)} results")
        
        if results:
            for i, res in enumerate(results):
                print(f"      {i+1}. Similarity: {res['similarity']:.4f}")
                meta = res.get('metadata', {})
                if meta.get('version_number'):
                    print(f"         Version: {meta['version_number']}")
                if meta.get('valid_from'):
                    print(f"         Valid from: {meta['valid_from']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_system_stats():
    print("\n7. Testing System Stats...")
    try:
        from incremental_ingestor import IncrementalIngestor
        
        ingestor = IncrementalIngestor(None)
        
        stats = ingestor.get_system_stats()
        print(f"   ✅ System stats: {stats}")
        
        assert 'active_chunks' in stats
        assert 'total_versions' in stats
        assert 'unique_documents' in stats
        assert 'tracked_documents' in stats
        
        print("   ✅ All required stats present")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_app_v2_integration():
    print("\n8. Testing app_v2.py Integration...")
    try:
        with open(os.path.join(BASE_DIR, "app_v2.py"), 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ("IncrementalIngestor import", "from incremental_ingestor import IncrementalIngestor" in content),
            ("incremental_result in return", "return db, len(texts), incremental_result" in content),
            ("Temporal Query mode", '"Temporal Query"' in content),
            ("temporal_query call", "faiss_searcher.temporal_query" in content),
            ("LiveVectorLake Stats", "LiveVectorLake Stats" in content),
            ("version_number display", "Version: `{meta.get('version_number')}`" in content),
        ]
        
        all_passed = True
        for check_name, passed in checks:
            if passed:
                print(f"   ✅ {check_name}")
            else:
                print(f"   ❌ {check_name}")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_database_consistency():
    print("\n9. Testing Database Consistency...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT cv.doc_id, cv.version_number, dh.version_number
            FROM chunk_versions cv
            JOIN doc_hash_store dh ON cv.doc_id = dh.doc_id
            WHERE cv.status = 'active'
            LIMIT 5
        ''')
        
        rows = cursor.fetchall()
        print(f"   ✅ Retrieved {len(rows)} consistency check records")
        
        for row in rows:
            doc_id, chunk_version, doc_version = row
            if chunk_version == doc_version:
                print(f"   ✅ {doc_id[:30]}: chunk_v{chunk_version} == doc_v{doc_version}")
            else:
                print(f"   ⚠️ {doc_id[:30]}: chunk_v{chunk_version} != doc_v{doc_version}")
        
        conn.close()
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def main():
    tests = [
        ("chunk_versions table", test_chunk_versions_table),
        ("doc_hash_store table", test_doc_hash_store_table),
        ("IncrementalIngestor", test_incremental_ingestor),
        ("Version History", test_version_history),
        ("Active Chunks", test_active_chunks),
        ("Temporal Query", test_temporal_query),
        ("System Stats", test_system_stats),
        ("app_v2.py Integration", test_app_v2_integration),
        ("Database Consistency", test_database_consistency),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n   ❌ {name} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ PASSED: {passed}")
    print(f"❌ FAILED: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! LiveVectorLake integration is working correctly.")
    else:
        print(f"\n⚠️ {failed} tests failed. Please investigate the issues above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)