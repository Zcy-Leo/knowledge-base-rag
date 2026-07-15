import sys
sys.path.insert(0, '.')

from datetime import datetime

print("=" * 60)
print("Testing BM25 Index Sync and Temporal Query")
print("=" * 60)

print("\n--- Test 1: BM25 Index Sync ---")
try:
    from bm25_retriever import get_bm25_retriever, sync_bm25_with_chroma
    from db_reader import DBReader
    
    persist_dir = "./bm25_index"
    retriever = get_bm25_retriever(persist_dir)
    print(f"BM25 index size before sync: {retriever.get_index_size()}")
    
    db_reader = DBReader("./my_local_database/chroma.sqlite3")
    retriever = sync_bm25_with_chroma(db_reader, persist_dir)
    
    if retriever:
        print(f"BM25 index size after sync: {retriever.get_index_size()}")
        print("✅ BM25 sync successful!")
    else:
        print("❌ BM25 sync failed!")
except Exception as e:
    print(f"❌ BM25 sync error: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Test 2: Temporal Query ---")
try:
    from faiss_search import FAISSSearch
    
    searcher = FAISSSearch()
    searcher.initialize()
    
    test_query = "financial analysis"
    
    print(f"\nTest query: '{test_query}'")
    
    results = searcher.temporal_query(test_query, timestamp=None, k=3)
    print(f"Results with no timestamp (latest): {len(results)}")
    if results:
        for r in results[:2]:
            print(f"  - {r.get('metadata', {}).get('source_file', 'unknown')[:40]}")
    
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    results = searcher.temporal_query(test_query, timestamp=now, k=3)
    print(f"Results with current timestamp: {len(results)}")
    
    past = "2023-01-01T00:00:00"
    results = searcher.temporal_query(test_query, timestamp=past, k=3)
    print(f"Results with past timestamp ({past}): {len(results)}")
    
    future = "2026-07-10T00:00:00"
    results = searcher.temporal_query(test_query, timestamp=future, k=3)
    print(f"Results with timestamp 2026-07-10: {len(results)}")
    
    print("✅ Temporal Query works!")
except Exception as e:
    print(f"❌ Temporal Query error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)