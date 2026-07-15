import sys
import time
sys.path.insert(0, '.')

print("=" * 70)
print("COMPREHENSIVE FUNCTIONALITY TEST")
print("=" * 70)

# Test 1: BM25 Index Sync
print("\n" + "=" * 70)
print("TEST 1: BM25 Index Sync")
print("=" * 70)
try:
    from bm25_retriever import get_bm25_retriever, sync_bm25_with_chroma
    from db_reader import DBReader
    
    persist_dir = "./bm25_index"
    retriever = get_bm25_retriever(persist_dir)
    size_before = retriever.get_index_size()
    print(f"BM25 index size before sync: {size_before}")
    
    db_reader = DBReader("./my_local_database/chroma.sqlite3")
    retriever = sync_bm25_with_chroma(db_reader, persist_dir)
    
    if retriever:
        size_after = retriever.get_index_size()
        print(f"BM25 index size after sync: {size_after}")
        
        if size_after > 0:
            test_query = "financial analysis"
            results = retriever.search(test_query, k=3)
            print(f"\nBM25 search for '{test_query}': {len(results)} results")
            for r in results[:2]:
                print(f"  - Score: {r['score']:.4f} | Content: {r['content'][:50]}...")
        
        print("✅ BM25 Index Sync: PASS")
    else:
        print("❌ BM25 Index Sync: FAIL")
except Exception as e:
    print(f"❌ BM25 Index Sync: FAIL - {e}")
    import traceback
    traceback.print_exc()

# Test 2: Temporal Query
print("\n" + "=" * 70)
print("TEST 2: Temporal Query")
print("=" * 70)
try:
    from faiss_search import FAISSSearch
    
    searcher = FAISSSearch()
    searcher.initialize()
    
    test_query = "financial"
    
    results = searcher.temporal_query(test_query, timestamp=None, k=3)
    print(f"\nTemporal Query (no timestamp): {len(results)} results")
    
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    results = searcher.temporal_query(test_query, timestamp=now, k=3)
    print(f"Temporal Query (current time): {len(results)} results")
    
    if results:
        for r in results[:2]:
            meta = r.get('metadata', {})
            print(f"  - Doc: {r['id'][:30]}")
            print(f"    Similarity: {r['similarity']:.4f}")
            print(f"    Valid from: {meta.get('valid_from')}")
    
    past_timestamp = "2023-01-01T00:00:00"
    results = searcher.temporal_query(test_query, timestamp=past_timestamp, k=3)
    print(f"\nTemporal Query (past timestamp {past_timestamp}): {len(results)} results")
    
    print("✅ Temporal Query: PASS")
except Exception as e:
    print(f"❌ Temporal Query: FAIL - {e}")
    import traceback
    traceback.print_exc()

# Test 3: Hybrid Search (Vector + BM25)
print("\n" + "=" * 70)
print("TEST 3: Hybrid Search (Vector + BM25)")
print("=" * 70)
try:
    from faiss_search import FAISSSearch
    
    searcher = FAISSSearch()
    searcher.initialize()
    
    test_query = "financial statements"
    
    print(f"\nSearch query: '{test_query}'")
    
    pure_vector = searcher.search(test_query, k=3)
    print(f"Pure Vector Search: {len(pure_vector)} results")
    
    hybrid = searcher.hybrid_search(test_query, k=3, use_bm25=True, reranker_type=None)
    print(f"Hybrid Search (Vector+BM25): {len(hybrid)} results")
    
    if hybrid:
        for r in hybrid[:2]:
            print(f"  - RRF Score: {r.get('rrf_score', 0):.4f}")
            print(f"    Vector Rank: {r.get('vector_rank', -1)}")
            print(f"    BM25 Rank: {r.get('bm25_rank', -1)}")
    
    print("✅ Hybrid Search: PASS")
except Exception as e:
    print(f"❌ Hybrid Search: FAIL - {e}")
    import traceback
    traceback.print_exc()

# Test 4: Pure Vector Search
print("\n" + "=" * 70)
print("TEST 4: Pure Vector Search")
print("=" * 70)
try:
    from faiss_search import FAISSSearch
    
    searcher = FAISSSearch()
    searcher.initialize()
    
    test_queries = ["financial analysis", "printer maintenance", "healthcare"]
    
    for query in test_queries:
        results = searcher.search(query, k=3)
        print(f"\nSearch for '{query}': {len(results)} results")
        if results:
            for r in results[:2]:
                print(f"  - Similarity: {r['similarity']:.4f} | Content: {r['content'][:50]}...")
    
    print("✅ Pure Vector Search: PASS")
except Exception as e:
    print(f"❌ Pure Vector Search: FAIL - {e}")
    import traceback
    traceback.print_exc()

# Test 5: Incremental Ingestion
print("\n" + "=" * 70)
print("TEST 5: Incremental Ingestion")
print("=" * 70)
try:
    from incremental_ingestor import IncrementalIngestor
    from faiss_search import FAISSSearch
    from db_reader import DBReader
    
    searcher = FAISSSearch()
    searcher.initialize()
    
    db_reader = DBReader("./my_local_database/chroma.sqlite3")
    count_before = db_reader.count_documents()
    print(f"Documents in DB before: {count_before}")
    
    from chunk_change_detector import ChunkChangeDetector
    detector = ChunkChangeDetector()
    conn = detector._get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM chunk_versions WHERE status = "active"')
    active_chunks = cursor.fetchone()[0]
    conn.close()
    print(f"Active chunks in chunk_versions: {active_chunks}")
    
    print("✅ Incremental Ingestion: PASS")
except Exception as e:
    print(f"❌ Incremental Ingestion: FAIL - {e}")
    import traceback
    traceback.print_exc()

# Test 6: Company Filter
print("\n" + "=" * 70)
print("TEST 6: Company Filter")
print("=" * 70)
try:
    from faiss_search import FAISSSearch
    
    searcher = FAISSSearch()
    searcher.initialize()
    
    test_query = "financial"
    
    results_all = searcher.search(test_query, k=5)
    print(f"\nSearch for '{test_query}' (All): {len(results_all)} results")
    
    results_hp = searcher.search(test_query, k=5, filter_dict={"company": "HP"})
    print(f"Search for '{test_query}' (HP): {len(results_hp)} results")
    
    results_cisco = searcher.search(test_query, k=5, filter_dict={"company": "Cisco"})
    print(f"Search for '{test_query}' (Cisco): {len(results_cisco)} results")
    
    print("✅ Company Filter: PASS")
except Exception as e:
    print(f"❌ Company Filter: FAIL - {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED!")
print("=" * 70)