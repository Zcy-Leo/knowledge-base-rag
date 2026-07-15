import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from faiss_search import FAISSSearch

print("=== Testing FAISS Search ===")
searcher = FAISSSearch()
print("Initializing...")
searcher.initialize()
print("Initialized!")

print("\n=== Testing Vector Search ===")
start = time.time()
results = searcher.search("paper jam", k=10)
latency = time.time() - start
print(f"Found {len(results)} results in {latency:.3f}s")
for r in results[:3]:
    print(f"  - ID: {r['id']}, Title: {r.get('metadata', {}).get('title', '')[:50]}")

print("\n=== Testing Hybrid Search ===")
start = time.time()
results = searcher.hybrid_search("paper jam", k=10)
latency = time.time() - start
print(f"Found {len(results)} results in {latency:.3f}s")
for r in results[:3]:
    print(f"  - ID: {r.get('doc_id', r.get('id', ''))}, Title: {r.get('metadata', {}).get('title', '')[:50]}")

print("\n=== Testing CrossEncoder Reranking ===")
start = time.time()
results = searcher.hybrid_search("paper jam", k=10, reranker_type='crossencoder', reranker_model='cross-encoder/ms-marco-MiniLM-L-6-v2')
latency = time.time() - start
print(f"Found {len(results)} results in {latency:.3f}s")
for r in results[:3]:
    print(f"  - ID: {r.get('doc_id', r.get('id', ''))}, Title: {r.get('metadata', {}).get('title', '')[:50]}")

print("\n✅ All tests completed successfully!")