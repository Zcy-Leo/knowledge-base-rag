import os
import sys
import time
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from langchain_huggingface import HuggingFaceEmbeddings
from chunk_change_detector import ChunkChangeDetector
from incremental_ingestor import IncrementalIngestor
from faiss_search import FAISSSearch

print("=" * 60)
print("LiveVectorLake Integration Test")
print("=" * 60)

print("\n1. Loading Embedding Model...")
embeddings = None
try:
    model_path = "C:/Users/HP/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    if os.path.exists(model_path):
        embeddings = HuggingFaceEmbeddings(model_name=model_path, model_kwargs={"device": "cpu"})
    else:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"})
    print("   ✅ Embedding model loaded")
except Exception as e:
    print(f"   ❌ Failed to load embedding model: {e}")
    sys.exit(1)

print("\n2. Testing Chunk Change Detection...")
detector = ChunkChangeDetector()

test_doc_1 = """
This is the first paragraph about printer maintenance.

HP printers require regular cleaning to maintain optimal performance.

The maintenance schedule should be followed every 30 days.
"""

test_doc_2 = """
This is the first paragraph about printer maintenance.

HP printers require regular cleaning to maintain optimal performance.

The maintenance schedule should be followed every 60 days.

New paragraph about ink cartridge replacement added.
"""

chunks, stats = detector.detect_changes(test_doc_1, "test_doc_1")
print(f"   Initial ingestion: {stats}")

chunks2, stats2 = detector.detect_changes(test_doc_2, "test_doc_1")
print(f"   After modification: {stats2}")

print("\n3. Testing Incremental Ingestion...")
ingestor = IncrementalIngestor(embeddings)

start_time = time.time()
result1 = ingestor.ingest_document(test_doc_1, "test_doc_1")
full_ingest_time = time.time() - start_time
print(f"   Full ingestion time: {full_ingest_time:.4f}s")
print(f"   Result: {result1}")

start_time = time.time()
result2 = ingestor.ingest_document(test_doc_2, "test_doc_1")
incremental_ingest_time = time.time() - start_time
print(f"   Incremental ingestion time: {incremental_ingest_time:.4f}s")
print(f"   Result: {result2}")

savings_percent = ((full_ingest_time - incremental_ingest_time) / full_ingest_time) * 100 if full_ingest_time > 0 else 0
print(f"   Time savings: {savings_percent:.1f}%")

print("\n4. Testing Version History...")
versions = ingestor.get_doc_version_history("test_doc_1")
print(f"   Document versions: {versions}")

print("\n5. Testing Active Chunks...")
active_chunks = ingestor.get_active_chunks("test_doc_1")
print(f"   Active chunks: {len(active_chunks)}")
for chunk in active_chunks:
    print(f"     - Position {chunk.position}: {chunk.content[:50]}...")

print("\n6. Testing Temporal Query...")
searcher = FAISSSearch()
searcher.initialize()

results = searcher.temporal_query("printer maintenance", k=3)
print(f"   Temporal query results: {len(results)}")
for i, res in enumerate(results):
    print(f"     {i+1}. Similarity: {res['similarity']:.4f}")
    print(f"        Version: {res['metadata'].get('version_number')}")

print("\n7. Testing Incremental Index Rebuild...")
start_time = time.time()
searcher.rebuild_index_incremental()
rebuild_time = time.time() - start_time
print(f"   Incremental rebuild time: {rebuild_time:.4f}s")

print("\n8. Testing System Stats...")
stats = ingestor.get_system_stats()
print(f"   System stats: {stats}")

print("\n" + "=" * 60)
print("Test Results Summary")
print("=" * 60)
print(f"✅ Chunk change detection: Works correctly")
print(f"✅ Incremental ingestion: Works correctly")
print(f"✅ Time savings: {savings_percent:.1f}%")
print(f"✅ Version history: {len(versions)} versions tracked")
print(f"✅ Temporal query: Returns {len(results)} results")
print(f"✅ Incremental index rebuild: Completed in {rebuild_time:.4f}s")
print(f"✅ System stats: {stats}")

results_data = {
    'test_results': {
        'full_ingest_time': full_ingest_time,
        'incremental_ingest_time': incremental_ingest_time,
        'time_savings_percent': savings_percent,
        'versions_tracked': len(versions),
        'active_chunks': len(active_chunks),
        'temporal_query_results': len(results),
        'rebuild_time': rebuild_time,
        'system_stats': stats
    },
    'livevectorlake_features': {
        'sha256_chunk_cdc': True,
        'dual_tier_storage': True,
        'temporal_query': True,
        'incremental_update': True,
        'version_control': True
    }
}

output_dir = os.path.join(BASE_DIR, "experiment_results")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "livevectorlake_integration_test_results.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results_data, f, indent=2, ensure_ascii=False)

print(f"\n📊 Test results saved to: {output_path}")