from faiss_search import FAISSSearch
import time

searcher = FAISSSearch()

queries = [
    "how to reset HP printer password",
    "printer installation guide",
    "WiFi connection setup",
    "how to fix paper jam",
    "scan documents to computer"
]

print("="*80)
print("CrossEncoder vs Gemini Reranking Comparison")
print("="*80)
print()

for query in queries:
    print(f"Query: '{query}'")
    print("-" * 60)
    
    # CrossEncoder
    start_time = time.time()
    ce_results = searcher.hybrid_search(query, k=5, use_bm25=True, reranker_type='crossencoder')
    ce_time = time.time() - start_time
    
    # Gemini
    start_time = time.time()
    gemini_results = searcher.hybrid_search(query, k=5, use_bm25=True, reranker_type='gemini')
    gemini_time = time.time() - start_time
    
    print(f"CrossEncoder ({ce_time:.2f}s):")
    for i, r in enumerate(ce_results):
        title = r["metadata"].get("title", "Untitled")[:50]
        score = r.get('rerank_score', r.get('rrf_score', 0))
        print(f"  {i+1}. [{score:.4f}] {title}")
    
    print(f"\nGemini ({gemini_time:.2f}s):")
    for i, r in enumerate(gemini_results):
        title = r["metadata"].get("title", "Untitled")[:50]
        score = r.get('rerank_score', r.get('rrf_score', 0))
        print(f"  {i+1}. [{score:.4f}] {title}")
    
    print("\n" + "="*80)
    print()
