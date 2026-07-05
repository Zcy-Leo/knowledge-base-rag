from faiss_search import FAISSSearch

searcher = FAISSSearch()

print("=== Analyzing Search Results for 'how to install printer' ===")
print()

queries = ["install printer", "printer setup", "setup wizard", "connect printer"]
for query in queries:
    print(f"Query: '{query}'")
    results = searcher.search(query, k=5)
    for i, r in enumerate(results):
        title = r["metadata"].get("title", "Untitled")[:60]
        content_preview = r["content"][:100].replace("\n", " ")
        print(f"  {i+1}. [{r['similarity']:.4f}] {title}")
        print(f"     {content_preview}")
    print()
