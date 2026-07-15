import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Step 1: Setup")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_DIRECTORY = os.path.join(BASE_DIR, "my_local_database")
    print(f"✅ DB_DIR: {DB_DIRECTORY}")
    
    print("\nStep 2: Create embeddings")
    from langchain_huggingface import HuggingFaceEmbeddings
    local_model_path = "C:/Users/HP/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    embeddings = HuggingFaceEmbeddings(model_name=local_model_path, model_kwargs={'device': 'cpu'})
    print("✅ Embeddings created")
    
    print("\nStep 3: Test embed_query")
    query_vec = embeddings.embed_query("test")
    print(f"✅ embed_query OK: {len(query_vec)} dimensions")
    
    print("\nStep 4: Connect to Chroma")
    from langchain_chroma import Chroma
    db = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)
    count = db._collection.count()
    print(f"✅ Chroma OK: {count} documents")
    
    print("\nStep 5: Initialize FAISSSearch")
    from faiss_search import FAISSSearch
    searcher = FAISSSearch(embedding_model=embeddings)
    searcher.initialize()
    print("✅ FAISSSearch initialized")
    
    print("\nStep 6: Test search")
    results = searcher.search("test", k=3)
    print(f"✅ Search OK: {len(results)} results")
    
    print("\n" + "="*60)
    print("All tests passed!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)