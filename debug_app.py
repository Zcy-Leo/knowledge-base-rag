import sys
import traceback

try:
    print("Step 1: Basic imports")
    import streamlit as st
    import os
    print("✅ Basic imports OK")
    
    print("\nStep 2: sys.path")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    print("✅ sys.path OK")
    
    print("\nStep 3: knowledge_schema")
    from knowledge_schema import KnowledgeBase, KnowledgeEntry
    print("✅ knowledge_schema OK")
    
    print("\nStep 4: langchain_chroma")
    from langchain_chroma import Chroma
    print("✅ langchain_chroma OK")
    
    print("\nStep 5: langchain_huggingface")
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ langchain_huggingface OK")
    
    print("\nStep 6: faiss_search import")
    import faiss_search
    print("✅ faiss_search import OK")
    
    print("\nStep 7: bm25_retriever")
    from bm25_retriever import get_bm25_retriever
    print("✅ bm25_retriever OK")
    
    print("\n" + "="*60)
    print("All imports successful!")
    
except Exception as e:
    print(f"\n❌ ERROR at step: {e}")
    traceback.print_exc()
    sys.exit(1)