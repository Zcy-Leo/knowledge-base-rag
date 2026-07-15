import sys

print("Step 1: Basic imports")
import streamlit as st
import os
import json
import shutil
import tempfile
from datetime import datetime
print("✅ Basic imports OK")

print("\nStep 2: knowledge_schema")
from knowledge_schema import KnowledgeBase, KnowledgeEntry
print("✅ knowledge_schema OK")

print("\nStep 3: HuggingFaceEmbeddings")
from langchain_huggingface import HuggingFaceEmbeddings
local_model_path = "C:/Users/HP/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
embeddings = HuggingFaceEmbeddings(model_name=local_model_path, model_kwargs={'device': 'cpu'})
print("✅ HuggingFaceEmbeddings OK")

print("\nStep 4: Chroma")
from langchain_chroma import Chroma
db = Chroma(persist_directory='my_local_database', embedding_function=embeddings)
print(f"✅ Chroma OK ({db._collection.count()} docs)")

print("\nStep 5: FAISS")
try:
    from faiss_search import FAISSSearch
    searcher = FAISSSearch(embedding_model=embeddings)
    searcher.initialize()
    print("✅ FAISSSearch OK")
except Exception as e:
    print(f"❌ FAISSSearch failed: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 6: BM25")
try:
    from bm25_retriever import get_bm25_retriever
    print("✅ BM25 retriever OK")
except Exception as e:
    print(f"❌ BM25 failed: {e}")

print("\n" + "="*60)
print("All modules tested!")