import sys
import os

print("1. Creating embeddings...")
from langchain_huggingface import HuggingFaceEmbeddings
local_model_path = "C:/Users/HP/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
embeddings = HuggingFaceEmbeddings(model_name=local_model_path, model_kwargs={'device': 'cpu'})
print("   ✅ Embeddings created")

print("\n2. Testing embed_query...")
query_vec = embeddings.embed_query("test")
print(f"   ✅ embed_query OK: {len(query_vec)} dimensions")

print("\n3. Creating Chroma DB...")
from langchain_chroma import Chroma
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"   BASE_DIR: {BASE_DIR}")
DB_DIRECTORY = os.path.join(BASE_DIR, "my_local_database")
db = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)
print("   ✅ Chroma created")

print("\n4. Counting documents...")
count = db._collection.count()
print(f"   ✅ {count} documents")

print("\nDone!")