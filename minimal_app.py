import streamlit as st
import os

st.title("Minimal Test App")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIRECTORY = os.path.join(BASE_DIR, "my_local_database")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'})
    st.success("✅ Embedding model loaded")
except Exception as e:
    st.error(f"❌ Embedding model failed: {e}")
    import traceback
    traceback.print_exc()
    st.stop()

try:
    from langchain_chroma import Chroma
    db = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)
    st.success(f"✅ Chroma DB connected (documents: {db._collection.count()})")
except Exception as e:
    st.error(f"❌ Chroma DB failed: {e}")
    import traceback
    traceback.print_exc()
    st.stop()

try:
    from faiss_search import FAISSSearch
    searcher = FAISSSearch(embedding_model=embeddings)
    searcher.initialize()
    st.success(f"✅ FAISS search initialized")
except Exception as e:
    st.error(f"❌ FAISS search failed: {e}")
    import traceback
    traceback.print_exc()
    st.stop()

st.write("All components loaded successfully!")