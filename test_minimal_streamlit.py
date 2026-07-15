import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_reader import DBReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIRECTORY = os.path.join(BASE_DIR, "my_local_database")
DB_PATH = os.path.join(DB_DIRECTORY, "chroma.sqlite3")

st.title("Test Minimal App")

try:
    reader = DBReader(DB_PATH)
    total = reader.count_documents()
    st.success(f"✅ Connected to database: {total} documents")
    
    metas = reader.get_all_metadatas()
    st.info(f"✅ Metadata loaded: {len(metas)} entries")
    
    docs = reader.get_documents_with_metadatas()
    st.info(f"✅ Documents loaded: {len(docs)} entries")
    
except Exception as e:
    st.error(f"❌ Error: {e}")
    import traceback
    st.code(traceback.format_exc())
