import json
import os

# 使用绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIRECTORY = os.path.join(BASE_DIR, "my_local_database")

def ingest_hp_data():
    json_file = os.path.join(BASE_DIR, "knowledge_json_output", "knowledge_llm_HP_restored.json")
    
    if not os.path.exists(json_file):
        print(f"File not found: {json_file}")
        return
    
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    entries_data = data.get("entries", [])
    print(f"Found {len(entries_data)} entries")
    
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    if os.path.exists(DB_DIRECTORY):
        db = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)
    else:
        db = Chroma.from_texts(
            texts=[],
            embedding=embeddings,
            persist_directory=DB_DIRECTORY
        )
    
    from knowledge_schema import KnowledgeEntry
    
    texts = []
    metas = []
    
    for item in entries_data:
        entry = KnowledgeEntry(**item)
        texts.append(entry.to_chroma_text())
        metas.append(entry.to_chroma_metadata())
    
    db.add_texts(texts=texts, metadatas=metas)
    print(f"Added {len(texts)} entries to database")
    print(f"Total in database: {db.count()}")

if __name__ == "__main__":
    ingest_hp_data()