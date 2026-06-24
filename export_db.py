from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import json
import os
from datetime import datetime

def export_database(output_file=None):
    print("Connecting to local database...")
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma(persist_directory="./my_local_database", embedding_function=embeddings)
    
    all_data = db.get(include=["documents", "metadatas"])
    
    total = len(all_data["ids"])
    print(f"Found {total} entries in database.")
    
    export_data = []
    for i, (entry_id, doc, meta) in enumerate(zip(
        all_data["ids"], 
        all_data["documents"], 
        all_data["metadatas"]
    )):
        entry = {
            "id": entry_id,
            "content": doc,
            "metadata": meta if meta else {}
        }
        export_data.append(entry)
    
    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"db_export_{ts}.json"
    
    os.makedirs("exports", exist_ok=True)
    output_path = os.path.join("exports", output_file)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully exported {total} entries to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    return output_path

if __name__ == "__main__":
    export_database()