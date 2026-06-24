from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from knowledge_schema import KnowledgeBase
import os

def import_llm_results_from_json(json_path):
    print(f"Loading LLM classification results from: {json_path}")
    
    kb = KnowledgeBase.load_json(json_path)
    print(f"Loaded {len(kb.entries)} entries")
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma(persist_directory="./my_local_database", embedding_function=embeddings)
    
    all_db_data = db.get(include=["documents", "metadatas"])
    print(f"Database has {len(all_db_data['ids'])} entries")
    
    updated_count = 0
    for entry in kb.entries:
        idx = None
        for i, db_id in enumerate(all_db_data['ids']):
            if db_id == entry.id:
                idx = i
                break
        
        if idx is not None:
            updated_meta = entry.to_chroma_metadata()
            if entry.metadata:
                updated_meta['llm_type'] = entry.metadata.get('llm_type', '')
                updated_meta['llm_keywords'] = ','.join(entry.metadata.get('llm_keywords', []))
                updated_meta['llm_confidence'] = entry.metadata.get('llm_confidence', 0)
            if 'metadata' in updated_meta:
                del updated_meta['metadata']
            
            db._collection.update(
                ids=[entry.id],
                metadatas=[updated_meta]
            )
            updated_count += 1
            if updated_count % 50 == 0:
                print(f"Updated {updated_count} entries...")
    
    print(f"\nSuccessfully updated {updated_count} entries with LLM classification data")
    return updated_count

if __name__ == "__main__":
    json_files = [
        "knowledge_json_output/knowledge_llm_HP_20260615_164026.json",
        "knowledge_json_output/knowledge_llm_HKSB_20260614_151758.json"
    ]
    
    total_updated = 0
    for json_file in json_files:
        if os.path.exists(json_file):
            print(f"\n=== Processing {json_file} ===")
            count = import_llm_results_from_json(json_file)
            total_updated += count
        else:
            print(f"File not found: {json_file}")
    
    print(f"\n=== Total updated: {total_updated} entries ===")