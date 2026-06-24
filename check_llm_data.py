from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("Checking LLM classification data in database...\n")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
db = Chroma(persist_directory="./my_local_database", embedding_function=embeddings)

all_data = db.get(include=["documents", "metadatas"])

total = len(all_data["ids"])
print(f"Total entries: {total}")

llm_count = 0
no_llm_count = 0
llm_types = {}

for i, meta in enumerate(all_data["metadatas"]):
    has_llm = False
    
    if meta is not None and isinstance(meta, dict):
        if "metadata" in meta and isinstance(meta["metadata"], dict):
            if "llm_type" in meta["metadata"]:
                has_llm = True
                llm_type = meta["metadata"]["llm_type"]
                llm_types[llm_type] = llm_types.get(llm_type, 0) + 1
        elif "llm_type" in meta:
            has_llm = True
            llm_type = meta["llm_type"]
            llm_types[llm_type] = llm_types.get(llm_type, 0) + 1
    
    if has_llm:
        llm_count += 1
    else:
        no_llm_count += 1

print(f"\nEntries with LLM classification: {llm_count}")
print(f"Entries without LLM classification: {no_llm_count}")

if llm_types:
    print("\nLLM Type distribution:")
    for t, cnt in llm_types.items():
        print(f"  {t}: {cnt}")

print("\n--- Sample metadata structure ---")
sample_count = 0
for i, meta in enumerate(all_data["metadatas"]):
    if meta and isinstance(meta, dict):
        print(f"\nEntry {i}:")
        print(f"  Keys: {list(meta.keys())}")
        if "metadata" in meta:
            print(f"  metadata keys: {list(meta['metadata'].keys()) if isinstance(meta['metadata'], dict) else 'Not a dict'}")
        sample_count += 1
        if sample_count >= 3:
            break