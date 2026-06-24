from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import json

print("=== Database Analysis Report ===\n")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
db = Chroma(persist_directory="./my_local_database", embedding_function=embeddings)

all_data = db.get(include=["documents", "metadatas"])

total = len(all_data["ids"])
print(f"Total entries in database: {total}")

classified_count = 0
unclassified_count = 0
general_count = 0
sop_count = 0
faq_count = 0

company_distribution = {}
source_distribution = {}

for i, meta in enumerate(all_data["metadatas"]):
    if meta is None:
        meta = {}
    
    has_llm_type = False
    llm_type = "unknown"
    
    if isinstance(meta, dict):
        if "metadata" in meta and isinstance(meta["metadata"], dict):
            if "llm_type" in meta["metadata"]:
                has_llm_type = True
                llm_type = meta["metadata"]["llm_type"]
        
        if has_llm_type:
            classified_count += 1
        else:
            unclassified_count += 1
        
        if "type" in meta:
            t = meta["type"]
            if t == "sop_step":
                sop_count += 1
            elif t == "faq":
                faq_count += 1
            elif t == "general":
                general_count += 1
        
        company = meta.get("company", meta.get("metadata", {}).get("company", "Unassigned"))
        company_distribution[company] = company_distribution.get(company, 0) + 1
        
        source = meta.get("source_file", "unknown")
        source_distribution[source] = source_distribution.get(source, 0) + 1

print("\n--- LLM Classification Status ---")
print(f"OK Classified by Gemini API: {classified_count} ({(classified_count/total)*100:.1f}%)")
print(f"-- Not classified yet: {unclassified_count} ({(unclassified_count/total)*100:.1f}%)")

print("\n--- Content Type Distribution (Rule-based) ---")
print(f"General: {general_count}")
print(f"SOP Steps: {sop_count}")
print(f"FAQ: {faq_count}")

print("\n--- Company Distribution ---")
for company, count in sorted(company_distribution.items(), key=lambda x: -x[1]):
    print(f"  {company}: {count} entries")

print("\n--- Source File Distribution ---")
for source, count in sorted(source_distribution.items(), key=lambda x: -x[1]):
    display_source = source if len(source) <= 40 else source[:37] + "..."
    print(f"  {display_source}: {count} entries")

print("\n=== Summary ===")
if classified_count == total:
    print("All entries have been classified by Gemini API")
elif classified_count > 0:
    print(f"Partial classification: {classified_count}/{total} entries processed")
    print("   You can run LLM Classification in the UI to process remaining entries")
else:
    print("No entries have been classified by Gemini API yet")

if unclassified_count > 0:
    print(f"\nTip: Go to Database tab -> Click 'Run LLM Classification' to process remaining {unclassified_count} entries")