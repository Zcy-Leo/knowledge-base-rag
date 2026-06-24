"""
assign_company_to_db.py
============================
Analyze database entries and assign company based on source file name.
Update the 'company' field in metadata for entries that don't have it.
"""

import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration
DB_DIRECTORY = "./my_local_database"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Company mapping based on source file patterns
COMPANY_MAPPING = {
    # HP related
    "hp": "HP",
    "printer": "HP",
    
    # NETGEAR / Router related
    "router": "NETGEAR",
    "netgear": "NETGEAR",
    "kb.netgear": "NETGEAR",
    
    # Samsung
    "samsung": "Samsung",
    "galaxy": "Samsung",
    
    # Apple
    "apple": "Apple",
    "iphone": "Apple",
    "macbook": "Apple",
    
    # Sony
    "sony": "Sony",
    "playstation": "Sony",
    
    # Dell
    "dell": "Dell",
    
    # User Manual generic
    "usermanual": "Generic",
    "manual": "Generic",
    "sample_manual": "Generic",
}


def infer_company_from_source(source_file: str) -> str:
    """Infer company name from source file name."""
    source_lower = source_file.lower()
    
    for pattern, company in COMPANY_MAPPING.items():
        if pattern in source_lower:
            return company
    
    # If no match, return "Unknown"
    return "Unknown"


def main():
    print("="*60)
    print("  Company Assignment Tool for Database Entries")
    print("="*60)
    
    # Load embedding model
    print("\n[1] Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load Chroma database
    print("\n[2] Loading Chroma database...")
    if not os.path.exists(DB_DIRECTORY):
        print(f"Error: Database directory '{DB_DIRECTORY}' not found!")
        return
    
    vectorstore = Chroma(
        persist_directory=DB_DIRECTORY,
        embedding_function=embeddings
    )
    
    # Get all entries
    print("\n[3] Retrieving all entries from database...")
    all_entries = vectorstore.get()
    
    if not all_entries or not all_entries.get('ids'):
        print("No entries found in database!")
        return
    
    ids = all_entries['ids']
    metadatas = all_entries.get('metadatas', [])
    documents = all_entries.get('documents', [])
    
    print(f"Total entries: {len(ids)}")
    
    # Analyze current company distribution
    print("\n[4] Analyzing current company distribution...")
    company_stats = {}
    no_company_entries = []
    
    for i, (id_, meta) in enumerate(zip(ids, metadatas)):
        company = meta.get('company', None)
        source = meta.get('source_file', 'Unknown')
        
        if company:
            company_stats[company] = company_stats.get(company, 0) + 1
        else:
            inferred_company = infer_company_from_source(source)
            no_company_entries.append({
                'id': id_,
                'index': i,
                'source': source,
                'inferred_company': inferred_company
            })
    
    # Print current distribution
    print("\n[Current Company Distribution]")
    print("-"*40)
    if company_stats:
        for company, count in sorted(company_stats.items()):
            print(f"  {company}: {count} entries")
    else:
        print("  No entries with company assigned")
    
    print(f"\n  Entries without company: {len(no_company_entries)}")
    
    if no_company_entries:
        print("\n[Entries needing company assignment]")
        print("-"*40)
        
        # Group by inferred company
        inferred_stats = {}
        for entry in no_company_entries:
            inferred = entry['inferred_company']
            inferred_stats[inferred] = inferred_stats.get(inferred, 0) + 1
        
        for company, count in sorted(inferred_stats.items()):
            print(f"  {company}: {count} entries (inferred)")
        
        # Ask user to confirm
        print("\n" + "="*60)
        print("  Proposed Company Assignments:")
        print("="*60)
        
        for company, count in sorted(inferred_stats.items()):
            print(f"  Assign '{company}' to {count} entries")
        
        print("\n[5] Updating database...")
        
        # Update entries with inferred company
        updated_count = 0
        for entry in no_company_entries:
            if entry['inferred_company'] != "Unknown":
                # Update metadata
                new_metadata = metadatas[entry['index']].copy()
                new_metadata['company'] = entry['inferred_company']
                
                # Update in vectorstore
                vectorstore._collection.update(
                    ids=[entry['id']],
                    metadatas=[new_metadata]
                )
                updated_count += 1
        
        print(f"\n[DONE] Updated {updated_count} entries with company assignment")
        
        # Show final distribution
        print("\n[Final Company Distribution]")
        print("-"*40)
        
        # Reload to get updated stats
        all_entries_updated = vectorstore.get()
        final_stats = {}
        for meta in all_entries_updated.get('metadatas', []):
            company = meta.get('company', 'Unknown')
            final_stats[company] = final_stats.get(company, 0) + 1
        
        for company, count in sorted(final_stats.items()):
            print(f"  {company}: {count} entries")
    
    else:
        print("\n[DONE] All entries already have company assigned!")
    
    print("\n" + "="*60)
    print("  Done!")
    print("="*60)


if __name__ == "__main__":
    main()