"""
Batch process documents from Unstructured example-docs directory.
Auto-detects format and uses best extractor.
"""

import os
import json
import shutil
from datetime import datetime

SOURCE_DIR = "./unstructured_extracted/unstructured-main/example-docs"
OUTPUT_DIR = "./knowledge_json_output/batch_results"
COMPANY_NAME = "Unstructured_Demo"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FORMAT_CATEGORIES = {
    "pdf": [".pdf"],
    "word": [".doc", ".docx"],
    "excel": [".xls", ".xlsx", ".csv", ".tsv"],
    "powerpoint": [".ppt", ".pptx"],
    "image": [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".heic"],
    "email": [".eml", ".msg"],
    "text": [".txt", ".md", ".html", ".xml", ".json"],
    "other": [".odt", ".rtf", ".epub"],
}

def categorize_file(filepath):
    _, ext = os.path.splitext(filepath.lower())
    for category, exts in FORMAT_CATEGORIES.items():
        if ext in exts:
            return category
    return "other"

def collect_all_files(base_dir):
    files = []
    for root, dirs, filenames in os.walk(base_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            files.append(filepath)
    return files

def main():
    all_files = collect_all_files(SOURCE_DIR)
    print(f"Found {len(all_files)} files in total")
    
    categorized = {}
    for filepath in all_files:
        category = categorize_file(filepath)
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(filepath)
    
    print("\n=== File Distribution ===")
    total_to_process = 0
    for category, files in categorized.items():
        print(f"  {category}: {len(files)} files")
        total_to_process += len(files)
    
    print(f"\nTotal files to process: {total_to_process}")
    
    print("\n=== Starting Batch Processing ===")
    
    from unified_extractor import extract_knowledge_from_file
    from knowledge_schema import KnowledgeBase
    
    all_kbs = []
    processed_count = 0
    
    for category, files in categorized.items():
        print(f"\n--- Processing {category} ({len(files)} files) ---")
        
        for filepath in files:
            filename = os.path.basename(filepath)
            print(f"  [{processed_count+1}/{total_to_process}] {filename}")
            
            try:
                kb = extract_knowledge_from_file(filepath, company=COMPANY_NAME)
                kb.source_file = filename
                all_kbs.append(kb)
                
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                base = os.path.splitext(filename)[0].replace(" ", "_")
                json_path = os.path.join(OUTPUT_DIR, f"{base}_{ts}.json")
                kb.save_json(json_path)
                
                print("    [OK] Extracted %d entries" % len(kb.entries))
                processed_count += 1
                
            except Exception as e:
                print("    [FAIL] Failed: %s" % str(e)[:100])
    
    print("\n=== Summary ===")
    print("Total files processed: %d" % processed_count)
    
    total_entries = sum(len(kb.entries) for kb in all_kbs)
    print("Total entries extracted: %d" % total_entries)
    
    combined_kb = KnowledgeBase(source_file="batch_combined")
    for kb in all_kbs:
        for entry in kb.entries:
            combined_kb.add(entry)
    
    combined_path = os.path.join(OUTPUT_DIR, "batch_combined_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    combined_kb.save_json(combined_path)
    print("Combined JSON saved to: %s" % combined_path)
    
    return combined_kb

if __name__ == "__main__":
    main()