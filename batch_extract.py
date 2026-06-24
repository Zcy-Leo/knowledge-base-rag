import os
import json
import time
import sys
from unified_extractor import extract_knowledge_from_file

SOURCE_DIR = r"e:\code\c\Knowledge Base Automation\organized_documents"
OUTPUT_DIR = r"e:\code\c\Knowledge Base Automation\分析文档"

EXCLUDE_FOLDERS = {"PDF_Documents"}
EXCLUDE_EXTENSIONS = {'.pdf', '.wav', '.zip', '.py', '.go', '.p7s', '.xsl'}

SUPPORTED_EXTENSIONS = {
    '.doc', '.docx', '.odt', '.rtf', '.txt', '.md', '.rst', '.org',
    '.html', '.htm', '.xml', '.json', '.ndjson',
    '.csv', '.xls', '.xlsx', '.tsv', '.ppt', '.pptx',
    '.eml', '.msg', '.epub', '.yaml', '.yml',
    '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.heic'
}

def process_files(files_to_process=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_files = 0
    success_count = 0
    fail_count = 0
    total_entries = 0
    
    if files_to_process:
        all_files = [(os.path.join(SOURCE_DIR, f), f) for f in files_to_process]
    else:
        all_files = []
        for root, dirs, files in os.walk(SOURCE_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_FOLDERS]
            for filename in files:
                _, ext = os.path.splitext(filename.lower())
                if ext in EXCLUDE_EXTENSIONS or ext not in SUPPORTED_EXTENSIONS:
                    continue
                file_path = os.path.join(root, filename)
                all_files.append((file_path, filename))
    
    for file_path, filename in all_files:
        total_files += 1
        
        print("\n" + "="*60)
        print("  [%d] Processing: %s" % (total_files, filename))
        print("="*60)
        
        try:
            kb = extract_knowledge_from_file(file_path)
            
            if kb and kb.entries:
                base_name = os.path.splitext(filename)[0]
                output_name = "knowledge_%s.json" % base_name
                output_path = os.path.join(OUTPUT_DIR, output_name)
                
                kb.save_json(output_path)
                success_count += 1
                total_entries += len(kb.entries)
                print("  OK: %d entries extracted" % len(kb.entries))
            else:
                print("  No entries extracted (file may be empty or unsupported)")
                fail_count += 1
        
        except Exception as e:
            print("  FAILED: %s" % str(e))
            fail_count += 1
    
    print("\n" + "="*60)
    print("  Summary:")
    print("    Total files processed: %d" % total_files)
    print("    Successfully extracted: %d" % success_count)
    print("    Failed: %d" % fail_count)
    print("    Total entries created: %d" % total_entries)
    print("    Output directory: %s" % OUTPUT_DIR)
    print("="*60)

if __name__ == "__main__":
    start_time = time.time()
    print("Starting batch extraction at %s" % time.strftime('%Y-%m-%d %H:%M:%S'))
    print("Source: %s" % SOURCE_DIR)
    print("Output: %s\n" % OUTPUT_DIR)
    
    process_files()
    
    elapsed_time = time.time() - start_time
    print("\nDone! Total time: %.2f seconds" % elapsed_time)