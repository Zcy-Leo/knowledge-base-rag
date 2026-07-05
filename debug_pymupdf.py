import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz

pdf_path = "organized_documents/PDF_Documents/single_table.pdf"

print(f"Testing PyMuPDF find_tables() on: {pdf_path}")
print(f"File exists: {os.path.exists(pdf_path)}")

doc = fitz.open(pdf_path)
print(f"Number of pages: {len(doc)}")

for page_idx, page in enumerate(doc):
    print(f"\n=== Page {page_idx + 1} ===")
    
    tables = page.find_tables()
    print(f"Tables found: {len(tables.tables)}")
    
    for i, table in enumerate(tables.tables):
        print(f"\nTable {i + 1}:")
        print(f"  BBox: {table.bbox}")
        print(f"  Number of cells: {table.num_cells}")
        print(f"  Number of rows: {table.num_rows}")
        print(f"  Number of cols: {table.num_cols}")
        
        rows = table.extract()
        print(f"  Extracted rows:")
        for row in rows[:5]:
            print(f"    {row}")
        if len(rows) > 5:
            print(f"    ... ({len(rows) - 5} more rows)")

doc.close()

print("\n" + "="*60)
print("Now testing get_text() for comparison")
print("="*60)

doc = fitz.open(pdf_path)
for page in doc:
    text = page.get_text()
    print(f"Text length: {len(text)} chars")
    print(f"First 200 chars:\n{text[:200]}")
doc.close()
