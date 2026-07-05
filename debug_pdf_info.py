import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz

pdf_path = "organized_documents/PDF_Documents/single_table.pdf"

doc = fitz.open(pdf_path)

print(f"PDF Info:")
print(f"  Pages: {doc.page_count}")
print(f"  Encrypted: {doc.needs_pass}")
print(f"  Metadata: {doc.metadata}")
print(f"  Page mode: {doc.pagemode}")

for page_idx, page in enumerate(doc):
    print(f"\nPage {page_idx + 1}:")
    print(f"  Rotation: {page.rotation}")
    print(f"  Rect: {page.rect}")
    print(f"  Has text: {len(page.get_text()) > 0}")
    
    text_blocks = page.get_text("blocks")
    print(f"  Text blocks: {len(text_blocks)}")
    
    if text_blocks:
        for b in text_blocks[:3]:
            print(f"    Block: {b[:5]}")
    
    images = page.get_images(full=True)
    print(f"  Images: {len(images)}")
    if images:
        for img in images[:3]:
            print(f"    Image: {img[:5]}")

doc.close()

print("\nChecking if this is an image-only PDF...")
doc = fitz.open(pdf_path)
total_images = 0
total_text = 0
for page in doc:
    total_images += len(page.get_images(full=True))
    total_text += len(page.get_text())

doc.close()

print(f"Total images: {total_images}")
print(f"Total text: {total_text} chars")

if total_images > 0 and total_text == 0:
    print("\n⚠️ This appears to be an image-only (scanned) PDF!")
    print("You need OCR to extract text from this PDF.")
