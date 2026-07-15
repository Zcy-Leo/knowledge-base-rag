import os
import sys
import tempfile
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

print("=" * 60)
print("Testing Incremental Update Display")
print("=" * 60)

def create_test_pdf(content, filename):
    from reportlab.pdfgen import canvas
    pdf_path = os.path.join(tempfile.gettempdir(), filename)
    c = canvas.Canvas(pdf_path)
    lines = content.split('\n')
    y = 750
    for line in lines:
        c.drawString(50, y, line)
        y -= 20
        if y < 50:
            c.showPage()
            y = 750
    c.save()
    return pdf_path

print("\n1. Testing with NEW document (first upload)...")
test_content_v1 = """HP Printer Maintenance Guide

Chapter 1: Introduction
HP printers require regular maintenance to ensure optimal performance.

Chapter 2: Cleaning Procedures
Clean the print heads every 30 days.

Chapter 3: Troubleshooting
Common issues include paper jams and ink cartridge errors."""

print("\n2. Testing with MODIFIED document (second upload)...")
test_content_v2 = """HP Printer Maintenance Guide

Chapter 1: Introduction
HP printers require regular maintenance to ensure optimal performance.

Chapter 2: Cleaning Procedures
Clean the print heads every 60 days.

Chapter 3: Troubleshooting
Common issues include paper jams and ink cartridge errors.

Chapter 4: Advanced Settings
Configure duplex printing for better efficiency."""

print("\n3. Simulating the display logic...")
print("-" * 40)

from langchain_huggingface import HuggingFaceEmbeddings
from incremental_ingestor import IncrementalIngestor

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"})
ingestor = IncrementalIngestor(embeddings)

print("\n📤 Uploading Version 1 (first time)...")
result1 = ingestor.ingest_document(test_content_v1, "test_display.pdf")
print(f"Result: {result1}")

if result1:
    print("\n📊 LiveVectorLake Incremental Update")
    print(f"   New: {result1.get('new', 0)}")
    print(f"   Modified: {result1.get('modified', 0)}")
    print(f"   Deleted: {result1.get('deleted', 0)}")
    print(f"   Unchanged: {result1.get('unchanged', 0)}")
    if "embeddings_computed" in result1:
        computed = result1.get("embeddings_computed", 0)
        reused = result1.get("embeddings_reused", 0)
        total = computed + reused
        if total > 0:
            savings = ((total - computed) / total) * 100
            print(f"✅ Embedding savings: {savings:.1f}% ({reused}/{total} chunks reused)")
    if "version_number" in result1:
        print(f"📝 Document version: {result1['version_number']}")

print("\n" + "-" * 40)
print("\n📤 Uploading Version 2 (modified, second time)...")
result2 = ingestor.ingest_document(test_content_v2, "test_display.pdf")
print(f"Result: {result2}")

if result2:
    print("\n📊 LiveVectorLake Incremental Update")
    print(f"   New: {result2.get('new', 0)}")
    print(f"   Modified: {result2.get('modified', 0)}")
    print(f"   Deleted: {result2.get('deleted', 0)}")
    print(f"   Unchanged: {result2.get('unchanged', 0)}")
    if "embeddings_computed" in result2:
        computed = result2.get("embeddings_computed", 0)
        reused = result2.get("embeddings_reused", 0)
        total = computed + reused
        if total > 0:
            savings = ((total - computed) / total) * 100
            print(f"✅ Embedding savings: {savings:.1f}% ({reused}/{total} chunks reused)")
    if "version_number" in result2:
        print(f"📝 Document version: {result2['version_number']}")

print("\n" + "=" * 60)
print("Display Test Complete!")
print("=" * 60)
print("\n💡 How to see this in the UI:")
print("   1. Open http://localhost:8502")
print("   2. Go to 'Ingest' page")
print("   3. Upload a PDF document")
print("   4. Modify the document (change some text)")
print("   5. Upload the modified document again")
print("   6. You should see the 'LiveVectorLake Incremental Update' panel!")