"""Test unified extractor with different file formats"""
import os
import tempfile

# Create test files
test_files = []

# Test 1: Simple text file
txt_content = """
User Manual
===========

This is the user manual for the HP Printer.

1. Getting Started
------------------
To set up your printer, follow these steps:
- Unpack the printer
- Connect power cable
- Install ink cartridges

2. Troubleshooting
------------------
If you encounter issues:
- Check power connection
- Verify ink levels
- Restart the device
"""
txt_path = os.path.join(tempfile.mkdtemp(), "test_manual.txt")
with open(txt_path, 'w') as f:
    f.write(txt_content)
test_files.append(("Text", txt_path))

# Test 2: Markdown file  
md_content = """
# Product Guide

## Features

- High quality printing
- Wireless connectivity
- Energy efficient

## Specifications

| Feature | Value |
|---------|-------|
| Resolution | 1200 dpi |
| Speed | 20 ppm |
| Memory | 256 MB |
"""
md_path = os.path.join(tempfile.mkdtemp(), "test_guide.md")
with open(md_path, 'w') as f:
    f.write(md_content)
test_files.append(("Markdown", md_path))

print("Testing unified extractor...\n")

from unified_extractor import extract_knowledge_from_file

for file_type, file_path in test_files:
    print(f"=== Testing {file_type} file ===")
    print(f"File: {file_path}")
    
    try:
        kb = extract_knowledge_from_file(file_path, company="TestCompany")
        print(f"Success! Extracted {len(kb.entries)} entries")
        
        for i, entry in enumerate(kb.entries[:3]):
            print(f"\n  Entry {i+1}:")
            print(f"    Type: {entry.type}")
            print(f"    Title: {entry.title[:60]}")
            print(f"    Keywords: {entry.keywords}")
            print(f"    Company: {entry.metadata.get('company', '')}")
        
        print(f"\n  Total entries: {len(kb.entries)}")
        
    except Exception as e:
        print(f"Failed: {e}")
    
    print("-" * 50)

print("\n✅ All tests completed!")