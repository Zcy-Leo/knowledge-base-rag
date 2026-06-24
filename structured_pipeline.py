"""
structured_pipeline.py
============================
Complete knowledge base construction pipeline:
  PDF -> marker parse (local) -> structured KnowledgeEntry (JSON Schema)
  -> vectorize (bge local) -> store in Chroma + export JSON file

Usage:
  python structured_pipeline.py [pdf_file_path]
"""

import os
import sys
import json
import shutil
from datetime import datetime

from knowledge_schema import KnowledgeBase, KnowledgeEntry
from marker_extractor import extract_knowledge_from_pdf

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Configuration
DEFAULT_PDF      = "sample_manual.pdf"
DB_DIRECTORY     = "./my_local_database"
OUTPUT_JSON_DIR  = "./knowledge_json_output"
EMBED_MODEL      = "BAAI/bge-small-en-v1.5"


def step1_extract(pdf_path: str) -> KnowledgeBase:
    """Step 1: PDF -> structured KnowledgeBase"""
    print("\n" + "="*55)
    print("  [Step 1/4] Parse PDF -> structured knowledge entries")
    print("="*55)
    kb = extract_knowledge_from_pdf(pdf_path)
    return kb


def step2_export_json(kb: KnowledgeBase, pdf_path: str) -> str:
    """Step 2: Export structured JSON file"""
    print("\n" + "="*55)
    print("  [Step 2/4] Export structured JSON file")
    print("="*55)

    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    json_path = os.path.join(OUTPUT_JSON_DIR, f"{base_name}_{timestamp}.json")

    kb.save_json(json_path)
    print(f"  JSON saved: {json_path}")
    return json_path


def step3_embed_and_store(kb: KnowledgeBase) -> Chroma:
    """Step 3: Vectorize + store in Chroma vector database"""
    print("\n" + "="*55)
    print("  [Step 3/4] Vectorize -> store in Chroma database")
    print("="*55)

    if not kb.entries:
        print("  [WARN] Knowledge base is empty, skipping vectorization")
        return None

    # Clear old database to avoid duplicate entries
    if os.path.exists(DB_DIRECTORY):
        print(f"  Clearing old database at: {DB_DIRECTORY}")
        shutil.rmtree(DB_DIRECTORY)

    texts = []
    metadatas = []
    for entry in kb.entries:
        texts.append(entry.to_chroma_text())
        metadatas.append(entry.to_chroma_metadata())

    print(f"  {len(texts)} entries to vectorize...")
    print(f"  Loading local embedding model: {EMBED_MODEL}")
    print("  (First run will download the model, please wait...)")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Batch processing for large datasets
    BATCH_SIZE = 500
    if len(texts) <= BATCH_SIZE:
        print(f"  Writing to Chroma: {DB_DIRECTORY}")
        db = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=DB_DIRECTORY
        )
    else:
        print(f"  Large dataset ({len(texts)} entries), processing in batches of {BATCH_SIZE}...")
        db = None
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_metas = metadatas[i:i+BATCH_SIZE]
            if db is None:
                db = Chroma.from_texts(
                    texts=batch_texts,
                    embedding=embeddings,
                    metadatas=batch_metas,
                    persist_directory=DB_DIRECTORY
                )
            else:
                db.add_texts(texts=batch_texts, metadatas=batch_metas)
            print(f"    Batch {i//BATCH_SIZE + 1}: {min(i+BATCH_SIZE, len(texts))}/{len(texts)} entries stored")

    print(f"  Vectorization complete. {len(texts)} entries stored.")
    return db


def step4_verify(db: Chroma):
    """Step 4: Verify database with a test query"""
    print("\n" + "="*55)
    print("  [Step 4/4] Verify database")
    print("="*55)

    if db is None:
        print("  [WARN] Database is empty, skipping verification")
        return

    test_query = "How to reset the device?"
    print(f"  Test query: '{test_query}'")
    results = db.similarity_search(test_query, k=2)

    if results:
        print(f"  Search OK. {len(results)} results returned:")
        for i, res in enumerate(results):
            meta = res.metadata
            print(f"    [{i+1}] [{meta.get('type','?')}] {meta.get('title','?')[:60]}")
            print(f"         Content: {res.page_content[:100]}...")
    else:
        print("  [WARN] No results returned, check database content")


def run_pipeline(pdf_path: str) -> dict:
    """
    Run the complete knowledge base construction pipeline.

    Returns:
        result dict with: kb, json_path, db, stats
    """
    print("\n" + "="*50)
    print("  Enterprise Knowledge Base Pipeline v2.0")
    print("  (All processing is local, no data leaves this machine)")
    print("="*50)

    start_time = datetime.now()

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"[ERROR] PDF not found: {pdf_path}")

    kb        = step1_extract(pdf_path)
    json_path = step2_export_json(kb, pdf_path)
    db        = step3_embed_and_store(kb)
    step4_verify(db)

    elapsed = (datetime.now() - start_time).total_seconds()
    type_counts = {}
    for e in kb.entries:
        type_counts[e.type] = type_counts.get(e.type, 0) + 1

    stats = {
        "total_entries": len(kb.entries),
        "type_distribution": type_counts,
        "json_output": json_path,
        "db_directory": DB_DIRECTORY,
        "elapsed_seconds": round(elapsed, 1)
    }

    print("\n" + "="*50)
    print("  Pipeline complete.")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Total entries: {len(kb.entries)}")
    print(f"  Type distribution: {type_counts}")
    print(f"  JSON output: {json_path}")
    print(f"  Vector DB: {DB_DIRECTORY}")
    print("="*50 + "\n")

    return {"kb": kb, "json_path": json_path, "db": db, "stats": stats}


if __name__ == "__main__":
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    result = run_pipeline(pdf_file)
