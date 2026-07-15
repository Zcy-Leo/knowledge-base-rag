"""
app_v2.py -- Knowledge Base Automation System
================================================================
Features:
  1. Ingest from 3 sources: PDF (marker), pre-parsed JSON, Website URL
  2. Vectorize with bge-small-en-v1.5 and store in Chroma
  3. Database overview: view all ingested sources and stats
  4. Browse structured entries (filter by type/source/company)
  5. Semantic search with source attribution and company filter
  6. Company management: add/remove companies dynamically

Architecture: All embedding and storage runs locally (CPU).
             PDF parsing uses marker (GPU server or local).
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)
print(f"[DEBUG] Loaded .env from: {env_path}")
print(f"[DEBUG] LLM_PROVIDER: {os.environ.get('LLM_PROVIDER', 'NOT FOUND')}")
print(f"[DEBUG] DEEPSEEK_API_KEY: {os.environ.get('DEEPSEEK_API_KEY', 'NOT FOUND')[:20]}...")

import sys

def debug_print(msg):
    sys.stderr.write(f"[DEBUG] {msg}\n")
    sys.stderr.flush()

import os
import json
import shutil
import tempfile
import uuid
from datetime import datetime
import streamlit as st

debug_print("App starting...")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIRECTORY = os.path.join(BASE_DIR, "my_local_database")
DB_PATH = os.path.join(DB_DIRECTORY, "chroma.sqlite3")
JSON_OUTPUT_DIR = os.path.join(BASE_DIR, "knowledge_json_output")

from db_reader import DBReader


_embedding_model_cache = None

def get_embedding_model():
    global _embedding_model_cache
    if _embedding_model_cache is None:
        print("[INFO get_embedding_model] Creating new embedding model instance...")
        _embedding_model_cache = load_embeddings()
        print("[INFO get_embedding_model] Embedding model created successfully")
    else:
        print("[INFO get_embedding_model] Reusing cached embedding model")
    return _embedding_model_cache


def show_incremental_results(incremental_result, kb=None, count=0):
    if not incremental_result:
        return
    
    st.markdown("---")
    st.subheader("📊 LiveVectorLake Incremental Update")
    
    col_inc1, col_inc2, col_inc3, col_inc4 = st.columns(4)
    col_inc1.metric("New Entries", incremental_result.get("new", 0))
    col_inc2.metric("Modified", incremental_result.get("modified", 0))
    col_inc3.metric("Deleted", incremental_result.get("deleted", 0))
    col_inc4.metric("Unchanged", incremental_result.get("unchanged", 0))
    
    if "embeddings_computed" in incremental_result:
        computed = incremental_result.get("embeddings_computed", 0)
        reused = incremental_result.get("embeddings_reused", 0)
        total = computed + reused
        if total > 0:
            savings = ((total - computed) / total) * 100
            st.success(f"✅ Embedding savings: {savings:.1f}% ({reused}/{total} entries reused)")
    
    if "version_number" in incremental_result:
        st.info(f"📝 Document version: {incremental_result['version_number']}")
    
    if kb:
        tc = {}
        for e in kb.entries:
            tc[e.type] = tc.get(e.type, 0) + 1
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Entries", count)
        col2.metric("SOP Steps", tc.get("sop_step", 0))
        col3.metric("FAQ", tc.get("faq", 0))


@st.cache_resource(show_spinner="Initializing search engine...", ttl=0)
def init_faiss_search():
    print("[DEBUG init_faiss_search] Called (caching enabled)")
    try:
        import sys
        print("[DEBUG init_faiss_search] Python version:", sys.version)
        
        from faiss_search import FAISSSearch
        print("[DEBUG init_faiss_search] Imported FAISSSearch")
        
        print("[DEBUG init_faiss_search] Getting embedding model...")
        embeddings = get_embedding_model()
        print("[DEBUG init_faiss_search] Got embedding model:", embeddings)
        
        print("[DEBUG init_faiss_search] Creating FAISSSearch instance...")
        searcher = FAISSSearch(embedding_model=embeddings)
        print("[DEBUG init_faiss_search] Created FAISSSearch instance:", searcher)
        
        print("[DEBUG init_faiss_search] Initializing FAISSSearch...")
        searcher.initialize()
        print("[DEBUG init_faiss_search] FAISSSearch initialized, docs:", len(searcher.doc_ids))
        
        return searcher
    except Exception as e:
        print(f"[DEBUG init_faiss_search] Failed to initialize FAISSSearch: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


@st.cache_resource(show_spinner="Initializing incremental ingestor...", ttl=0)
def init_incremental_ingestor():
    try:
        from incremental_ingestor import IncrementalIngestor
        ingestor = IncrementalIngestor()
        return ingestor
    except Exception as e:
        print(f"Failed to initialize IncrementalIngestor: {e}")
        return None


def get_reranker_models_cache():
    if "reranker_models" not in st.session_state:
        try:
            from reranker import get_available_models
            models = get_available_models(language="english")
            st.session_state.reranker_models = [m for m in models if m["type"] == "crossencoder"]
        except Exception as e:
            print(f"Failed to load reranker models: {e}")
            st.session_state.reranker_models = []
    return st.session_state.reranker_models
KNOWN_COMPANIES = [
    "HP", "Hewlett Packard", "Cisco", "Reliance", "Google", "Microsoft", "Apple", 
    "Samsung", "HSBC", "IBM", "Intel", "NVIDIA", "AMD", "Dell", "Lenovo", 
    "Oracle", "SAP", "Salesforce", "Adobe", "VMware", "McAfee", "Symantec",
    "Juniper", "Aruba", "Palo Alto", "Fortinet", "Check Point", "AWS", "Azure",
    "Tesla", "Toyota", "Ford", "General Motors", "Volkswagen", "BMW", "Mercedes",
    "JPMorgan", "Goldman Sachs", "Morgan Stanley", "Bank of America", "Citigroup",
    "Walmart", "Target", "Amazon", "Costco", "Home Depot", "Nestle", "Coca-Cola",
    "PepsiCo", "Unilever", "Procter Gamble", "Johnson Johnson", "Pfizer", "Merck",
    "Netflix", "Disney", "Warner", "Sony", "Meta", "Twitter", "LinkedIn", "Uber",
    "Airbnb", "Spotify", "Slack", "Zoom", "Atlassian", "Shopify", "Square"
]

# --- Topic Auto-Detection ---
KNOWN_TOPICS = {
    "TechDocs": ["manual", "guide", "user guide", "reference", "documentation", "techdoc", "technical"],
    "Books": ["book", "textbook", "paper", "article", "journal"],
    "Academic": ["academic", "research", "thesis", "dissertation", "conference", "paper", "arxiv", "preprint"],
    "Government": ["government", "policy", "regulation", "law", "legislation", "bill", "official"],
    "Financial": ["financial", "bank", "investment", "stock", "market", "fund", "account", "finance"],
    "Office": ["office", "meeting", "email", "memo", "report", "presentation", "slide"],
    "Email": ["email", "mail", "message", "correspondence"],
    "Images": ["image", "photo", "picture", "scan", "graphic"],
    "Data": ["data", "database", "spreadsheet", "table", "csv", "excel"],
    "Demo": ["demo", "example", "sample", "tutorial"],
    "Other": []
}

def detect_company_from_filename(filename):
    """Detect company from filename using keyword matching."""
    filename_lower = filename.lower()
    for company in KNOWN_COMPANIES:
        if company.lower() in filename_lower:
            return company
    return None

def detect_topic_from_filename(filename):
    """Detect topic from filename using keyword matching."""
    filename_lower = filename.lower()
    for topic, keywords in KNOWN_TOPICS.items():
        for kw in keywords:
            if kw in filename_lower:
                return topic
    return None

def detect_company_from_content(content, filename):
    """Detect company from content using keyword matching."""
    # First try filename
    detected = detect_company_from_filename(filename)
    if detected:
        return detected
    
    # Then try content (first 1000 chars)
    if content:
        preview = content[:1000].lower()
        for company in KNOWN_COMPANIES:
            if company.lower() in preview:
                return company
    
    return None

def detect_company_with_llm(content, filename):
    """Use LLM to detect company from content. Returns company name or None."""
    try:
        from llm_provider import call_llm
        
        prompt = f"""Analyze this document and identify the company or organization it belongs to.
        
Filename: {filename}

Content preview (first 500 chars):
{content[:500] if content else '(empty)'}

Respond with ONLY the company name, or "Unknown" if you cannot determine the company.
Do not include any explanation, just the company name."""

        company = call_llm(prompt, max_tokens=64, temperature=0.0)
        
        if company and company.lower() != "unknown":
            return company.strip()
        return None
        
    except Exception as e:
        print(f"LLM company detection failed: {e}")
        return None

def get_all_companies(db=None):
    """Get all companies from all sources (company_manager + database)."""
    default_companies = ["NA", "HP", "Generic"]
    
    stored_companies = []
    try:
        from company_manager import get_companies
        stored_companies = get_companies()
    except:
        pass
    
    db_companies = []
    try:
        reader = DBReader(DB_PATH)
        all_metas = reader.get_all_metadatas()
        for doc_id, meta in all_metas.items():
            company_name = str(meta.get("company", "")).strip()
            if not company_name:
                company_name = "NA"
            db_companies.append(company_name)
    except:
        pass
    
    return sorted(list(set(default_companies + stored_companies + db_companies)))

def get_metadata_counts(db=None):
    """Get counts of entries grouped by company and topic."""
    company_counts = {}
    topic_counts = {}
    
    try:
        reader = DBReader(DB_PATH)
        all_metas = reader.get_all_metadatas()
        for doc_id, m in all_metas.items():
            if m and isinstance(m, dict):
                company_name = str(m.get("company", "")).strip()
                topic_name = str(m.get("topic", "")).strip()
                
                if not company_name:
                    company_name = "NA"
                if not topic_name:
                    topic_name = "NA"
                
                company_counts[company_name] = company_counts.get(company_name, 0) + 1
                topic_counts[topic_name] = topic_counts.get(topic_name, 0) + 1
    except:
        pass
    
    return company_counts, topic_counts

def get_all_topics(db=None):
    """Get all topics from database."""
    default_topics = ["TechDocs", "Books", "Academic", "Government", "Financial", "Office", "Email", "Images", "Data", "Demo", "Other"]
    
    db_topics = []
    try:
        reader = DBReader(DB_PATH)
        all_metas = reader.get_all_metadatas()
        for doc_id, m in all_metas.items():
            if m and isinstance(m, dict):
                topic_name = str(m.get("topic", "")).strip()
                if not topic_name:
                    topic_name = "NA"
                db_topics.append(topic_name)
    except:
        pass
    
    return sorted(list(set(default_topics + db_topics)))

# --- LLM Classification Support ---
def run_llm_classification_on_db(company_filter="All Companies", topic_filter="All Topics"):
    """Run LLM classification on entries in the database.
    
    Args:
        company_filter: Company name to filter by, or "All Companies" for all entries.
        topic_filter: Topic name to filter by, or "All Topics" for all entries.
    """
    from knowledge_schema import KnowledgeBase, KnowledgeEntry
    
    # Load all entries from DB using direct SQLite access
    reader = DBReader(DB_PATH)
    all_docs = reader.get_documents_with_metadatas()
    
    # Filter by company and topic if specified
    filtered_docs = []
    filtered_metas = []
    filtered_ids = []
    skipped_count = 0
    
    for doc_id, data in all_docs.items():
        doc = data["content"]
        meta = data["metadata"]
        
        if meta is None:
            meta = {}
        
        company_name = str(meta.get("company", "")).strip()
        topic_name = str(meta.get("topic", "")).strip()
        
        already_classified = bool(meta.get("llm_type"))
        
        company_match = (company_filter == "All Companies" or company_name == company_filter)
        topic_match = (topic_filter == "All Topics" or topic_name == topic_filter)
        
        if company_match and topic_match:
            if already_classified:
                skipped_count += 1
            else:
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_ids.append(doc_id)
    
    if not filtered_docs:
        filter_desc = []
        if company_filter != "All Companies":
            filter_desc.append(f"company: {company_filter}")
        if topic_filter != "All Topics":
            filter_desc.append(f"topic: {topic_filter}")
        yield ("info", 0, f"No unclassified entries found for {', '.join(filter_desc)}. {skipped_count} entries already classified.", skipped_count)
        return
    
    yield ("info", len(filtered_docs), f"Found {len(filtered_docs)} unclassified entries. Skipping {skipped_count} already classified entries.", skipped_count)
    
    # Convert to KnowledgeBase
    kb = KnowledgeBase(source_file="database")
    for i, (doc, meta) in enumerate(zip(filtered_docs, filtered_metas)):
        entry = KnowledgeEntry(
            id=filtered_ids[i],
            type=meta.get("type", "general"),
            title=meta.get("title", ""),
            content=doc,
            source_file=meta.get("source_file", ""),
            source_page=meta.get("source_page", 0),
            keywords=meta.get("keywords", []),
            metadata=meta
        )
        kb.add(entry)
    
    # Run LLM classification (batch mode for speed)
    try:
        from llm_classify import classify_batch_with_llm
        
        batch_size = 5
        total_entries = len(kb.entries)
        processed_count = 0
        
        for batch_start in range(0, total_entries, batch_size):
            batch_end = min(batch_start + batch_size, total_entries)
            batch_entries = kb.entries[batch_start:batch_end]
            
            batch_input = [{
                'title': e.title,
                'content': e.content
            } for e in batch_entries]
            
            results = classify_batch_with_llm(batch_input)
            
            for i, entry in enumerate(batch_entries):
                if i < len(results):
                    result = results[i]
                    entry.type = result.get('type', entry.type)
                    entry.metadata['llm_type'] = result.get('type', '')
                    entry.metadata['llm_keywords'] = result.get('keywords', [])
                    entry.metadata['llm_confidence'] = result.get('confidence', 0)
                
                processed_count += 1
                yield (processed_count, total_entries, entry.title, result)
        
        # Save to JSON
        os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        company_suffix = f"_{company_filter.replace(' ', '_')}" if company_filter != "All Companies" else ""
        json_path = os.path.join(JSON_OUTPUT_DIR, f"knowledge_llm{company_suffix}_{ts}.json")
        kb.save_json(json_path)
        
        # Update database with LLM classification results
        from db_writer import DBWriter
        from faiss_search import get_embeddings_model
        
        updated_metas = []
        texts_to_update = []
        for entry in kb.entries:
            meta = {
                "id": entry.id,
                "title": entry.title,
                "type": entry.type,
                "source_file": entry.source_file,
                "source_page": entry.source_page,
                "keywords": ", ".join(entry.keywords) if entry.keywords else "",
                "company": entry.metadata.get("company", ""),
                "topic": entry.metadata.get("topic", ""),
                "llm_type": entry.metadata.get("llm_type", ""),
                "llm_keywords": ", ".join(entry.metadata.get("llm_keywords", [])) if entry.metadata.get("llm_keywords") else "",
                "llm_confidence": entry.metadata.get("llm_confidence", 0),
            }
            updated_metas.append(meta)
            texts_to_update.append(entry.to_chroma_text())
        
        ids_to_update = [e.id for e in kb.entries]
        
        db_writer = DBWriter(DB_PATH)
        deleted = db_writer.delete_documents_by_entry_id(ids_to_update)
        yield ("info", deleted, f"Deleted {deleted} documents for update", None)
        
        embeddings_model = get_embeddings_model()
        added = db_writer.add_documents(texts_to_update, updated_metas, embeddings_model)
        yield ("info", added, f"Added {added} documents with LLM classification", None)
        
        yield ("done", len(kb.entries), json_path, skipped_count)
        
    except Exception as ex:
        yield ("error", 0, str(ex), None)


st.set_page_config(
    page_title="Knowledge Base System",
    page_icon="KB",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Shared utilities ---

# 使用绝对路径确保数据库始终指向正确位置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIRECTORY = os.path.join(BASE_DIR, "my_local_database")
JSON_OUTPUT_DIR = os.path.join(BASE_DIR, "knowledge_json_output")


def load_embeddings():
    import torch
    
    try:
        torch.cuda.empty_cache()
    except:
        pass
    
    torch.set_default_device(torch.device('cpu'))
    
    os.environ["HF_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    from transformers import AutoTokenizer, AutoModel
    import numpy as np
    
    model_path = os.path.join(os.environ.get("USERPROFILE", "~"), ".cache", "huggingface", "hub", "models--BAAI--bge-small-en-v1.5", "snapshots")
    
    if os.path.exists(model_path):
        import glob
        snapshots = glob.glob(os.path.join(model_path, "*"))
        if snapshots:
            model_path = snapshots[0]
        else:
            model_path = "BAAI/bge-small-en-v1.5"
    else:
        model_path = "BAAI/bge-small-en-v1.5"
    
    class CustomEmbeddings:
        def __init__(self, model_path):
            print(f"[INFO] Loading tokenizer from {model_path} (offline mode)...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            print(f"[INFO] Loading model config from {model_path}...")
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            
            print("[INFO] Creating model from config (empty weights)...")
            self.model = AutoModel.from_config(config)
            
            print("[INFO] Manually loading weights from safetensors...")
            from safetensors.torch import load_file
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path, device="cpu")
                state_dict.pop("embeddings.position_ids", None)
                self.model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(f"model.safetensors not found at {safetensors_path}")
            
            self.model.eval()
            print("[INFO] Model loaded successfully")
        
        def _encode(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            
            all_embeddings = []
            device = torch.device("cpu")
            
            for text in texts:
                inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    all_embeddings.append(embeddings.detach().cpu().numpy()[0])
            
            if len(all_embeddings) == 1:
                return all_embeddings[0].tolist()
            return [e.tolist() for e in all_embeddings]
        
        def embed_documents(self, texts):
            return self._encode(texts)
        
        def embed_query(self, text):
            return self._encode(text)
    
    try:
        return CustomEmbeddings(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load embedding model from {model_path}: {e}")
        print("[ERROR] Please check if the model files are complete in the cache directory.")
        print(f"[ERROR] Expected path: {model_path}")
        raise


@st.cache_resource(show_spinner=False)
def get_db(embeddings):
    """Get or create Chroma database connection."""
    from langchain_chroma import Chroma
    if os.path.exists(DB_DIRECTORY):
        return Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)
    return None


@st.cache_resource(show_spinner=False)
def load_bm25_retriever():
    from bm25_retriever import get_bm25_retriever, sync_bm25_with_chroma
    from db_reader import DBReader
    persist_dir = os.path.join(BASE_DIR, "bm25_index")
    retriever = get_bm25_retriever(persist_dir)
    
    if retriever.get_index_size() == 0:
        print("[DEBUG] BM25 index is empty, performing initial sync from Chroma...")
        try:
            db_reader = DBReader(DB_PATH)
            retriever = sync_bm25_with_chroma(db_reader, persist_dir)
            print("[DEBUG] BM25 initial sync completed")
        except Exception as sync_ex:
            print(f"[DEBUG] BM25 initial sync failed: {sync_ex}")
    
    return retriever


def load_reranker():
    return None


def ingest_entries_to_db(entries, embeddings, source_label=""):
    """Vectorize entries and add to Chroma database using direct SQLite writes with incremental updates."""
    from db_writer import DBWriter

    texts = [e.to_chroma_text() for e in entries]
    metas = [e.to_chroma_metadata() for e in entries]
    
    entry_dicts = []
    for i, entry in enumerate(entries):
        entry_dicts.append({
            'id': entry.id,
            'text': texts[i],
            'metadata': metas[i]
        })

    incremental_result = None
    count = 0
    texts_to_write = []
    metas_to_write = []
    deleted_entries = []
    
    try:
        from incremental_ingestor import IncrementalIngestor
        ingestor = IncrementalIngestor()
        doc_id = source_label if source_label else str(hash("\n\n".join(texts)))
        
        print(f"[DEBUG] doc_id={doc_id}, entries={len(entry_dicts)}")
        
        incremental_result = ingestor.ingest_entries(entry_dicts, doc_id)
        print(f"Incremental update: {incremental_result}")
        
        new_entries = incremental_result.get("new_entries", [])
        modified_entries = incremental_result.get("modified_entries", [])
        deleted_entries = incremental_result.get("deleted_entries", [])
        
        entries_to_write = []
        
        for i, entry in enumerate(entries):
            if entry.id in new_entries or entry.id in modified_entries:
                entries_to_write.append(entry)
                texts_to_write.append(texts[i])
                metas_to_write.append(metas[i])
        
        if deleted_entries:
            print(f"[DEBUG] Deleting {len(deleted_entries)} entries from database")
            writer = DBWriter(DB_PATH)
            deleted_count = writer.delete_documents_by_entry_id(deleted_entries)
            print(f"[DEBUG] Successfully deleted {deleted_count} entries")
            count = -deleted_count
        
        if len(texts_to_write) == 0 and len(deleted_entries) == 0:
            print(f"[DEBUG] No changes to apply, skipping database write")
            return None, 0, incremental_result
        
        if texts_to_write:
            writer = DBWriter(DB_PATH)
            count = writer.add_documents(texts_to_write, metas_to_write, embeddings)
            print(f"Successfully ingested {count} documents")
    except Exception as inc_ex:
        print(f"Warning: Incremental detection failed, proceeding without it: {inc_ex}")
        incremental_result = None
        
        writer = DBWriter(DB_PATH)
        count = writer.add_documents(texts, metas, embeddings)
        print(f"Successfully ingested {count} documents")
    
    print(f"[DEBUG] Before load_bm25_retriever")
    try:
        bm25_retriever = load_bm25_retriever()
        print(f"[DEBUG] After load_bm25_retriever, retriever={bm25_retriever}")
        if bm25_retriever:
            if bm25_retriever.get_index_size() > 0:
                if len(texts_to_write) > 0:
                    bm25_retriever.add_documents(texts_to_write, metadata=metas_to_write)
                    print(f"[DEBUG] BM25 incremental update completed (added {len(texts_to_write)} documents)")
                else:
                    print(f"[DEBUG] No changes to BM25 index (only unchanged/deleted entries)")
            else:
                bm25_retriever.build_index(texts, metadata=metas)
                print(f"[DEBUG] BM25 initial index built")
        else:
            print(f"[DEBUG] Skipping BM25 update (retriever not available)")
    except Exception as bm25_ex:
        print(f"Warning: Failed to update BM25 index: {bm25_ex}")
    
    print(f"[DEBUG] ingest_entries_to_db returning count={count}")
    return None, count, incremental_result


def save_json_copy(kb, source_name):
    """Save a JSON backup of ingested entries."""
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(source_name)[0].replace(" ", "_")
    json_path = os.path.join(JSON_OUTPUT_DIR, f"{base}_{ts}.json")
    kb.save_json(json_path)
    return json_path


# --- Sidebar ---

with st.sidebar:
    st.title("KB Console")
    st.markdown("---")
    st.markdown("**Processing Mode:** Local CPU")
    st.markdown("**Embedding:** bge-small-en-v1.5")
    st.markdown("**Vector DB:** Chroma")
    st.markdown("**PDF Parser:** marker")
    st.markdown("**Other Docs:** unstructured")
    st.markdown("---")

    # Show database stats (no model load needed, read SQLite directly)
    if os.path.exists(DB_DIRECTORY):
        try:
            @st.cache_data(ttl=60, show_spinner=False)
            def get_db_count():
                reader = DBReader(DB_PATH)
                return reader.count_documents()
            
            count = get_db_count()
            st.metric("Total entries in DB", count)
        except Exception:
            st.caption("DB exists")
    else:
        st.info("Database empty")

    st.markdown("---")
    chunk_size = st.slider("Chunk size (chars)", 400, 1200, 800, 50)
    st.caption("marker + LangChain + Chroma")

    st.markdown("---")
    st.markdown("### 📊 LiveVectorLake Stats")
    try:
        from chunk_change_detector import ChunkChangeDetector
        detector = ChunkChangeDetector()
        conn = detector._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM chunk_versions WHERE status = 'active'")
        active_chunks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM chunk_versions")
        unique_docs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chunk_versions")
        total_versions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM doc_hash_store")
        tracked_docs = cursor.fetchone()[0]
        
        try:
            cursor.execute("SELECT COUNT(*) FROM doc_metadata")
            tracked_docs += cursor.fetchone()[0]
        except:
            pass
        
        conn.close()
        
        col_stat1, col_stat2 = st.columns(2)
        col_stat1.metric("Active Chunks", active_chunks)
        col_stat2.metric("Tracked Docs", tracked_docs)
        col_stat1.metric("Total Versions", total_versions)
        col_stat2.metric("Unique Docs", unique_docs)
        
        if tracked_docs > 0:
            with st.expander("📝 Recent Version History"):
                conn = detector._get_connection()
                cursor = conn.cursor()
                
                all_versions = []
                try:
                    cursor.execute('''
                        SELECT doc_id, version_number, last_updated 
                        FROM doc_hash_store 
                        ORDER BY last_updated DESC 
                        LIMIT 20
                    ''')
                    for row in cursor.fetchall():
                        all_versions.append((row[0], row[1], row[2], 'legacy'))
                except:
                    pass
                
                try:
                    cursor.execute('''
                        SELECT doc_id, last_version, last_updated 
                        FROM doc_metadata 
                        ORDER BY last_updated DESC 
                        LIMIT 20
                    ''')
                    for row in cursor.fetchall():
                        all_versions.append((row[0], row[1], row[2], 'new'))
                except:
                    pass
                
                latest_versions = {}
                for doc_id, version, timestamp, source in all_versions:
                    if doc_id not in latest_versions:
                        latest_versions[doc_id] = (version, timestamp)
                    elif timestamp > latest_versions[doc_id][1]:
                        latest_versions[doc_id] = (version, timestamp)
                
                sorted_versions = sorted(
                    latest_versions.items(), 
                    key=lambda x: x[1][1], 
                    reverse=True
                )[:5]
                
                for doc_id, (version, last_updated) in sorted_versions:
                    st.markdown(f"- `{doc_id[:30]}`: v{version} ({str(last_updated)[:19]})")
                
                conn.close()
                    
    except Exception as e:
        st.caption(f"LiveVectorLake not yet active")


# --- Main ---

st.title("Knowledge Base System")
st.markdown("Ingest documents from multiple sources, vectorize, and search.")
st.markdown("---")

# Initialize database connection for unified company/topic retrieval
# Using DBReader to avoid Chroma C extension compatibility issues
with st.spinner("Connecting to database..."):
    db = None
    try:
        reader = DBReader(DB_PATH)
        total = reader.count_documents()
        print(f"Database connected: {total} documents")
    except Exception as e:
        st.warning(f"Database connection warning: {e}")

tab1, tab2, tab3, tab4 = st.tabs(["Ingest", "Database", "Browse", "Search"])


# ========================================
# TAB 1: INGEST
# ========================================
with tab1:
    old_widget_keys = ['main_company_select', 'main_company_select_v2', 'main_topic_select', 'main_topic_select_v2']
    for key in old_widget_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    if st.button("🔄 Reset"):
        if 'json_uploads' in st.session_state:
            del st.session_state['json_uploads']
        if hasattr(st.session_state, 'ingest_success_message'):
            st.session_state.ingest_success_message = None
        st.rerun()
    
    if hasattr(st.session_state, 'ingest_success_message') and st.session_state.ingest_success_message:
        st.success(st.session_state.ingest_success_message)
        st.session_state.ingest_success_message = None
    
    st.subheader("Data Ingestion")

    ingest_mode = st.radio(
        "Select source type",
        ["Upload PDF (marker parse)", "Upload Document (unstructured)", "Upload JSON (pre-parsed)", "Website URL"],
        horizontal=True
    )

    col_meta1, col_meta2 = st.columns(2)
    
    company_counts, topic_counts = get_metadata_counts(db)
    
    with col_meta1:
        companies = get_all_companies(db if db else None)
        if not companies:
            companies = ["NA", "HP", "Generic"]
        
        company_options = ["--- Select Company ---"]
        for c in companies:
            count = company_counts.get(c, 0)
            company_options.append(f"{c} ({count})")
        company_options.append("+ Add New Company")
        
        if len(company_options) <= 2:
            company_options = ["--- Select Company ---", "NA (0)", "HP (0)", "Generic (0)", "+ Add New Company"]
        
        if hasattr(st.session_state, 'pending_company_select') and st.session_state.pending_company_select:
            new_company = st.session_state.pending_company_select
            count = company_counts.get(new_company, 0)
            target_value = f"{new_company} ({count})"
            if target_value in company_options:
                st.session_state.company_dropdown_2026_new = target_value
                st.session_state.pending_company_select = None
        
        selected_company_raw = st.selectbox("Select Company", company_options, key="company_dropdown_2026_new")
        
        if selected_company_raw == "--- Select Company ---":
            selected_company = selected_company_raw
        elif selected_company_raw == "+ Add New Company":
            selected_company = selected_company_raw
        else:
            selected_company = selected_company_raw.split(" (")[0]
    
    with col_meta2:
        topic_options_base = get_all_topics()
        if not topic_options_base:
            topic_options_base = ["TechDocs", "Books", "Academic", "Government", "Financial", "Office", "Email", "Images", "Data", "Demo", "Other"]
        
        topic_options = ["--- Select Topic ---"]
        for t in topic_options_base:
            count = topic_counts.get(t, 0)
            topic_options.append(f"{t} ({count})")
        
        if len(topic_options) <= 2:
            topic_options = ["--- Select Topic ---", "TechDocs (0)", "Books (0)", "Academic (0)", "Government (0)", "Financial (0)", "Office (0)", "Email (0)", "Images (0)", "Data (0)", "Demo (0)", "Other (0)"]
        
        selected_topic_raw = st.selectbox("Select Topic", topic_options, key="ingest_topic_v3")
        
        if selected_topic_raw == "--- Select Topic ---":
            selected_topic = selected_topic_raw
        else:
            selected_topic = selected_topic_raw.split(" (")[0]
    
    st.caption("*Select 'NA' for files without specific company ownership (books, emails, government forms, etc.)*")
    
    # Handle "Add New Company" option
    if selected_company == "+ Add New Company":
        new_company = st.text_input("Enter new company name", key="new_company_input")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Confirm Add", key="btn_confirm_add"):
                if new_company.strip():
                    try:
                        from company_manager import add_company
                        if add_company(new_company.strip()):
                            st.success(f"✅ Company '{new_company}' added successfully!")
                            st.session_state.companies_cache = []
                            st.session_state.pending_company_select = new_company.strip()
                        else:
                            st.warning("⚠️ Already exists or invalid")
                    except Exception as e:
                        st.error(f"Failed to add company: {e}")
                else:
                    st.warning("Please enter a company name")
        with col_btn2:
            if st.button("Cancel", key="btn_cancel_add"):
                st.session_state.pending_company_select = None
        selected_company = "--- Select Company ---"
    
    # Show success message from previous action (if any)
    if hasattr(st.session_state, 'company_add_success') and st.session_state.company_add_success:
        st.success(st.session_state.company_add_success)
        st.session_state.company_add_success = None  # Clear after showing

    st.markdown("---")

    # --- MODE 1: PDF Upload ---
    if ingest_mode == "Upload PDF (marker parse)":
        st.markdown("Upload a PDF file. It will be parsed using **marker** to extract structured knowledge entries.")

        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")

        if uploaded_pdf:
            import fitz
            tmp_dir = tempfile.mkdtemp()
            pdf_path = os.path.join(tmp_dir, uploaded_pdf.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.read())
            doc = fitz.Document(pdf_path)
            page_count = len(doc)
            doc.close()

            st.success(f"File: `{uploaded_pdf.name}` ({page_count} pages, {uploaded_pdf.size/1024:.1f} KB)")
            st.info(f"Company: **{selected_company}** | Topic: **{selected_topic}**")

            if selected_company == "--- Select Company ---":
                st.warning("⚠️ Please select a company before ingesting.")
            elif selected_topic == "--- Select Topic ---":
                st.warning("⚠️ Please select a topic before ingesting.")
            else:
                if st.button("Parse and Ingest", type="primary", key="btn_pdf"):
                    progress = st.progress(0, "Initializing...")

                    try:
                        # Step 1: Parse with marker
                        progress.progress(10, "Parsing with marker...")
                        from marker_extractor import parse_pdf_with_marker, markdown_to_knowledge_entries
                        from knowledge_schema import KnowledgeBase

                        md_text = parse_pdf_with_marker(pdf_path)

                        # Step 2: Structure
                        progress.progress(50, "Structuring entries...")
                        entries = markdown_to_knowledge_entries(md_text, uploaded_pdf.name, chunk_size=chunk_size)
                        kb = KnowledgeBase(source_file=uploaded_pdf.name)
                        
                        for e in entries:
                            e.metadata = {"company": selected_company, "topic": selected_topic}
                            kb.add(e)

                        # Step 3: Save JSON
                        progress.progress(70, "Saving JSON...")
                        json_path = save_json_copy(kb, uploaded_pdf.name)

                        # Step 4: Vectorize and store
                        progress.progress(80, "Vectorizing and storing...")
                        embeddings = get_embedding_model()
                        db, count, incremental_result = ingest_entries_to_db(kb.entries, embeddings, uploaded_pdf.name)

                        progress.progress(100, "Done")
                        st.success(f"Ingested {count} entries from `{uploaded_pdf.name}` into vector database.")

                        show_incremental_results(incremental_result, kb, count)

                        # Download JSON
                        with open(json_path, "r", encoding="utf-8") as f:
                            st.download_button("Download JSON", f.read(), os.path.basename(json_path), "application/json")

                    except Exception as ex:
                        progress.empty()
                        st.error(f"Error: {ex}")
                    st.exception(ex)

    # --- MODE 2: Upload Document (unstructured) ---
    elif ingest_mode == "Upload Document (unstructured)":
        st.markdown("Upload any document (Word, Excel, PowerPoint, email, image, etc.). Uses **unstructured** library for multi-format support.")
        
        supported_types = ["doc", "docx", "odt", "rtf", "txt", "md", "html", "xml", "json",
                          "csv", "xls", "xlsx", "tsv", "ppt", "pptx",
                          "eml", "msg", "epub",
                          "png", "jpg", "jpeg", "tiff", "bmp", "heic"]
        
        uploaded_doc = st.file_uploader("Upload Document", type=supported_types, key="doc_upload")
        
        if uploaded_doc:
            file_ext = os.path.splitext(uploaded_doc.name)[1].lower()
            
            ext_to_topic = {
                ".doc": "Office", ".docx": "Office", ".odt": "Office", ".rtf": "Office",
                ".txt": "Books", ".md": "Books", ".epub": "Books",
                ".html": "Data", ".xml": "Data", ".json": "Data",
                ".csv": "Data", ".xls": "Data", ".xlsx": "Data", ".tsv": "Data",
                ".ppt": "Office", ".pptx": "Office",
                ".eml": "Email", ".msg": "Email",
                ".png": "Images", ".jpg": "Images", ".jpeg": "Images",
                ".tiff": "Images", ".bmp": "Images", ".heic": "Images"
            }
            auto_topic = ext_to_topic.get(file_ext, "Other")
            
            st.success(f"File: `{uploaded_doc.name}` ({uploaded_doc.size/1024:.1f} KB)")
            
            all_companies_list = get_all_companies(db if db else None)
            if not all_companies_list:
                all_companies_list = ["NA", "HP", "Generic"]
            
            all_topics_list = get_all_topics()
            if not all_topics_list:
                all_topics_list = ["TechDocs", "Books", "Academic", "Government", "Financial", "Office", "Email", "Images", "Data", "Demo", "Other"]
            
            col_f1, col_f2 = st.columns(2)
            
            with col_f1:
                company_options_file = ["--- Select Company ---"] + all_companies_list
                if selected_company != "--- Select Company ---" and selected_company in company_options_file:
                    default_idx_file = company_options_file.index(selected_company)
                else:
                    default_idx_file = 0
                file_company = st.selectbox(
                    "Company",
                    company_options_file,
                    index=default_idx_file,
                    key=f"file_company_doc"
                )
            
            with col_f2:
                topic_options_file = ["--- Select Topic ---"] + all_topics_list
                if auto_topic and auto_topic in topic_options_file:
                    default_topic_idx = topic_options_file.index(auto_topic)
                elif selected_topic != "--- Select Topic ---" and selected_topic in topic_options_file:
                    default_topic_idx = topic_options_file.index(selected_topic)
                else:
                    default_topic_idx = 0
                file_topic = st.selectbox(
                    "Topic",
                    topic_options_file,
                    index=default_topic_idx,
                    key=f"file_topic_doc"
                )
            
            if auto_topic:
                st.caption(f"🔍 Auto-detected: Topic='{auto_topic}'")
            
            if file_company == "--- Select Company ---":
                st.warning("⚠️ Please select a company before ingesting.")
            elif file_topic == "--- Select Topic ---":
                st.warning("⚠️ Please select a topic before ingesting.")
            else:
                if st.button("Parse and Ingest", type="primary", key="btn_doc"):
                    progress = st.progress(0, "Initializing...")
                    
                    try:
                        tmp_dir = tempfile.mkdtemp()
                        file_path = os.path.join(tmp_dir, uploaded_doc.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_doc.read())
                        
                        progress.progress(20, "Parsing with unstructured...")
                        from unified_extractor import extract_knowledge_from_file
                        
                        kb = extract_knowledge_from_file(file_path, file_company)
                        
                        if kb.metadata.get("encrypted"):
                            st.warning(f"⚠️ Skipping encrypted PDF: `{uploaded_doc.name}`")
                            progress.progress(100, "Done")
                        else:
                            for entry in kb.entries:
                                if not entry.metadata.get("topic"):
                                    entry.metadata["topic"] = file_topic
                            
                            progress.progress(60, "Saving JSON...")
                            json_path = save_json_copy(kb, uploaded_doc.name)
                            
                            progress.progress(80, "Vectorizing and storing...")
                            embeddings = get_embedding_model()
                            db, count, incremental_result = ingest_entries_to_db(kb.entries, embeddings, uploaded_doc.name)
                            
                            progress.progress(100, "Done")
                            st.success(f"Ingested {count} entries from `{uploaded_doc.name}` into vector database.")

                            show_incremental_results(incremental_result, kb, count)
                        
                        with open(json_path, "r", encoding="utf-8") as f:
                            st.download_button("Download JSON", f.read(), os.path.basename(json_path), "application/json")
                        
                    except Exception as ex:
                        progress.empty()
                        st.error(f"Error: {ex}")
                        st.exception(ex)

    # --- MODE 3: Upload JSON ---
    elif ingest_mode == "Upload JSON (pre-parsed)":
        st.markdown("Upload JSON files produced by marker/Docling/LlamaParse on Colab. **You can upload multiple files and assign different Company/Topic for each.**")

        uploaded_jsons = st.file_uploader("Upload JSON (multiple allowed)", type=["json"], accept_multiple_files=True, key="json_uploads")

        # Save uploaded files to session_state for persistence across reruns
        if uploaded_jsons and len(uploaded_jsons) > 0:
            st.session_state['uploaded_files_data'] = []
            for f in uploaded_jsons:
                st.session_state['uploaded_files_data'].append({
                    'name': f.name,
                    'data': f.getvalue()
                })
            print(f"[DEBUG] Saved {len(uploaded_jsons)} files to session_state")
        
        # Load files from session_state if available
        files_to_process = []
        if 'uploaded_files_data' in st.session_state and st.session_state['uploaded_files_data']:
            for f in st.session_state['uploaded_files_data']:
                class FakeUploadedFile:
                    def __init__(self, name, data):
                        self.name = name
                        self._data = data
                    def getvalue(self):
                        return self._data
                files_to_process.append(FakeUploadedFile(f['name'], f['data']))
        
        if files_to_process:
            st.markdown(f"### 📁 {len(files_to_process)} file(s) uploaded")
            st.caption("Adjust Company/Topic for each file below, or leave blank to use the default values at the top.")
            
            file_configs = []
            all_companies_list = get_all_companies()
            all_topics_list = get_all_topics()
            
            for idx, uploaded_json in enumerate(files_to_process):
                try:
                    file_name = uploaded_json.name
                    file_data = uploaded_json.getvalue()
                    data = json.loads(file_data.decode("utf-8"))
                    entries_data = data.get("entries", [])
                    source = data.get("source_file", file_name)
                    
                    is_marker_raw = False
                    if len(entries_data) == 1:
                        first_entry = entries_data[0]
                        content = first_entry.get("content", "")
                        if content.startswith("[") and content.endswith("]"):
                            try:
                                json.loads(content)
                                is_marker_raw = True
                            except:
                                pass
                    
                    if is_marker_raw:
                        marker_elements = json.loads(content)
                        entries_data = []
                        for element in marker_elements:
                            elem_type = element.get("type", "")
                            elem_text = element.get("text", "")
                            if elem_text and isinstance(elem_text, str):
                                entry_dict = {
                                    "id": str(uuid.uuid4()),
                                    "type": "general",
                                    "title": f"{elem_type}: {elem_text[:50]}..." if len(elem_text) > 50 else f"{elem_type}: {elem_text}",
                                    "content": elem_text,
                                    "source_file": source,
                                    "source_page": 0,
                                    "keywords": [],
                                    "created_at": datetime.now().isoformat(),
                                    "metadata": {"element_type": elem_type}
                                }
                                entries_data.append(entry_dict)
                    
                    detected_company = ""
                    detected_topic = ""
                    if entries_data:
                        first_meta = entries_data[0].get("metadata", {})
                        detected_company = first_meta.get("company", "")
                        detected_topic = first_meta.get("topic", "")
                    
                    if not detected_company:
                        filename_detected = detect_company_from_filename(source)
                        if filename_detected:
                            detected_company = filename_detected
                    if not detected_topic:
                        topic_detected = detect_topic_from_filename(source)
                        if topic_detected:
                            detected_topic = topic_detected
                    
                    final_company = detected_company
                    if not final_company and selected_company != "--- Select Company ---":
                        final_company = selected_company
                    if not final_company:
                        final_company = "NA"
                    
                    final_topic = detected_topic
                    if not final_topic and selected_topic != "--- Select Topic ---":
                        final_topic = selected_topic
                    if not final_topic:
                        final_topic = "Other"
                    
                except Exception as e:
                    print(f"[DEBUG UPLOAD ERROR] File {file_name}: {e}")
                    entries_data = []
                    source = file_name
                    final_company = selected_company if selected_company != "--- Select Company ---" else "NA"
                    final_topic = selected_topic if selected_topic != "--- Select Topic ---" else "Other"
                
                with st.container():
                    st.markdown(f"#### 📄 {file_name}")
                    
                    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
                    
                    with col_f1:
                        company_options_file = ["--- Select Company ---"] + all_companies_list
                        if final_company and final_company in company_options_file:
                            default_idx_file = company_options_file.index(final_company)
                        elif final_company:
                            company_options_file.insert(1, final_company)
                            default_idx_file = 1
                        else:
                            default_idx_file = 0
                        file_company = st.selectbox(
                            "Company",
                            company_options_file,
                            index=default_idx_file,
                            key=f"file_company_{idx}"
                        )
                    
                    with col_f2:
                        topic_options_file = ["--- Select Topic ---"] + all_topics_list
                        if final_topic and final_topic in topic_options_file:
                            default_topic_idx = topic_options_file.index(final_topic)
                        else:
                            default_topic_idx = 0
                        file_topic = st.selectbox(
                            "Topic",
                            topic_options_file,
                            index=default_topic_idx,
                            key=f"file_topic_{idx}"
                        )
                    
                    with col_f3:
                        st.markdown(f"**Entries:**")
                        st.markdown(f"`{len(entries_data)}`")
                    
                    if detected_company or detected_topic:
                        st.caption(f"🔍 Auto-detected: Company='{detected_company}', Topic='{detected_topic}'")
                    
                    is_valid = file_company != "--- Select Company ---" and file_topic != "--- Select Topic ---"
                    file_configs.append({
                        "name": file_name,
                        "data": data if 'data' in dir() else None,
                        "entries": entries_data,
                        "source": source,
                        "company": file_company,
                        "topic": file_topic,
                        "valid": is_valid
                    })
                    
                    st.markdown("---")
            
            valid_count = sum(1 for fc in file_configs if fc["valid"])
            invalid_count = len(file_configs) - valid_count
            
            print(f"[DEBUG] valid_count={valid_count}, invalid_count={invalid_count}, total={len(file_configs)}")
            for i, fc in enumerate(file_configs):
                print(f"[DEBUG] File {i}: name={fc['name']}, valid={fc['valid']}, company='{fc['company']}', topic='{fc['topic']}', entries={len(fc['entries'])}")
            
            print(f"[DEBUG BUTTON] valid_count={valid_count}, invalid_count={invalid_count}, total_files={len(file_configs)}", flush=True)
            for i, fc in enumerate(file_configs):
                print(f"[DEBUG BUTTON] File {i}: name={fc['name']}, valid={fc['valid']}, company={fc['company']}, topic={fc['topic']}, entries={len(fc['entries'])}", flush=True)
            
            if invalid_count > 0:
                st.warning(f"⚠️ {invalid_count} file(s) missing Company/Topic. They will be skipped.")
            
            button_disabled = (valid_count == 0)
            print(f"[DEBUG BUTTON] Button disabled: {button_disabled}", flush=True)
            
            if st.button(f"🚀 Ingest {valid_count} file(s) to Database", type="primary", key="btn_json_batch", disabled=button_disabled):
                import time
                t_start = time.time()
                debug_print(f"Ingest button clicked at {time.time()}")
                
                progress = st.progress(0, "Initializing...")
                
                debug_print(f"Step 1: Loading knowledge_schema at {time.time()-t_start:.2f}s")
                from knowledge_schema import KnowledgeBase, KnowledgeEntry
                
                debug_print(f"Step 2: Loading embedding model at {time.time()-t_start:.2f}s")
                embeddings = get_embedding_model()
                debug_print(f"Step 2 done: embedding model loaded at {time.time()-t_start:.2f}s")
                
                total_ingested = 0
                for i, fc in enumerate(file_configs):
                    if not fc["valid"]:
                        continue
                    
                    progress.progress(int((i / max(len(file_configs), 1)) * 100), f"Processing {fc['name']}...")
                    
                    try:
                        debug_print(f"Step 3: Creating KnowledgeBase for {fc['name']} at {time.time()-t_start:.2f}s")
                        kb = KnowledgeBase(source_file=fc["source"])
                        debug_print(f"Step 3 done: KnowledgeBase created at {time.time()-t_start:.2f}s")
                        
                        debug_print(f"Step 4: Creating {len(fc['entries'])} KnowledgeEntry objects at {time.time()-t_start:.2f}s")
                        for item in fc["entries"]:
                            try:
                                entry = KnowledgeEntry(**item)
                                if not entry.metadata:
                                    entry.metadata = {}
                                entry.metadata["company"] = fc["company"]
                                entry.metadata["topic"] = fc["topic"]
                                kb.add(entry)
                            except Exception as entry_ex:
                                debug_print(f"ERROR: Failed to create KnowledgeEntry: {entry_ex}")
                                import traceback
                                traceback.print_exc()
                        debug_print(f"Step 4 done: {len(kb.entries)} entries added at {time.time()-t_start:.2f}s")
                        
                        debug_print(f"Step 5: Calling ingest_entries_to_db at {time.time()-t_start:.2f}s")
                        try:
                            _, count, incremental_result = ingest_entries_to_db(kb.entries, embeddings, fc["source"])
                            debug_print(f"Step 5 done: ingest_entries_to_db returned count={count} at {time.time()-t_start:.2f}s")
                        except Exception as ingest_ex:
                            debug_print(f"ERROR: ingest_entries_to_db failed: {ingest_ex}")
                            import traceback
                            traceback.print_exc()
                            raise
                        
                        debug_print(f"Step 6: Saving JSON copy at {time.time()-t_start:.2f}s")
                        save_json_copy(kb, fc["name"])
                        debug_print(f"Step 6 done: JSON copy saved at {time.time()-t_start:.2f}s")
                        
                        total_ingested += count
                        
                        if count == 0:
                            result_text = f"⏭️ {fc['name']}: All entries are unchanged, skipped"
                            st.info(result_text)
                        elif count < 0:
                            result_text = f"🗑️ {fc['name']}: {-count} entries deleted"
                            st.warning(result_text)
                        else:
                            result_text = f"✅ {fc['name']}: {count} entries → {fc['company']}/{fc['topic']}"
                            st.success(result_text)
                        
                        if incremental_result:
                            st.markdown("---")
                            st.subheader("📊 LiveVectorLake Incremental Update")
                            col_inc1, col_inc2, col_inc3, col_inc4 = st.columns(4)
                            col_inc1.metric("New Entries", incremental_result.get("new", 0))
                            col_inc2.metric("Modified", incremental_result.get("modified", 0))
                            col_inc3.metric("Deleted", incremental_result.get("deleted", 0))
                            col_inc4.metric("Unchanged", incremental_result.get("unchanged", 0))
                            if "embeddings_computed" in incremental_result:
                                computed = incremental_result.get("embeddings_computed", 0)
                                reused = incremental_result.get("embeddings_reused", 0)
                                total = computed + reused
                                if total > 0:
                                    savings = ((total - computed) / total) * 100
                                    st.success(f"✅ Embedding savings: {savings:.1f}% ({reused}/{total} entries reused)")
                            if "version_number" in incremental_result:
                                st.info(f"📝 Document version: {incremental_result['version_number']}")
                        
                    except Exception as ex:
                        st.error(f"❌ {fc['name']}: {ex}")
                
                progress.progress(100, "Done!")
                if total_ingested > 0:
                    st.success(f"🎉 Total: {total_ingested} entries added/modified in database!")
                elif total_ingested < 0:
                    st.warning(f"🗑️ Total: {-total_ingested} entries removed from database!")
                else:
                    st.info("⏭️ No changes were made to the database.")
                
                st.session_state.ingest_success_message = f"✅ Process completed (total change: {total_ingested} entries)"
                
                # Clear uploaded files after successful ingestion
                if 'uploaded_files_data' in st.session_state:
                    del st.session_state['uploaded_files_data']
                    print(f"[DEBUG] Cleared uploaded_files_data from session_state")

    # --- MODE 4: Website URL ---
    elif ingest_mode == "Website URL":
        st.markdown("Enter a URL to extract content using **trafilatura** and ingest it into the knowledge base.")

        url_input = st.text_input("Enter URL", placeholder="https://kb.netgear.com/000049616/How-do-I-set-up-my-NETGEAR-router")

        if url_input and st.button("Extract and Ingest", type="primary", key="btn_url"):
            progress = st.progress(0, "Fetching page...")

            try:
                import trafilatura
                from knowledge_schema import KnowledgeBase, KnowledgeEntry
                from marker_extractor import markdown_to_knowledge_entries

                downloaded = trafilatura.fetch_url(url_input)
                if not downloaded:
                    st.error("Could not fetch the URL. Check the address and try again.")
                else:
                    progress.progress(30, "Extracting content...")
                    text = trafilatura.extract(downloaded, include_tables=True, output_format="txt")

                    if not text or len(text) < 50:
                        st.error("No usable content found on this page (may require JavaScript rendering).")
                    else:
                        progress.progress(50, "Structuring entries...")
                        # Use URL path as context
                        structured_text = f"## {url_input}\n\n{text}"
                        entries = markdown_to_knowledge_entries(structured_text, url_input, chunk_size=chunk_size)

                        kb = KnowledgeBase(source_file=url_input)
                        for e in entries:
                            e.metadata = {"company": selected_company}
                            kb.add(e)

                        progress.progress(70, "Vectorizing...")
                        embeddings = get_embedding_model()
                        db, count = ingest_entries_to_db(kb.entries, embeddings, url_input)

                        # Save JSON
                        url_name = url_input.split("/")[-1][:40] or "website"
                        save_json_copy(kb, f"web_{url_name}.json")

                        progress.progress(100, "Done")
                        st.success(f"Ingested {count} entries from URL into vector database.")
                        st.info(f"Extracted {len(text)} characters of content.")

                        # Show extracted content preview
                        with st.expander("Extracted Content Preview", expanded=True):
                            for i, e in enumerate(kb.entries):
                                st.markdown(f"**[{e.type}] {e.title[:80]}**")
                                st.text(e.content[:500] + ("..." if len(e.content) > 500 else ""))
                                if i < len(kb.entries) - 1:
                                    st.markdown("---")

            except ImportError:
                st.error("trafilatura not installed. Run: pip install trafilatura")
            except Exception as ex:
                progress.empty()
                st.error(f"Error: {ex}")


# ========================================
# TAB 2: DATABASE OVERVIEW
# ========================================
with tab2:
    st.subheader("Database Overview")

    if not os.path.exists(DB_DIRECTORY):
        st.info("No data in database yet. Use the Ingest tab to add documents.")
    else:
        try:
            # Get total count using direct SQLite access
            reader = DBReader(DB_PATH)
            total = reader.count_documents()
            st.metric("Total entries stored", total)
            st.markdown("---")

            # LLM Classification Button
            st.markdown("### 🤖 LLM Classification")
            st.markdown("Run **LLM** classification on entries in the database.")
            st.markdown("This will analyze each entry and update its type based on AI understanding.")
            
            # Get all companies and topics from unified source
            llm_companies = get_all_companies()
            llm_topics = get_all_topics()
            
            # Get counts for display
            llm_company_counts, llm_topic_counts = get_metadata_counts()
            
            col_llm1, col_llm2 = st.columns(2)
            with col_llm1:
                llm_company_options = ["All Companies"]
                for c in sorted(llm_companies):
                    count = llm_company_counts.get(c, 0)
                    llm_company_options.append(f"{c} ({count})")
                selected_llm_company_raw = st.selectbox("Filter by company", llm_company_options, index=0, key="llm_company_select")
                selected_llm_company = "All Companies" if selected_llm_company_raw == "All Companies" else selected_llm_company_raw.split(" (")[0]
            with col_llm2:
                llm_topic_options = ["All Topics"]
                for t in sorted(llm_topics):
                    count = llm_topic_counts.get(t, 0)
                    llm_topic_options.append(f"{t} ({count})")
                selected_llm_topic_raw = st.selectbox("Filter by topic", llm_topic_options, index=0, key="llm_topic_select")
                selected_llm_topic = "All Topics" if selected_llm_topic_raw == "All Topics" else selected_llm_topic_raw.split(" (")[0]
            
            if st.button("Run LLM Classification", type="primary", key="btn_llm"):
                progress = st.progress(0, "Initializing...")
                status_text = st.empty()
                
                # First pass: get statistics before starting
                stats_result = None
                stats_generator = run_llm_classification_on_db(selected_llm_company, selected_llm_topic)
                try:
                    first_result = next(stats_generator)
                    status, total_cnt, msg, details = first_result
                    
                    if status == "info":
                        stats_result = first_result
                        status_text.info(f"ℹ️ {msg}")
                        
                        if total_cnt > 0:
                            estimated_time = (total_cnt * 4.5) / 60  # ~4.5s per API call
                            confirmation = st.radio(
                                f"⚠️ Confirm Classification",
                                [
                                    f"Proceed with {total_cnt} entries (~{estimated_time:.1f} mins, {total_cnt} API calls)",
                                    "Cancel"
                                ],
                                key="llm_confirm_radio"
                            )
                            
                            if confirmation == "Cancel":
                                progress.empty()
                                status_text.warning("Classification cancelled.")
                            else:
                                progress.progress(0, f"Starting classification of {total_cnt} entries...")
                                
                                # Process classification
                                classification_generator = stats_generator if stats_result else run_llm_classification_on_db(selected_llm_company, selected_llm_topic)
                                
                                for result in classification_generator:
                                    status, total_cnt, msg, details = result
                                    
                                    if status == "done":
                                        progress.progress(100, "Completed!")
                                        status_text.success(f"✅ LLM classification completed! Results saved to: {msg}")
                                        
                                        with st.expander("View classification results"):
                                            st.markdown("**LLM Classification Results:**")
                                            st.markdown(f"- Total entries processed: {total_cnt}")
                                            if details > 0:
                                                st.markdown(f"- Skipped (already classified): {details}")
                                            st.markdown(f"- Results saved to: `{msg}`")
                                            
                                    elif status == "error":
                                        progress.empty()
                                        status_text.error(f"❌ Error: {msg}")
                                        
                                    elif status == "info":
                                        pass
                                        
                                    else:
                                        progress.progress(int((status / total_cnt) * 100), f"Processing {status}/{total_cnt}...")
                                        status_text.info(f"Processing entry: {msg[:50]}...")
                    else:
                        stats_generator = run_llm_classification_on_db(selected_llm_company, selected_llm_topic)
                        classification_generator = stats_generator if stats_result else run_llm_classification_on_db(selected_llm_company, selected_llm_topic)
                        
                        for result in classification_generator:
                            status, total_cnt, msg, details = result
                            
                            if status == "done":
                                progress.progress(100, "Completed!")
                                status_text.success(f"✅ LLM classification completed! Results saved to: {msg}")
                                
                                with st.expander("View classification results"):
                                    st.markdown("**LLM Classification Results:**")
                                    st.markdown(f"- Total entries processed: {total_cnt}")
                                    if details > 0:
                                        st.markdown(f"- Skipped (already classified): {details}")
                                    st.markdown(f"- Results saved to: `{msg}`")
                                    
                            elif status == "error":
                                progress.empty()
                                status_text.error(f"❌ Error: {msg}")
                                
                            elif status == "info":
                                pass
                                
                            else:
                                progress.progress(int((status / total_cnt) * 100), f"Processing {status}/{total_cnt}...")
                                status_text.info(f"Processing entry: {msg[:50]}...")
                                
                except StopIteration:
                    pass

            st.markdown("---")

            # Rebuild FAISS Index
            st.markdown("### 🔄 Rebuild FAISS Index")
            st.markdown("Rebuild the FAISS vector index from the SQLite database. This is useful after deleting entries to clear cached data.")
            
            if st.button("Rebuild FAISS Index", type="secondary", key="btn_rebuild_faiss"):
                faiss_progress = st.progress(0, "Initializing...")
                faiss_status = st.empty()
                
                try:
                    faiss_status.info("Loading FAISS search module...")
                    from faiss_search import FAISSSearch
                    
                    faiss_status.info("Creating FAISS search instance...")
                    searcher = FAISSSearch()
                    
                    faiss_status.info("Building index from SQLite...")
                    searcher._build_from_sqlite()
                    
                    faiss_status.info("Clearing cache...")
                    st.cache_resource.clear()
                    
                    faiss_progress.progress(100, "Completed!")
                    faiss_status.success("✅ FAISS index rebuilt successfully!")
                    st.info(f"Index contains {len(searcher.doc_ids)} documents")
                except Exception as faiss_ex:
                    faiss_progress.empty()
                    faiss_status.error(f"❌ Failed to rebuild FAISS index: {str(faiss_ex)}")
            
            st.markdown("---")

            # Delete Data by Topic/Company
            st.markdown("### 🗑️ Delete Data from Database")
            st.markdown("Remove entries from the vector database by company and/or topic.")
            
            # Get companies and topics that actually exist in the database
            db_companies_in_data = get_all_companies()
            db_topics_in_data = get_all_topics()
            
            col_del1, col_del2 = st.columns(2)
            
            # Get counts for display
            del_company_counts, del_topic_counts = get_metadata_counts()
            
            with col_del1:
                del_company_options = ["All Companies"]
                for c in sorted(db_companies_in_data):
                    count = del_company_counts.get(c, 0)
                    del_company_options.append(f"{c} ({count})")
                del_company_raw = st.selectbox("Filter by company", del_company_options, index=0, key="del_company_select")
                # Extract actual company name
                if del_company_raw == "All Companies":
                    del_company = del_company_raw
                else:
                    del_company = del_company_raw.split(" (")[0]
            
            with col_del2:
                del_topic_options = ["All Topics"]
                for t in sorted(db_topics_in_data):
                    count = del_topic_counts.get(t, 0)
                    del_topic_options.append(f"{t} ({count})")
                del_topic_raw = st.selectbox("Filter by topic", del_topic_options, index=0, key="del_topic_select")
                # Extract actual topic name
                if del_topic_raw == "All Topics":
                    del_topic = del_topic_raw
                else:
                    del_topic = del_topic_raw.split(" (")[0]
            
            # Preview count of entries that will be deleted using DBReader
            reader = DBReader(DB_PATH)
            all_metas = reader.get_all_metadatas()
            
            # Apply filters manually
            filtered_preview_ids = []
            for doc_id, meta in all_metas.items():
                if not isinstance(meta, dict):
                    continue
                
                comp = str(meta.get("company", "")).strip()
                top = str(meta.get("topic", "")).strip()
                
                company_match = (del_company == "All Companies") or \
                    (del_company == "NA" and (comp == "" or comp == "NA")) or \
                    (del_company == comp)
                
                topic_match = (del_topic == "All Topics") or \
                    (del_topic == "NA" and (top == "" or top == "NA")) or \
                    (del_topic == top)
                
                if company_match and topic_match:
                    filtered_preview_ids.append(doc_id)
            
            preview_count = len(filtered_preview_ids)
            
            st.info(f"📊 This will affect **{preview_count}** entries in the database.")
            
            with st.expander("⚠️ Danger Zone - Confirm Deletion"):
                # Build proper filter description
                filter_desc_parts = []
                if del_company != "All Companies":
                    filter_desc_parts.append(f"company='{del_company}'")
                else:
                    filter_desc_parts.append("all companies")
                if del_topic != "All Topics":
                    filter_desc_parts.append(f"topic='{del_topic}'")
                else:
                    filter_desc_parts.append("all topics")
                
                filter_desc = " AND ".join(filter_desc_parts)
                st.warning(f"This will **permanently delete** {preview_count} entries with: {filter_desc}")
                
                col_confirm1, col_confirm2 = st.columns(2)
                with col_confirm1:
                    if st.button(f"🗑️ Delete {preview_count} Entries", type="primary", key="btn_delete_entries"):
                        try:
                            if preview_count == 0:
                                st.warning("No entries match the filter. Nothing to delete.")
                            else:
                                from db_writer import DBWriter
                                db_writer = DBWriter(DB_PATH)
                                deleted = db_writer.delete_documents_by_entry_id(filtered_preview_ids)
                                
                                st.success(f"✅ Successfully deleted {deleted} entries!")
                                st.rerun()
                        except Exception as del_ex:
                            st.error(f"❌ Deletion failed: {del_ex}")
                with col_confirm2:
                    st.info("⚠️ Cannot be undone!")

            st.markdown("---")

            # Company Management Section
            st.markdown("### 🏢 Company Management")
            
            reader = DBReader(DB_PATH)
            all_metas = reader.get_all_metadatas()
            db_companies = set()
            for m in all_metas.values():
                if isinstance(m, dict):
                    comp = str(m.get("company", "")).strip()
                    if comp and comp != "NA":
                        db_companies.add(comp)
            
            from company_manager import get_companies
            file_companies = get_companies()
            
            companies = sorted(list(set(file_companies + list(db_companies))))
            if "NA" not in companies:
                companies.insert(0, "NA")
                
            col_c1, col_c2 = st.columns(2)
                
            with col_c1:
                st.markdown("**Existing Companies:**")
                if companies:
                    for company in sorted(companies):
                        st.markdown(f"- `{company}`")
                else:
                    st.markdown("- No companies yet")
                
            with col_c2:
                st.markdown("**Add New Company:**")
                new_company = st.text_input("Company name", key="new_company")
                if st.button("Add Company", key="btn_add_company"):
                    if new_company.strip():
                        try:
                            from company_manager import add_company
                            if add_company(new_company.strip()):
                                st.success(f"✅ Added: {new_company}")
                                st.rerun()
                            else:
                                st.warning("⚠️ Already exists or invalid")
                        except Exception as e:
                            st.error(f"Failed to add company: {e}")
                
            if companies:
                st.markdown("**Remove Company:**")
                remove_company_name = st.selectbox("Select company to remove", companies, key="remove_company_select")
                if st.button("Remove Company", type="secondary", key="btn_remove_company"):
                    try:
                        from company_manager import remove_company
                        if remove_company(remove_company_name):
                            st.success(f"✅ Removed: {remove_company_name}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to remove company: {e}")

            st.markdown("---")

            # Multi-dimensional Filter
            st.markdown("### 🔍 Filter by Company & Topic")
            st.markdown("Select filters to view specific data:")
            
            # Get all metadata using DBReader
            reader = DBReader(DB_PATH)
            all_metas_dict = reader.get_all_metadatas()
            metas_list = list(all_metas_dict.values())
            
            # Get all companies and topics from unified source with counts
            company_counts = {}
            topic_counts = {}
            for m in metas_list:
                if isinstance(m, dict):
                    comp = m.get("company", "")
                    top = m.get("topic", "")
                    # Count all entries, including empty ones
                    comp_key = comp if comp else "NA"
                    top_key = top if top else "NA"
                    company_counts[comp_key] = company_counts.get(comp_key, 0) + 1
                    topic_counts[top_key] = topic_counts.get(top_key, 0) + 1
            
            # Build options with counts (sorted by count descending)
            company_options_with_count = ["All Companies"]
            for c, cnt in sorted(company_counts.items(), key=lambda x: -x[1]):
                company_options_with_count.append(f"{c} ({cnt})")
            
            topic_options_with_count = ["All Topics"]
            for t, cnt in sorted(topic_counts.items(), key=lambda x: -x[1]):
                topic_options_with_count.append(f"{t} ({cnt})")
            
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                default_company_idx = 0
                if hasattr(st.session_state, 'filter_company_select'):
                    try:
                        default_company_idx = company_options_with_count.index(st.session_state.filter_company_select)
                    except ValueError:
                        default_company_idx = 0
                selected_filter_company_raw = st.selectbox("Filter by company", company_options_with_count, index=default_company_idx, key="filter_company_select")
                selected_filter_company = selected_filter_company_raw.split(" (")[0] if " (" in selected_filter_company_raw else selected_filter_company_raw
            with col_filter2:
                default_topic_idx = 0
                if hasattr(st.session_state, 'filter_topic_select'):
                    try:
                        default_topic_idx = topic_options_with_count.index(st.session_state.filter_topic_select)
                    except ValueError:
                        default_topic_idx = 0
                selected_filter_topic_raw = st.selectbox("Filter by topic", topic_options_with_count, index=default_topic_idx, key="filter_topic_select")
                selected_filter_topic = selected_filter_topic_raw.split(" (")[0] if " (" in selected_filter_topic_raw else selected_filter_topic_raw

            if total > 0:
                metas = metas_list

                # Filter by company if selected
                if selected_filter_company != "All Companies":
                    def match_company(meta, target):
                        if not isinstance(meta, dict):
                            return False
                        comp = str(meta.get("company", "")).strip()
                        if not comp and isinstance(meta.get("metadata"), dict):
                            comp = str(meta["metadata"].get("company", "")).strip()
                        if target == "NA":
                            return comp == "" or comp == "NA"
                        return comp == target
                    
                    metas = [m for m in metas if match_company(m, selected_filter_company)]
                
                # Filter by topic if selected
                if selected_filter_topic != "All Topics":
                    def match_topic(meta, target):
                        if not isinstance(meta, dict):
                            return False
                        top = str(meta.get("topic", "")).strip()
                        if not top and isinstance(meta.get("metadata"), dict):
                            top = str(meta["metadata"].get("topic", "")).strip()
                        if target == "NA":
                            return top == "" or top == "NA"
                        return top == target
                    
                    metas = [m for m in metas if match_topic(m, selected_filter_topic)]
                
                filtered_total = len(metas)
                filter_desc = []
                if selected_filter_company != "All Companies":
                    filter_desc.append(f"Company: {selected_filter_company}")
                if selected_filter_topic != "All Topics":
                    filter_desc.append(f"Topic: {selected_filter_topic}")
                if filter_desc:
                    st.info(f"Showing data for **{', '.join(filter_desc)}**: {filtered_total} entries")
                else:
                    st.info(f"Showing all data: {filtered_total} entries")

                # Group by source file
                sources = {}
                types = {}
                llm_types = {}
                companies_in_db = {}
                topics_in_db = {}
                for m in metas:
                    if m is None:
                        m = {}
                    src = m.get("source_file", "unknown") or "unknown"
                    sources[src] = sources.get(src, 0) + 1
                    t = m.get("type", "general") or "general"
                    types[t] = types.get(t, 0) + 1
                    company = m.get("company", "") or "unknown"
                    companies_in_db[company] = companies_in_db.get(company, 0) + 1
                    topic = m.get("topic", "") or "unknown"
                    topics_in_db[topic] = topics_in_db.get(topic, 0) + 1
                    llm_t = m.get("llm_type", "not_set")
                    if llm_t == "not_set" and isinstance(m.get("metadata"), dict):
                        llm_t = m["metadata"].get("llm_type", "not_set")
                    llm_types[llm_t] = llm_types.get(llm_t, 0) + 1

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown("**By Source File**")
                    if sources:
                        for src, cnt in sorted(sources.items(), key=lambda x: -x[1])[:5]:
                            display_src = src if len(src) <= 40 else src[:37] + "..."
                            st.markdown(f"- `{display_src}`: **{cnt}**")
                    else:
                        st.markdown("- No data")

                with col2:
                    st.markdown("**By Type (Rule-based)**")
                    if types:
                        for t, cnt in sorted(types.items(), key=lambda x: -x[1]):
                            st.markdown(f"- `{t}`: **{cnt}**")
                    else:
                        st.markdown("- No data")
                
                with col3:
                    st.markdown("**By Topic**")
                    if topics_in_db:
                        for topic, cnt in sorted(topics_in_db.items(), key=lambda x: -x[1]):
                            display_topic = topic if topic != "unknown" else "NA"
                            st.markdown(f"- `{display_topic}`: **{cnt}**")
                    else:
                        st.markdown("- No topic data yet")
                
                with col4:
                    st.markdown("**By Company**")
                    if companies_in_db:
                        for company, cnt in sorted(companies_in_db.items(), key=lambda x: -x[1]):
                            display_company = company if company != "unknown" else "NA"
                            st.markdown(f"- `{display_company}`: **{cnt}**")
                    else:
                        st.markdown("- No company data yet")

                st.markdown("---")

                # Detailed Entries View
                st.markdown("### 📋 Detailed Entries")
                st.markdown(f"Showing {filtered_total} entries (first 50):")
                
                # Get all data including documents for detailed view using DBReader
                reader = DBReader(DB_PATH)
                all_docs = reader.get_documents_with_metadatas()
                
                # Apply both company and topic filters
                filtered_ids = []
                filtered_metadatas = []
                filtered_docs = []
                for entry_id, data in all_docs.items():
                    m = data.get("metadata", {})
                    doc = data.get("content", "")
                    
                    if not isinstance(m, dict):
                        continue
                    
                    comp = str(m.get("company", "")).strip()
                    top = str(m.get("topic", "")).strip()
                    
                    company_match = (selected_filter_company == "All Companies") or \
                        (selected_filter_company == "NA" and (comp == "" or comp == "NA")) or \
                        (selected_filter_company == comp)
                    
                    topic_match = (selected_filter_topic == "All Topics") or \
                        (selected_filter_topic == "NA" and (top == "" or top == "NA")) or \
                        (selected_filter_topic == top)
                    
                    if company_match and topic_match:
                        filtered_ids.append(entry_id)
                        filtered_metadatas.append(m)
                        filtered_docs.append(doc)
                
                detailed_data = {
                    "ids": filtered_ids,
                    "metadatas": filtered_metadatas,
                    "documents": filtered_docs
                }
                
                # Show entries in expanders
                for i, (entry_id, meta, doc) in enumerate(zip(
                    detailed_data["ids"], 
                    detailed_data["metadatas"], 
                    detailed_data["documents"]
                )):
                    if i >= 50:
                        st.info(f"... and {len(detailed_data['ids']) - 50} more entries")
                        break
                        
                    if meta is None:
                        meta = {}
                    
                    entry_type = meta.get("type", "general")
                    source_file = meta.get("source_file", "unknown")
                    company_name = ""
                    if isinstance(meta, dict):
                        if "company" in meta:
                            company_name = str(meta.get("company", "")).strip()
                        elif isinstance(meta.get("metadata"), dict) and "company" in meta["metadata"]:
                            company_name = str(meta["metadata"].get("company", "")).strip()
                    
                    title = meta.get("title", "")
                    if not title:
                        title = doc[:60] if doc else "(no content)"
                        if len(doc) > 60:
                            title += "..."
                    
                    page = meta.get("source_page", 0)
                    
                    llm_type = meta.get("llm_type", "")
                    if not llm_type:
                        llm_type = meta.get("metadata", {}).get("llm_type", "")
                    
                    llm_confidence = meta.get("llm_confidence", 0)
                    if llm_confidence == 0:
                        llm_confidence = meta.get("metadata", {}).get("llm_confidence", 0)
                    
                    llm_badge = ""
                    if llm_type:
                        llm_badge = f" [LLM:{llm_type}]"
                    
                    with st.expander(f"[{entry_type}]{llm_badge} {title[:70]} (p.{page})"):
                        content_part = doc
                        if "[Content] " in doc:
                            content_part = doc.split("[Content] ", 1)[1]
                            if "[Keywords]" in content_part:
                                content_part = content_part.split("[Keywords]")[0].strip()
                        elif "[Title] " in doc:
                            content_part = doc.split("\n", 1)[1] if "\n" in doc else doc
                        
                        import re
                        content_part = re.sub(r'<span\s+[^>]*></span>', '', content_part)
                        content_part = re.sub(r'<[^>]+>', '', content_part)
                        content_part = content_part.strip()
                        
                        if content_part:
                            st.markdown(content_part[:800])
                            if len(content_part) > 800:
                                st.caption("(truncated)")
                        else:
                            st.text("(no content)")
                        
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            if company_name:
                                st.caption(f"Company: {company_name}")
                            if meta.get("topic"):
                                st.caption(f"Topic: {meta['topic']}")
                        with col_info2:
                            st.caption(f"Source: {source_file}")
                        
                        if meta.get("keywords"):
                            st.caption(f"Keywords: {meta['keywords']}")
                        
                        if llm_type:
                            llm_keywords = meta.get("llm_keywords", "")
                            st.caption(f"LLM Type: {llm_type} (Confidence: {llm_confidence:.1%})")
                            if llm_keywords:
                                st.caption(f"LLM Keywords: {llm_keywords}")

                st.markdown("---")

                # Option to clear database
                with st.expander("Danger Zone"):
                    st.warning("This will permanently delete all data in the vector database.")
                    if st.button("Clear entire database", type="secondary"):
                        try:
                            import shutil
                            if os.path.exists(DB_DIRECTORY):
                                shutil.rmtree(DB_DIRECTORY)
                            bm25_dir = os.path.join(BASE_DIR, "bm25_index")
                            if os.path.exists(bm25_dir):
                                shutil.rmtree(bm25_dir)
                            for idx_file in ["faiss_index.bin", "faiss_index.bin.ids", "faiss_index.bin.docs",
                                             "faiss_index_pq.bin", "faiss_index_pq.bin.ids", "faiss_index_pq.bin.docs"]:
                                idx_path = os.path.join(DB_DIRECTORY.replace("my_local_database", "").strip(os.sep), idx_file)
                                if os.path.exists(idx_path):
                                    os.remove(idx_path)
                            st.success("Database cleared. Refresh the page to see changes.")
                            st.cache_resource.clear()
                            st.cache_data.clear()
                        except Exception as clear_ex:
                            st.error(f"Could not clear: {clear_ex}")

        except Exception as ex:
            st.error(f"Error reading database: {ex}")


# ========================================
# TAB 3: BROWSE ENTRIES
# ========================================
with tab3:
    st.subheader("Browse Entries")

    if not os.path.exists(DB_DIRECTORY):
        st.info("No data yet. Ingest documents first.")
    else:
        try:
            embeddings = get_embedding_model()
            reader = DBReader(DB_PATH)
            all_docs = reader.get_documents_with_metadatas()
            total = len(all_docs)

            if total == 0:
                st.info("Database is empty.")
            else:
                # Convert dict to lists
                ids_list = list(all_docs.keys())
                metas = [all_docs[id].get("metadata", {}) for id in ids_list]
                docs = [all_docs[id].get("content", "") for id in ids_list]

                # Normalize None metadata
                metas = [m if m is not None else {} for m in metas]
                docs = [d if d is not None else "" for d in docs]

                # Filters
                col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns([1, 1, 1, 1, 2])

                all_types = sorted(set(m.get("type", "general") for m in metas))
                all_sources = sorted(set(m.get("source_file", "") for m in metas))
                all_companies = get_all_companies()
                all_topics = get_all_topics()
                
                # Get counts for display
                browse_company_counts, browse_topic_counts = get_metadata_counts()

                def has_llm_classification(meta):
                    if isinstance(meta, dict):
                        if "llm_type" in meta:
                            return meta["llm_type"] != ""
                        if "metadata" in meta and isinstance(meta["metadata"], dict):
                            return "llm_type" in meta["metadata"]
                    return False

                col_f1, col_f2, col_f3, col_f4, col_f5, col_f6 = st.columns([1, 1, 1, 1, 1, 2])

                with col_f1:
                    filter_type = st.selectbox("Type", ["All"] + all_types, key="browse_type_select")
                with col_f2:
                    company_options_browse = ["All"]
                    for c in all_companies:
                        count = browse_company_counts.get(c, 0)
                        company_options_browse.append(f"{c} ({count})")
                    filter_company_raw = st.selectbox("Company", company_options_browse, key="browse_company_select")
                    filter_company = "All" if filter_company_raw == "All" else filter_company_raw.split(" (")[0]
                with col_f3:
                    topic_options_browse = ["All"]
                    for t in all_topics:
                        count = browse_topic_counts.get(t, 0)
                        topic_options_browse.append(f"{t} ({count})")
                    filter_topic_raw = st.selectbox("Topic", topic_options_browse, key="browse_topic_select")
                    filter_topic = "All" if filter_topic_raw == "All" else filter_topic_raw.split(" (")[0]
                with col_f4:
                    source_options = ["All"] + [s[:40] for s in all_sources]
                    filter_source = st.selectbox("Source", source_options, key="browse_source_select")
                with col_f5:
                    filter_llm = st.selectbox("LLM Classified", ["All", "Yes", "No"], key="browse_llm_select")
                with col_f6:
                    search_kw = st.text_input("Keyword", placeholder="Enter keyword...")

                # Apply filters
                filtered_indices = list(range(total))

                if filter_type != "All":
                    filtered_indices = [i for i in filtered_indices if metas[i].get("type") == filter_type]
                if filter_company != "All":
                    def match_browse_company(meta):
                        comp = str(meta.get("company", "")).strip()
                        if not comp and isinstance(meta.get("metadata"), dict):
                            comp = str(meta["metadata"].get("company", "")).strip()
                        if filter_company == "NA":
                            return comp == "" or comp == "NA"
                        return comp == filter_company
                    filtered_indices = [i for i in filtered_indices if match_browse_company(metas[i])]
                if filter_topic != "All":
                    def match_browse_topic(meta):
                        top = str(meta.get("topic", "")).strip()
                        if not top and isinstance(meta.get("metadata"), dict):
                            top = str(meta["metadata"].get("topic", "")).strip()
                        if filter_topic == "NA":
                            return top == "" or top == "NA"
                        return top == filter_topic
                    filtered_indices = [i for i in filtered_indices if match_browse_topic(metas[i])]
                if filter_source != "All":
                    filtered_indices = [i for i in filtered_indices
                                      if metas[i].get("source_file", "")[:40] == filter_source]
                if filter_llm != "All":
                    if filter_llm == "Yes":
                        filtered_indices = [i for i in filtered_indices if has_llm_classification(metas[i])]
                    else:
                        filtered_indices = [i for i in filtered_indices if not has_llm_classification(metas[i])]
                if search_kw:
                    kw_lower = search_kw.lower()
                    filtered_indices = [i for i in filtered_indices
                                      if kw_lower in docs[i].lower() or kw_lower in metas[i].get("title", "").lower()]

                st.markdown(f"**Showing {len(filtered_indices)} / {total} entries**")
                st.markdown("---")

                # Display entries
                for idx in filtered_indices[:50]:
                    m = metas[idx]
                    title = m.get("title", "(untitled)")
                    etype = m.get("type", "general")
                    
                    llm_type = m.get("llm_type", "")
                    if not llm_type:
                        llm_type = m.get("metadata", {}).get("llm_type", "")
                    
                    llm_confidence = m.get("llm_confidence", 0)
                    if llm_confidence == 0:
                        llm_confidence = m.get("metadata", {}).get("llm_confidence", 0)
                    
                    page = m.get("source_page", 0)
                    source = m.get("source_file", "")

                    llm_badge = ""
                    if llm_type:
                        llm_badge = f" [LLM:{llm_type}]"

                    with st.expander(f"[{etype}]{llm_badge} {title[:70]} (p.{page})"):
                        raw = docs[idx]
                        content_part = raw
                        if "[Content] " in raw:
                            content_part = raw.split("[Content] ", 1)[1]
                            if "[Keywords]" in content_part:
                                content_part = content_part.split("[Keywords]")[0].strip()
                        elif "[Title] " in raw:
                            content_part = raw.split("\n", 1)[1] if "\n" in raw else raw
                        
                        import re
                        content_part = re.sub(r'<span\s+[^>]*></span>', '', content_part)
                        content_part = re.sub(r'<[^>]+>', '', content_part)
                        content_part = content_part.strip()

                        if content_part:
                            st.markdown(content_part[:800])
                            if len(content_part) > 800:
                                st.caption("(truncated)")
                        else:
                            st.text("(no content)")

                        company_name = m.get("company", "")
                        topic_name = m.get("topic", "")
                        
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            if company_name:
                                st.caption(f"Company: {company_name}")
                            if topic_name:
                                st.caption(f"Topic: {topic_name}")
                        with col_info2:
                            st.caption(f"Source: {source}")

                        kw = m.get("keywords", "")
                        if kw:
                            st.caption(f"Keywords: {kw}")
                        
                        if llm_type:
                            llm_keywords = m.get("llm_keywords", "")
                            st.caption(f"LLM Type: {llm_type} (Confidence: {llm_confidence:.1%})")
                            if llm_keywords:
                                st.caption(f"LLM Keywords: {llm_keywords}")

                if len(filtered_indices) > 50:
                    st.info(f"Showing first 50 of {len(filtered_indices)} entries.")

        except Exception as ex:
            st.error(f"Error: {ex}")


# ========================================
# TAB 4: SEARCH
# ========================================
with tab4:
    st.subheader("Semantic Search")

    if not os.path.exists(DB_DIRECTORY):
        st.info("No data yet. Ingest documents first.")
    else:
        faiss_searcher = None
        
        if st.session_state.get("rebuild_faiss", False):
            import streamlit as st
            st.cache_resource.clear()
            st.session_state["rebuild_faiss"] = False
            st.rerun()
        
        with st.form("search_form", clear_on_submit=False):
            col_q, col_k, col_c, col_m, col_t = st.columns([3, 1, 1, 2, 2])
            with col_q:
                query = st.text_input("Enter your question", placeholder="e.g. How to reset the device?")
            with col_k:
                top_k = st.slider("Results", 1, 10, 3)
            with col_c:
                reader = DBReader(DB_PATH)
                all_metas = reader.get_all_metadatas()
                db_companies = set()
                for m in all_metas.values():
                    if isinstance(m, dict):
                        comp = str(m.get("company", "")).strip()
                        if comp and comp != "NA":
                            db_companies.add(comp)
                all_companies = sorted(list(db_companies))
                
                search_company_options = ["All"]
                for c in all_companies:
                    search_company_options.append(c)
                filter_company_raw = st.selectbox("Company filter", search_company_options, key="search_company_select")
                filter_company = "All" if filter_company_raw == "All" else filter_company_raw
            with col_m:
                search_mode = st.selectbox("Search Mode", 
                    ["Pure Vector", "Hybrid (Vector+BM25)", "Hybrid + CrossEncoder", "Hybrid + LLM", "CrossEncoder + AI Assistant"],
                    index=2,
                    key="search_mode_select"
                )
                
                reranker_model = None
                crossencoder_models = get_reranker_models_cache()
                if search_mode == "Hybrid + CrossEncoder" or search_mode == "CrossEncoder + AI Assistant":
                    if crossencoder_models:
                        model_options = [f"{m['name']} ({m['description']})" for m in crossencoder_models]
                        selected_model_str = st.selectbox("Reranker Model", model_options, key="reranker_model_select")
                        reranker_model = crossencoder_models[model_options.index(selected_model_str)]["id"]

            with col_t:
                st.markdown("**Temporal Query**")
                temporal_date = st.date_input("Date", value=None, key="temporal_date", help="Leave empty for latest data")
                temporal_time = st.time_input("Time", value=None, key="temporal_time", help="Leave empty for latest data")

            submit_button = st.form_submit_button("Search" if search_mode != "CrossEncoder + AI Assistant" else "Ask AI", use_container_width=True)

        cached_mode = st.session_state.get("last_search_mode", "")
        if cached_mode != search_mode:
            st.session_state["last_search_results"] = []
            st.session_state["last_llm_answer"] = ""

        print(f"[DEBUG SEARCH] submit_button={submit_button}, query={query}, search_mode={search_mode}")
        
        if submit_button and query:
            try:
                with st.spinner("Initializing search engine..."):
                    from faiss_search import FAISSSearch
                    embeddings = get_embedding_model()
                    faiss_searcher = FAISSSearch(embedding_model=embeddings)
                    faiss_searcher.initialize()
                
                if not faiss_searcher:
                    st.error("❌ Failed to initialize search engine. Please check FAISS index.")
                else:
                    with st.spinner("Searching..." if search_mode != "CrossEncoder + AI Assistant" else "Thinking..."):
                        filter_dict = {}
                        if filter_company != "All":
                            filter_dict["company"] = filter_company

                        timestamp = None
                        if temporal_date and temporal_time:
                            timestamp = f"{temporal_date}T{temporal_time}"
                        elif temporal_date:
                            timestamp = f"{temporal_date}T23:59:59"

                        if search_mode == "Pure Vector":
                            results = faiss_searcher.search(query, k=top_k, filter_dict=filter_dict, timestamp=timestamp)
                        elif search_mode == "Hybrid (Vector+BM25)":
                            results = faiss_searcher.hybrid_search(query, k=top_k, filter_dict=filter_dict, use_bm25=True, reranker_type=None, timestamp=timestamp)
                        elif search_mode == "Hybrid + CrossEncoder":
                            results = faiss_searcher.hybrid_search(query, k=top_k, filter_dict=filter_dict, use_bm25=True, reranker_type='crossencoder', reranker_model=reranker_model, timestamp=timestamp)
                        elif search_mode == "Hybrid + LLM":
                            results = faiss_searcher.hybrid_search(query, k=top_k, filter_dict=filter_dict, use_bm25=True, reranker_type='llm', timestamp=timestamp)
                        elif search_mode == "CrossEncoder + AI Assistant":
                            results = faiss_searcher.hybrid_search(query, k=top_k, filter_dict=filter_dict, use_bm25=True, reranker_type='crossencoder', reranker_model=reranker_model, timestamp=timestamp)
                            try:
                                from llm_qa import generate_answer
                                llm_result = generate_answer(query, results)
                                st.session_state["last_llm_answer"] = llm_result.get('answer', '')
                            except Exception as llm_ex:
                                st.session_state["last_llm_answer"] = f"LLM error: {str(llm_ex)}"
                        st.session_state["last_search_results"] = results
                        st.session_state["last_search_query"] = query
                        st.session_state["last_search_mode"] = search_mode
                        st.session_state["last_search_top_k"] = top_k
                        st.session_state["last_search_company"] = filter_company
            except Exception as search_ex:
                st.error(f"❌ Search failed: {str(search_ex)}")
            
            results = st.session_state.get("last_search_results", [])
            if results:
                mode_str = st.session_state.get("last_search_mode", "")
                query_str = st.session_state.get("last_search_query", "")
                st.markdown(f"**Query:** `{query_str}` | **Mode:** `{mode_str}` | **Results:** {len(results)}")
                
                st.markdown("---")

                current_mode = st.session_state.get("last_search_mode", "")
                if current_mode == "CrossEncoder + AI Assistant":
                    llm_answer = st.session_state.get("last_llm_answer", "")
                    if llm_answer:
                        st.markdown("### 🤖 AI Answer")
                        st.markdown(f"> {llm_answer}")
                        st.markdown("---")
                        st.markdown("### 📚 Sources")

                for i, res in enumerate(results):
                    meta = res.get('metadata', {})
                    raw = res.get('content', '')

                    score_info = ""
                    if 'similarity' in res:
                        score_info = f" | Similarity: {res['similarity']:.4f}"
                    elif 'rrf_score' in res:
                        score_info = f" | RRF: {res['rrf_score']:.4f}"
                    if 'rerank_score' in res:
                        score_info += f" | Rerank: {res['rerank_score']:.4f}"

                    meta_info = f"Type: `{meta.get('type', 'general')}`"
                    if meta.get('source_file'):
                        meta_info += f" | Source: `{meta.get('source_file')[:40]}`"
                    if meta.get('source_page'):
                        meta_info += f" | Page {meta.get('source_page')}"
                    if meta.get('version_number'):
                        meta_info += f" | Version: `{meta.get('version_number')}`"
                    if meta.get('valid_from'):
                        meta_info += f" | Valid from: `{meta.get('valid_from')}`"

                    with st.container():
                        st.markdown(
                            f"**#{i+1}** | {meta_info}{score_info}"
                        )

                        display_text = raw
                        if "[Content] " in raw:
                            display_text = raw.split("[Content] ", 1)[1]
                            if "[Keywords]" in display_text:
                                display_text = display_text.split("[Keywords]")[0].strip()
                        elif "[Title] " in raw:
                            lines = raw.split("\n")
                            content_lines = []
                            for line in lines:
                                if not line.startswith("["):
                                    content_lines.append(line)
                            display_text = "\n".join(content_lines).strip()

                        st.info(display_text[:500])
                        st.markdown("---")
