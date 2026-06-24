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

import streamlit as st
import os
import json
import shutil
import tempfile
from datetime import datetime

# --- Company Auto-Detection ---
from company_manager import get_companies, add_company, remove_company, has_company
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
        import google.generativeai as genai
        
        prompt = f"""Analyze this document and identify the company or organization it belongs to.
        
Filename: {filename}

Content preview (first 500 chars):
{content[:500] if content else '(empty)'}

Respond with ONLY the company name, or "Unknown" if you cannot determine the company.
Do not include any explanation, just the company name."""

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        company = response.text.strip()
        
        if company.lower() == "unknown" or not company:
            return None
        return company
        
    except Exception as e:
        print(f"LLM company detection failed: {e}")
        return None

def get_all_companies(db=None):
    """Get all companies from all sources (company_manager + database)."""
    default_companies = ["NA", "HP", "Generic"]
    stored_companies = get_companies()
    
    db_companies = []
    if db is not None:
        try:
            all_data = db.get(include=["metadatas"])
            for m in all_data.get("metadatas", []):
                if m and isinstance(m, dict):
                    company_name = str(m.get("company", "")).strip()
                    if not company_name and isinstance(m.get("metadata"), dict):
                        company_name = str(m["metadata"].get("company", "")).strip()
                    if not company_name:
                        company_name = "NA"
                    db_companies.append(company_name)
        except:
            pass
    
    return sorted(list(set(default_companies + stored_companies + db_companies)))

def get_metadata_counts(db):
    """Get counts of entries grouped by company and topic."""
    company_counts = {}
    topic_counts = {}
    
    if db is not None:
        try:
            all_data = db.get(include=["metadatas"])
            for m in all_data.get("metadatas", []):
                if m and isinstance(m, dict):
                    company_name = str(m.get("company", "")).strip()
                    if not company_name and isinstance(m.get("metadata"), dict):
                        company_name = str(m["metadata"].get("company", "")).strip()
                    
                    topic_name = str(m.get("topic", "")).strip()
                    if not topic_name and isinstance(m.get("metadata"), dict):
                        topic_name = str(m["metadata"].get("topic", "")).strip()
                    
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
    if db is not None:
        try:
            all_data = db.get(include=["metadatas"])
            for m in all_data.get("metadatas", []):
                if m and isinstance(m, dict):
                    topic_name = str(m.get("topic", "")).strip()
                    if not topic_name and isinstance(m.get("metadata"), dict):
                        topic_name = str(m["metadata"].get("topic", "")).strip()
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
    
    # Load all entries from DB
    embeddings = load_embeddings()
    db = get_db(embeddings)
    
    if not db:
        return None, "Could not connect to database"
    
    all_data = db.get(include=["documents", "metadatas"])
    docs = all_data["documents"]
    metas = all_data["metadatas"]
    ids = all_data["ids"]
    
    # Filter by company and topic if specified
    filtered_docs = []
    filtered_metas = []
    filtered_ids = []
    skipped_count = 0
    
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        if meta is None:
            meta = {}
        
        # Get company from metadata
        company_name = ""
        if isinstance(meta, dict):
            if "company" in meta:
                company_name = str(meta.get("company", "")).strip()
            elif isinstance(meta.get("metadata"), dict) and "company" in meta["metadata"]:
                company_name = str(meta["metadata"].get("company", "")).strip()
        
        # Get topic from metadata
        topic_name = ""
        if isinstance(meta, dict):
            if "topic" in meta:
                topic_name = str(meta.get("topic", "")).strip()
            elif isinstance(meta.get("metadata"), dict) and "topic" in meta["metadata"]:
                topic_name = str(meta["metadata"].get("topic", "")).strip()
        
        # Check if already classified by LLM (supports both flat and nested metadata)
        already_classified = False
        if isinstance(meta, dict):
            if "llm_type" in meta and meta["llm_type"]:
                already_classified = True
            elif isinstance(meta.get("metadata"), dict) and meta["metadata"].get("llm_type"):
                already_classified = True
        
        # Check if this entry matches the filters
        company_match = (company_filter == "All Companies" or company_name == company_filter)
        topic_match = (topic_filter == "All Topics" or topic_name == topic_filter)
        
        if company_match and topic_match:
            if already_classified:
                skipped_count += 1
            else:
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_ids.append(ids[i])
    
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
            metadata=meta.get("metadata", {})
        )
        kb.add(entry)
    
    # Run LLM classification (batch mode for speed)
    try:
        from llm_classify import classify_batch_with_gemini
        
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
            
            results = classify_batch_with_gemini(batch_input)
            
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
        updated_metas = []
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
        
        ids_to_update = [e.id for e in kb.entries]
        texts_to_update = [e.to_chroma_text() for e in kb.entries]
        db.delete(ids=ids_to_update)
        db.add_texts(texts=texts_to_update, metadatas=updated_metas)
        
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

DB_DIRECTORY = "./my_local_database"
JSON_OUTPUT_DIR = "./knowledge_json_output"


@st.cache_resource(show_spinner="Loading embedding model (bge-small-en-v1.5)...")
def load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def get_db(embeddings):
    """Get or create Chroma database connection."""
    from langchain_chroma import Chroma
    if os.path.exists(DB_DIRECTORY):
        return Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)
    return None


def ingest_entries_to_db(entries, embeddings, source_label=""):
    """Vectorize entries and add to Chroma database."""
    from langchain_chroma import Chroma

    texts = [e.to_chroma_text() for e in entries]
    metas = [e.to_chroma_metadata() for e in entries]

    if os.path.exists(DB_DIRECTORY):
        # Append to existing database
        db = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)
        db.add_texts(texts=texts, metadatas=metas)
    else:
        # Create new database
        db = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metas,
            persist_directory=DB_DIRECTORY
        )
    return db, len(texts)


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
            import chromadb
            client = chromadb.PersistentClient(path=DB_DIRECTORY)
            cols = client.list_collections()
            count = sum(c.count() for c in cols) if cols else 0
            st.metric("Total entries in DB", count)
        except Exception:
            st.caption("DB exists")
    else:
        st.info("Database empty")

    st.markdown("---")
    chunk_size = st.slider("Chunk size (chars)", 400, 1200, 800, 50)
    st.caption("marker + LangChain + Chroma")


# --- Main ---

st.title("Knowledge Base System")
st.markdown("Ingest documents from multiple sources, vectorize, and search.")
st.markdown("---")

# Initialize database connection for unified company/topic retrieval
try:
    embeddings = load_embeddings()
    db = get_db(embeddings)
except:
    db = None

tab1, tab2, tab3, tab4 = st.tabs(["Ingest", "Database", "Browse", "Search"])

# Track tab changes to clear uploaded files when switching away from Ingest
current_tab = None
for key in st.session_state.keys():
    if key.startswith('tab_'):
        current_tab = st.session_state[key]
        break

# Check if we just switched away from Ingest tab (index 0)
if hasattr(st.session_state, 'last_tab') and st.session_state.last_tab == 0 and current_tab != 0:
    st.session_state.ingest_needs_reset = True

# Update last tab
st.session_state.last_tab = current_tab


# ========================================
# TAB 1: INGEST
# ========================================
with tab1:
    # Clear uploaded files if we just came back to Ingest tab
    if hasattr(st.session_state, 'ingest_needs_reset') and st.session_state.ingest_needs_reset:
        st.session_state.ingest_needs_reset = False
        # Clear file uploader state
        if 'json_uploads' in st.session_state:
            del st.session_state['json_uploads']
        st.rerun()
    
    # Show success message if ingestion was successful
    if hasattr(st.session_state, 'ingest_success_message') and st.session_state.ingest_success_message:
        st.success(st.session_state.ingest_success_message)
        st.session_state.ingest_success_message = None
    
    st.subheader("Data Ingestion")

    ingest_mode = st.radio(
        "Select source type",
        ["Upload PDF (marker parse)", "Upload Document (unstructured)", "Upload JSON (pre-parsed)", "Website URL"],
        horizontal=True
    )

    # Company and Topic selection (shared across all modes)
    # Pre-set company if there was a successful add from previous action
    if hasattr(st.session_state, 'pending_company_select') and st.session_state.pending_company_select:
        st.session_state.main_company_select = f"{st.session_state.pending_company_select} (0)"
        st.session_state.pending_company_select = None
    
    col_meta1, col_meta2 = st.columns(2)
    
    # Get counts for display
    company_counts, topic_counts = get_metadata_counts(db)
    
    with col_meta1:
        companies = get_all_companies(db if db else None)
        company_options = ["--- Select Company ---"]
        for c in companies:
            count = company_counts.get(c, 0)
            company_options.append(f"{c} ({count})")
        company_options.append("+ Add New Company")
        
        # Safely get index to avoid out of bounds
        default_company_idx = 0
        if hasattr(st.session_state, 'main_company_select'):
            try:
                default_company_idx = company_options.index(st.session_state.main_company_select)
            except ValueError:
                default_company_idx = 0
        
        selected_company_raw = st.selectbox("Select Company", company_options, index=default_company_idx, key="main_company_select")
        # Extract actual company name (without count)
        if selected_company_raw == "--- Select Company ---":
            selected_company = selected_company_raw
        elif selected_company_raw == "+ Add New Company":
            selected_company = selected_company_raw
        else:
            selected_company = selected_company_raw.split(" (")[0]
    
    with col_meta2:
        topic_options_base = ["TechDocs", "Books", "Academic", "Government", "Financial", "Office", "Email", "Images", "Data", "Demo", "Other"]
        topic_options = ["--- Select Topic ---"]
        for t in topic_options_base:
            count = topic_counts.get(t, 0)
            topic_options.append(f"{t} ({count})")
        
        # Safely get index to avoid out of bounds
        default_topic_idx = 0
        if hasattr(st.session_state, 'main_topic_select'):
            try:
                default_topic_idx = topic_options.index(st.session_state.main_topic_select)
            except ValueError:
                default_topic_idx = 0
        
        selected_topic_raw = st.selectbox("Select Topic", topic_options, index=default_topic_idx, key="main_topic_select")
        # Extract actual topic name (without count)
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
                    if add_company(new_company.strip()):
                        st.success(f"✅ Company '{new_company}' added successfully!")
                        st.session_state.companies_cache = []
                        st.session_state.pending_company_select = new_company.strip()
                    else:
                        st.warning("⚠️ Already exists or invalid")
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
                        embeddings = load_embeddings()
                        db, count = ingest_entries_to_db(kb.entries, embeddings, uploaded_pdf.name)

                        progress.progress(100, "Done")
                        st.success(f"Ingested {count} entries from `{uploaded_pdf.name}` into vector database.")

                        # Stats
                        tc = {}
                        for e in kb.entries:
                            tc[e.type] = tc.get(e.type, 0) + 1
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Entries", count)
                        col2.metric("SOP Steps", tc.get("sop_step", 0))
                        col3.metric("FAQ", tc.get("faq", 0))

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
            st.info(f"Company: **{selected_company}** | Topic: **{selected_topic}** (auto-detected: {auto_topic})")
            
            if selected_company == "--- Select Company ---":
                st.warning("⚠️ Please select a company before ingesting.")
            elif selected_topic == "--- Select Topic ---":
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
                        
                        kb = extract_knowledge_from_file(file_path, selected_company)
                        
                        if kb.metadata.get("encrypted"):
                            st.warning(f"⚠️ Skipping encrypted PDF: `{uploaded_doc.name}`")
                            progress.progress(100, "Done")
                        else:
                            for entry in kb.entries:
                                if not entry.metadata.get("topic"):
                                    entry.metadata["topic"] = selected_topic
                            
                            progress.progress(60, "Saving JSON...")
                            json_path = save_json_copy(kb, uploaded_doc.name)
                            
                            progress.progress(80, "Vectorizing and storing...")
                            embeddings = load_embeddings()
                            db, count = ingest_entries_to_db(kb.entries, embeddings, uploaded_doc.name)
                            
                            progress.progress(100, "Done")
                            st.success(f"Ingested {count} entries from `{uploaded_doc.name}` into vector database.")
                        
                        tc = {}
                        for e in kb.entries:
                            tc[e.type] = tc.get(e.type, 0) + 1
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Entries", count)
                        col2.metric("SOP Steps", tc.get("sop_step", 0))
                        col3.metric("FAQ", tc.get("faq", 0))
                        
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

        # Clear button
        if hasattr(st.session_state, 'uploaded_json_files') and st.session_state.uploaded_json_files:
            if st.button("🗑️ Clear Uploaded Files"):
                st.session_state.uploaded_json_files = []
                uploaded_jsons = []
        
        if uploaded_jsons:
            st.markdown(f"### 📁 {len(uploaded_jsons)} file(s) uploaded")
            st.caption("Adjust Company/Topic for each file below, or leave blank to use the default values at the top.")
            
            # Per-file configuration
            file_configs = []
            all_companies_list = get_all_companies(db if db else None)
            all_topics_list = ["TechDocs", "Books", "Academic", "Government", "Financial", "Office", "Email", "Images", "Data", "Demo", "Other"]
            
            for idx, uploaded_json in enumerate(uploaded_jsons):
                # Handle both file objects and persisted dictionaries
                try:
                    file_name = uploaded_json.name
                    file_data = uploaded_json.getvalue()
                    
                    data = json.loads(file_data.decode("utf-8"))
                    entries_data = data.get("entries", [])
                    source = data.get("source_file", file_name)
                    
                    # Auto-detect from JSON metadata
                    detected_company = ""
                    detected_topic = ""
                    if entries_data:
                        first_meta = entries_data[0].get("metadata", {})
                        detected_company = first_meta.get("company", "")
                        detected_topic = first_meta.get("topic", "")
                    
                    # Auto-detect from filename if not found in metadata
                    if not detected_company:
                        filename_detected = detect_company_from_filename(source)
                        if filename_detected:
                            detected_company = filename_detected
                    if not detected_topic:
                        topic_detected = detect_topic_from_filename(source)
                        if topic_detected:
                            detected_topic = topic_detected
                    
                    # Priority: JSON detected > global selection > empty
                    final_company = detected_company
                    if not final_company and selected_company != "--- Select Company ---":
                        final_company = selected_company
                    
                    final_topic = detected_topic
                    if not final_topic and selected_topic != "--- Select Topic ---":
                        final_topic = selected_topic
                    
                except Exception:
                    entries_data = []
                    source = file_name
                    final_company = selected_company if selected_company != "--- Select Company ---" else ""
                    final_topic = selected_topic if selected_topic != "--- Select Topic ---" else ""
                
                # Per-file row
                with st.container():
                    st.markdown(f"#### 📄 {file_name}")
                    
                    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
                    
                    with col_f1:
                        # Company selector with detected pre-fill
                        company_options_file = ["--- Select Company ---"] + all_companies_list
                        if final_company and final_company in company_options_file:
                            default_idx_file = company_options_file.index(final_company)
                        elif final_company:
                            # If detected company not in list, add it temporarily
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
                    
                    file_configs.append({
                        "name": file_name,
                        "data": data if 'data' in dir() else None,
                        "entries": entries_data,
                        "source": source,
                        "company": file_company,
                        "topic": file_topic,
                        "valid": file_company != "--- Select Company ---" and file_topic != "--- Select Topic ---"
                    })
                    
                    st.markdown("---")
            
            # Single button to ingest all configured files
            valid_count = sum(1 for fc in file_configs if fc["valid"])
            invalid_count = len(file_configs) - valid_count
            
            if invalid_count > 0:
                st.warning(f"⚠️ {invalid_count} file(s) missing Company/Topic. They will be skipped.")
            
            if st.button(f"🚀 Ingest {valid_count} file(s) to Database", type="primary", key="btn_json_batch", disabled=(valid_count==0)):
                progress = st.progress(0, "Initializing...")
                from knowledge_schema import KnowledgeBase, KnowledgeEntry
                embeddings = load_embeddings()
                db = get_db(embeddings)
                
                total_ingested = 0
                for i, fc in enumerate(file_configs):
                    if not fc["valid"]:
                        continue
                    
                    progress.progress(int((i / max(len(file_configs), 1)) * 100), f"Processing {fc['name']}...")
                    
                    try:
                        kb = KnowledgeBase(source_file=fc["source"])
                        for item in fc["entries"]:
                            entry = KnowledgeEntry(**item)
                            if not entry.metadata:
                                entry.metadata = {}
                            entry.metadata["company"] = fc["company"]
                            entry.metadata["topic"] = fc["topic"]
                            kb.add(entry)
                        
                        _, count = ingest_entries_to_db(kb.entries, embeddings, fc["source"])
                        save_json_copy(kb, fc["name"])
                        total_ingested += count
                        st.success(f"✅ {fc['name']}: {count} entries → {fc['company']}/{fc['topic']}")
                        
                    except Exception as ex:
                        st.error(f"❌ {fc['name']}: {ex}")
                
                progress.progress(100, "Done!")
                st.success(f"🎉 Total: {total_ingested} entries ingested into database!")
                
                st.session_state.ingest_success_message = f"✅ Successfully ingested {total_ingested} entries!"

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
                        embeddings = load_embeddings()
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
            embeddings = load_embeddings()
            db = get_db(embeddings)
            if db is None:
                st.warning("Could not connect to database.")
            else:
                # Get all metadata
                all_data = db.get(include=["metadatas"])
                total = len(all_data["ids"])

                st.metric("Total entries stored", total)
                st.markdown("---")

                # LLM Classification Button
                st.markdown("### 🤖 LLM Classification")
                st.markdown("Run **Gemini AI** classification on entries in the database.")
                st.markdown("This will analyze each entry and update its type based on AI understanding.")
                
                # Get all companies and topics from unified source
                llm_companies = get_all_companies(db)
                llm_topics = get_all_topics(db)
                
                # Get counts for display
                llm_company_counts, llm_topic_counts = get_metadata_counts(db)
                
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
                                    st.stop()
                                
                                progress.progress(0, f"Starting classification of {total_cnt} entries...")
                        else:
                            stats_generator = run_llm_classification_on_db(selected_llm_company, selected_llm_topic)
                            
                    except StopIteration:
                        pass
                    
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

                st.markdown("---")

                # Delete Data by Topic/Company
                st.markdown("### 🗑️ Delete Data from Database")
                st.markdown("Remove entries from the vector database by company and/or topic.")
                
                # Get companies and topics that actually exist in the database
                db_companies_in_data = get_all_companies(db)
                db_topics_in_data = get_all_topics(db)
                
                col_del1, col_del2 = st.columns(2)
                
                # Get counts for display
                del_company_counts, del_topic_counts = get_metadata_counts(db)
                
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
                
                # Preview count of entries that will be deleted
                # Get all data and filter in Python to handle NA (empty strings)
                all_preview_data = db.get(include=["metadatas"])
                preview_ids = all_preview_data["ids"]
                preview_metas = all_preview_data["metadatas"]
                
                # Apply filters manually
                filtered_preview_ids = []
                for i, meta in enumerate(preview_metas):
                    if not isinstance(meta, dict):
                        continue
                    
                    comp = str(meta.get("company", "")).strip()
                    if not comp and isinstance(meta.get("metadata"), dict):
                        comp = str(meta["metadata"].get("company", "")).strip()
                    
                    top = str(meta.get("topic", "")).strip()
                    if not top and isinstance(meta.get("metadata"), dict):
                        top = str(meta["metadata"].get("topic", "")).strip()
                    
                    company_match = (del_company == "All Companies") or \
                        (del_company == "NA" and (comp == "" or comp == "NA")) or \
                        (del_company == comp)
                    
                    topic_match = (del_topic == "All Topics") or \
                        (del_topic == "NA" and (top == "" or top == "NA")) or \
                        (del_topic == top)
                    
                    if company_match and topic_match:
                        filtered_preview_ids.append(preview_ids[i])
                
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
                                    db.delete(ids=filtered_preview_ids)
                                    
                                    st.success(f"✅ Successfully deleted {preview_count} entries!")
                                    st.rerun()
                            except Exception as del_ex:
                                st.error(f"❌ Deletion failed: {del_ex}")
                    with col_confirm2:
                        st.info("⚠️ Cannot be undone!")

                st.markdown("---")

                # Company Management Section
                st.markdown("### 🏢 Company Management")
                companies = get_companies()
                
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
                            if add_company(new_company.strip()):
                                st.success(f"✅ Added: {new_company}")
                                st.rerun()
                            else:
                                st.warning("⚠️ Already exists or invalid")
                
                if companies:
                    st.markdown("**Remove Company:**")
                    remove_company_name = st.selectbox("Select company to remove", companies, key="remove_company_select")
                    if st.button("Remove Company", type="secondary", key="btn_remove_company"):
                        if remove_company(remove_company_name):
                            st.success(f"✅ Removed: {remove_company_name}")
                            st.rerun()

                st.markdown("---")

                # Multi-dimensional Filter
                st.markdown("### 🔍 Filter by Company & Topic")
                st.markdown("Select filters to view specific data:")
                
                # Get all companies and topics from unified source with counts
                company_counts = {}
                topic_counts = {}
                for m in all_data["metadatas"]:
                    if isinstance(m, dict):
                        comp = m.get("company", "")
                        top = m.get("topic", "")
                        if not comp and isinstance(m.get("metadata"), dict):
                            comp = m["metadata"].get("company", "")
                            top = m["metadata"].get("topic", "")
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
                    # Safely get index to avoid out of bounds
                    default_company_idx = 0
                    if hasattr(st.session_state, 'filter_company_select'):
                        try:
                            default_company_idx = company_options_with_count.index(st.session_state.filter_company_select)
                        except ValueError:
                            default_company_idx = 0
                    selected_filter_company_raw = st.selectbox("Filter by company", company_options_with_count, index=default_company_idx, key="filter_company_select")
                    selected_filter_company = selected_filter_company_raw.split(" (")[0] if " (" in selected_filter_company_raw else selected_filter_company_raw
                with col_filter2:
                    # Safely get index to avoid out of bounds
                    default_topic_idx = 0
                    if hasattr(st.session_state, 'filter_topic_select'):
                        try:
                            default_topic_idx = topic_options_with_count.index(st.session_state.filter_topic_select)
                        except ValueError:
                            default_topic_idx = 0
                    selected_filter_topic_raw = st.selectbox("Filter by topic", topic_options_with_count, index=default_topic_idx, key="filter_topic_select")
                    selected_filter_topic = selected_filter_topic_raw.split(" (")[0] if " (" in selected_filter_topic_raw else selected_filter_topic_raw

                if total > 0:
                    metas = all_data["metadatas"]

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
                        # Track company distribution
                        company = m.get("company", "") or "unknown"
                        companies_in_db[company] = companies_in_db.get(company, 0) + 1
                        # Track topic distribution
                        topic = m.get("topic", "") or "unknown"
                        topics_in_db[topic] = topics_in_db.get(topic, 0) + 1
                        # Track LLM types if available
                        if isinstance(m.get("metadata"), dict):
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
                    
                    # Get all data including documents for detailed view
                    detailed_data = db.get(include=["metadatas", "documents"])
                    
                    # Apply both company and topic filters
                    if selected_filter_company != "All Companies" or selected_filter_topic != "All Topics":
                        filtered_ids = []
                        filtered_metadatas = []
                        filtered_docs = []
                        for i, m in enumerate(detailed_data.get("metadatas", [])):
                            if not isinstance(m, dict):
                                continue
                            
                            comp = str(m.get("company", "")).strip()
                            if not comp and isinstance(m.get("metadata"), dict):
                                comp = str(m["metadata"].get("company", "")).strip()
                            
                            top = str(m.get("topic", "")).strip()
                            if not top and isinstance(m.get("metadata"), dict):
                                top = str(m["metadata"].get("topic", "")).strip()
                            
                            company_match = (selected_filter_company == "All Companies") or \
                                (selected_filter_company == "NA" and (comp == "" or comp == "NA")) or \
                                (selected_filter_company == comp)
                            
                            topic_match = (selected_filter_topic == "All Topics") or \
                                (selected_filter_topic == "NA" and (top == "" or top == "NA")) or \
                                (selected_filter_topic == top)
                            
                            if company_match and topic_match:
                                filtered_ids.append(detailed_data["ids"][i])
                                filtered_metadatas.append(m)
                                filtered_docs.append(detailed_data["documents"][i])
                        
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
                                # Use chromadb API to delete collection (avoids file lock)
                                import chromadb
                                client = chromadb.PersistentClient(path=DB_DIRECTORY)
                                for col in client.list_collections():
                                    client.delete_collection(col.name)
                                del client
                                st.success("Database cleared. Refresh the page to see changes.")
                                st.cache_resource.clear()
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
            embeddings = load_embeddings()
            db = get_db(embeddings)
            if db:
                all_data = db.get(include=["documents", "metadatas"])
                total = len(all_data["ids"])

                if total == 0:
                    st.info("Database is empty.")
                else:
                    metas = all_data["metadatas"]
                    docs = all_data["documents"]

                    # Normalize None metadata
                    metas = [m if m is not None else {} for m in metas]
                    docs = [d if d is not None else "" for d in docs]

                    # Filters
                    col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns([1, 1, 1, 1, 2])

                    all_types = sorted(set(m.get("type", "general") for m in metas))
                    all_sources = sorted(set(m.get("source_file", "") for m in metas))
                    all_companies = get_all_companies(db)
                    all_topics = get_all_topics(db)
                    
                    # Get counts for display
                    browse_company_counts, browse_topic_counts = get_metadata_counts(db)

                    def has_llm_classification(meta):
                        if isinstance(meta, dict):
                            if "llm_type" in meta:
                                return meta["llm_type"] != ""
                            if "metadata" in meta and isinstance(meta["metadata"], dict):
                                return "llm_type" in meta["metadata"]
                        return False

                    col_f1, col_f2, col_f3, col_f4, col_f5, col_f6 = st.columns([1, 1, 1, 1, 1, 2])

                    with col_f1:
                        filter_type = st.selectbox("Filter by type", ["All"] + all_types, key="browse_type_select")
                    with col_f2:
                        # Company dropdown with counts
                        company_options_browse = ["All"]
                        for c in all_companies:
                            count = browse_company_counts.get(c, 0)
                            company_options_browse.append(f"{c} ({count})")
                        filter_company_raw = st.selectbox("Filter by company", company_options_browse, key="browse_company_select")
                        filter_company = "All" if filter_company_raw == "All" else filter_company_raw.split(" (")[0]
                    with col_f3:
                        # Topic dropdown with counts
                        topic_options_browse = ["All"]
                        for t in all_topics:
                            count = browse_topic_counts.get(t, 0)
                            topic_options_browse.append(f"{t} ({count})")
                        filter_topic_raw = st.selectbox("Filter by topic", topic_options_browse, key="browse_topic_select")
                        filter_topic = "All" if filter_topic_raw == "All" else filter_topic_raw.split(" (")[0]
                    with col_f4:
                        source_options = ["All"] + [s[:40] for s in all_sources]
                        filter_source = st.selectbox("Filter by source", source_options, key="browse_source_select")
                    with col_f5:
                        filter_llm = st.selectbox("LLM Classified", ["All", "Yes", "No"], key="browse_llm_select")
                    with col_f6:
                        search_kw = st.text_input("Keyword filter", placeholder="Enter keyword...")

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
# TAB 4: SEMANTIC SEARCH
# ========================================
with tab4:
    st.subheader("Semantic Search")

    if not os.path.exists(DB_DIRECTORY):
        st.info("No data yet. Ingest documents first.")
    else:
        try:
            embeddings = load_embeddings()
            db = get_db(embeddings)

            if db:
                # Get all companies from unified source
                all_companies = get_all_companies(db)
                
                # Get counts for display
                search_company_counts, _ = get_metadata_counts(db)

                col_q, col_k, col_c = st.columns([3, 1, 1])
                with col_q:
                    query = st.text_input("Enter your question", placeholder="e.g. How to reset the device?")
                with col_k:
                    top_k = st.slider("Results", 1, 10, 3)
                with col_c:
                    # Company dropdown with counts
                    search_company_options = ["All"]
                    for c in all_companies:
                        count = search_company_counts.get(c, 0)
                        search_company_options.append(f"{c} ({count})")
                    filter_company_raw = st.selectbox("Company filter", search_company_options, key="search_company_select")
                    filter_company = "All" if filter_company_raw == "All" else filter_company_raw.split(" (")[0]

                if query:
                    with st.spinner("Searching..."):
                        # Build filter if company is selected
                        filter_dict = {}
                        if filter_company != "All":
                            filter_dict["company"] = filter_company

                        if filter_dict:
                            results = db.similarity_search(query, k=top_k, filter=filter_dict)
                        else:
                            results = db.similarity_search(query, k=top_k)

                    st.markdown(f"**Query:** `{query}` | **Results:** {len(results)}")
                    st.markdown("---")

                    for i, res in enumerate(results):
                        meta = res.metadata
                        with st.container():
                            st.markdown(
                                f"**#{i+1}** | Type: `{meta.get('type', 'general')}` | "
                                f"Source: `{meta.get('source_file', '?')[:40]}` | Page {meta.get('source_page', 0)}"
                            )
                            st.markdown(f"**{meta.get('title', 'Untitled')}**")

                            # Clean display: strip [Title]/[Content]/[Keywords] tags
                            raw = res.page_content
                            display_text = raw
                            if "[Content] " in raw:
                                display_text = raw.split("[Content] ", 1)[1]
                                if "[Keywords]" in display_text:
                                    display_text = display_text.split("[Keywords]")[0].strip()
                            elif "[Title] " in raw:
                                display_text = raw.split("\n", 1)[1] if "\n" in raw else raw

                            st.info(display_text[:500])
                            st.markdown("---")

        except Exception as ex:
            st.error(f"Error: {ex}")
