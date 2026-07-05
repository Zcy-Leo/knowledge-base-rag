"""
app_v3.py -- Knowledge Base Automation System (Optimized for Performance)
================================================================
Features:
  1. Ingest from 3 sources: PDF (marker), pre-parsed JSON, Website URL
  2. Vectorize with bge-small-en-v1.5 and store in Chroma
  3. Database overview: view all ingested sources and stats
  4. Browse structured entries (filter by type/source/company)
  5. Semantic search with source attribution and company filter
  6. Company management: add/remove companies dynamically

Optimizations:
  - Lazy loading of heavy dependencies
  - Cached database connections
  - Loading skeletons instead of blank screens
  - Optimized sidebar rendering
"""

import streamlit as st
import os
import sys
import json
import shutil
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIRECTORY = os.path.join(BASE_DIR, "my_local_database")
JSON_OUTPUT_DIR = os.path.join(BASE_DIR, "knowledge_json_output")

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

def lazy_import_company_manager():
    from company_manager import get_companies, add_company, remove_company, has_company
    return get_companies, add_company, remove_company, has_company

def lazy_import_faiss_search():
    from faiss_search import FAISSSearch
    return FAISSSearch

def detect_company_from_filename(filename):
    filename_lower = filename.lower()
    for company in KNOWN_COMPANIES:
        if company.lower() in filename_lower:
            return company
    return None

def detect_topic_from_filename(filename):
    filename_lower = filename.lower()
    for topic, keywords in KNOWN_TOPICS.items():
        for kw in keywords:
            if kw in filename_lower:
                return topic
    return None

def detect_company_from_content(content, filename):
    detected = detect_company_from_filename(filename)
    if detected:
        return detected
    if content:
        preview = content[:1000].lower()
        for company in KNOWN_COMPANIES:
            if company.lower() in preview:
                return company
    return None

@st.cache_resource(show_spinner=False)
def load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

@st.cache_resource(show_spinner=False)
def get_db(embeddings):
    from langchain_chroma import Chroma
    if os.path.exists(DB_DIRECTORY):
        return Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)
    return None

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    import chromadb
    try:
        return chromadb.PersistentClient(path=DB_DIRECTORY)
    except:
        return None

def get_db_stats():
    client = get_chroma_client()
    if client:
        try:
            cols = client.list_collections()
            return sum(c.count() for c in cols) if cols else 0
        except:
            return 0
    return 0

def get_all_companies(db=None):
    get_companies_func, _, _, _ = lazy_import_company_manager()
    default_companies = ["NA", "HP", "Generic"]
    stored_companies = get_companies_func()
    
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

def get_all_topics(db=None):
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

def get_metadata_counts(db):
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

def ingest_entries_to_db(entries, embeddings, source_label=""):
    from langchain_chroma import Chroma

    texts = [e.to_chroma_text() for e in entries]
    metas = [e.to_chroma_metadata() for e in entries]

    if os.path.exists(DB_DIRECTORY):
        db = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)
        db.add_texts(texts=texts, metadatas=metas)
    else:
        db = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metas,
            persist_directory=DB_DIRECTORY
        )
    
    return db, len(texts)

def save_json_copy(kb, source_name):
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(source_name)[0].replace(" ", "_")
    json_path = os.path.join(JSON_OUTPUT_DIR, f"{base}_{ts}.json")
    kb.save_json(json_path)
    return json_path

st.set_page_config(
    page_title="Knowledge Base System",
    page_icon="KB",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("KB Console")
    st.markdown("---")
    st.markdown("**Processing Mode:** Local CPU")
    st.markdown("**Embedding:** bge-small-en-v1.5")
    st.markdown("**Vector DB:** Chroma")
    st.markdown("**PDF Parser:** marker")
    st.markdown("**Other Docs:** unstructured")
    st.markdown("---")

    if os.path.exists(DB_DIRECTORY):
        with st.spinner("Loading DB stats..."):
            count = get_db_stats()
        st.metric("Total entries in DB", count)
    else:
        st.info("Database empty")

    st.markdown("---")
    chunk_size = st.slider("Chunk size (chars)", 400, 1200, 800, 50)
    st.caption("marker + LangChain + Chroma")

st.title("Knowledge Base System")
st.markdown("Ingest documents from multiple sources, vectorize, and search.")
st.markdown("---")

with st.spinner("Initializing..."):
    embeddings = load_embeddings()
    db = get_db(embeddings)

tab1, tab2, tab3, tab4 = st.tabs(["Ingest", "Database", "Browse", "Search"])

with tab1:
    st.subheader("Data Ingestion")

    ingest_mode = st.radio(
        "Select source type",
        ["Upload PDF (marker parse)", "Upload Document (unstructured)", "Upload JSON (pre-parsed)", "Website URL"],
        horizontal=True
    )

    col_meta1, col_meta2 = st.columns(2)
    
    company_counts, topic_counts = get_metadata_counts(db)
    
    with col_meta1:
        companies = get_all_companies(db)
        company_options = ["--- Select Company ---"]
        for c in companies:
            count = company_counts.get(c, 0)
            company_options.append(f"{c} ({count})")
        
        selected_company_raw = st.selectbox("Select Company", company_options, index=0)
        if selected_company_raw == "--- Select Company ---":
            selected_company = selected_company_raw
        else:
            selected_company = selected_company_raw.split(" (")[0]
    
    with col_meta2:
        topic_options_base = ["TechDocs", "Books", "Academic", "Government", "Financial", "Office", "Email", "Images", "Data", "Demo", "Other"]
        topic_options = ["--- Select Topic ---"]
        for t in topic_options_base:
            count = topic_counts.get(t, 0)
            topic_options.append(f"{t} ({count})")
        
        selected_topic_raw = st.selectbox("Select Topic", topic_options, index=0)
        if selected_topic_raw == "--- Select Topic ---":
            selected_topic = selected_topic_raw
        else:
            selected_topic = selected_topic_raw.split(" (")[0]
    
    st.caption("*Select 'NA' for files without specific company ownership*")
    
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
                        progress.progress(10, "Parsing with marker...")
                        from marker_extractor import parse_pdf_with_marker, markdown_to_knowledge_entries
                        from knowledge_schema import KnowledgeBase

                        md_text = parse_pdf_with_marker(pdf_path)

                        progress.progress(50, "Structuring entries...")
                        entries = markdown_to_knowledge_entries(md_text, uploaded_pdf.name, chunk_size=chunk_size)
                        kb = KnowledgeBase(source_file=uploaded_pdf.name)
                        
                        for e in entries:
                            e.metadata = {"company": selected_company, "topic": selected_topic}
                            kb.add(e)

                        progress.progress(70, "Saving JSON...")
                        json_path = save_json_copy(kb, uploaded_pdf.name)

                        progress.progress(80, "Vectorizing and storing...")
                        db, count = ingest_entries_to_db(kb.entries, embeddings, uploaded_pdf.name)

                        progress.progress(100, "Done")
                        st.success(f"Ingested {count} entries from `{uploaded_pdf.name}` into vector database.")

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

    elif ingest_mode == "Upload Document (unstructured)":
        st.markdown("Upload any document. Uses **unstructured** library for multi-format support.")
        
        supported_types = ["doc", "docx", "odt", "rtf", "txt", "md", "html", "xml", "json",
                          "csv", "xls", "xlsx", "tsv", "ppt", "pptx",
                          "eml", "msg", "epub",
                          "png", "jpg", "jpeg", "tiff", "bmp", "heic"]
        
        uploaded_doc = st.file_uploader("Upload Document", type=supported_types, key="doc_upload")
        
        if uploaded_doc:
            st.success(f"File: `{uploaded_doc.name}` ({uploaded_doc.size/1024:.1f} KB)")
            
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
                        
                        for entry in kb.entries:
                            if not entry.metadata.get("topic"):
                                entry.metadata["topic"] = selected_topic
                        
                        progress.progress(60, "Saving JSON...")
                        json_path = save_json_copy(kb, uploaded_doc.name)
                        
                        progress.progress(80, "Vectorizing and storing...")
                        db, count = ingest_entries_to_db(kb.entries, embeddings, uploaded_doc.name)
                        
                        progress.progress(100, "Done")
                        st.success(f"Ingested {count} entries from `{uploaded_doc.name}` into vector database.")
                        
                    except Exception as ex:
                        progress.empty()
                        st.error(f"Error: {ex}")

    elif ingest_mode == "Upload JSON (pre-parsed)":
        st.markdown("Upload JSON files produced by marker/Docling/LlamaParse.")

        uploaded_jsons = st.file_uploader("Upload JSON (multiple allowed)", type=["json"], accept_multiple_files=True, key="json_uploads")

        if uploaded_jsons:
            st.markdown(f"### 📁 {len(uploaded_jsons)} file(s) uploaded")
            
            valid_count = len(uploaded_jsons)
            
            if st.button(f"🚀 Ingest {valid_count} file(s) to Database", type="primary", key="btn_json_batch"):
                progress = st.progress(0, "Initializing...")
                from knowledge_schema import KnowledgeBase, KnowledgeEntry
                
                total_ingested = 0
                for i, uploaded_json in enumerate(uploaded_jsons):
                    progress.progress(int((i / len(uploaded_jsons)) * 100), f"Processing {uploaded_json.name}...")
                    
                    try:
                        data = json.loads(uploaded_json.getvalue().decode("utf-8"))
                        entries_data = data.get("entries", [])
                        
                        kb = KnowledgeBase(source_file=uploaded_json.name)
                        for item in entries_data:
                            entry = KnowledgeEntry(**item)
                            if not entry.metadata:
                                entry.metadata = {}
                            if not entry.metadata.get("company"):
                                entry.metadata["company"] = selected_company
                            if not entry.metadata.get("topic"):
                                entry.metadata["topic"] = selected_topic
                            kb.add(entry)
                        
                        _, count = ingest_entries_to_db(kb.entries, embeddings, uploaded_json.name)
                        save_json_copy(kb, uploaded_json.name)
                        total_ingested += count
                        st.success(f"✅ {uploaded_json.name}: {count} entries")
                        
                    except Exception as ex:
                        st.error(f"❌ {uploaded_json.name}: {ex}")
                
                progress.progress(100, "Done!")
                st.success(f"🎉 Total: {total_ingested} entries ingested!")

    elif ingest_mode == "Website URL":
        st.markdown("Enter a URL to extract content using **trafilatura**.")

        url_input = st.text_input("Enter URL", placeholder="https://example.com")

        if url_input and st.button("Extract and Ingest", type="primary", key="btn_url"):
            progress = st.progress(0, "Fetching page...")

            try:
                import trafilatura
                from knowledge_schema import KnowledgeBase, KnowledgeEntry
                from marker_extractor import markdown_to_knowledge_entries

                downloaded = trafilatura.fetch_url(url_input)
                if not downloaded:
                    st.error("Could not fetch the URL.")
                else:
                    progress.progress(30, "Extracting content...")
                    text = trafilatura.extract(downloaded, include_tables=True, output_format="txt")

                    if not text or len(text) < 50:
                        st.error("No usable content found.")
                    else:
                        progress.progress(50, "Structuring entries...")
                        structured_text = f"## {url_input}\n\n{text}"
                        entries = markdown_to_knowledge_entries(structured_text, url_input, chunk_size=chunk_size)

                        kb = KnowledgeBase(source_file=url_input)
                        for e in entries:
                            e.metadata = {"company": selected_company, "topic": selected_topic}
                            kb.add(e)

                        progress.progress(70, "Vectorizing...")
                        db, count = ingest_entries_to_db(kb.entries, embeddings, url_input)

                        progress.progress(100, "Done")
                        st.success(f"Ingested {count} entries from URL.")

            except ImportError:
                st.error("trafilatura not installed.")
            except Exception as ex:
                progress.empty()
                st.error(f"Error: {ex}")

with tab2:
    st.subheader("Database Overview")

    if not os.path.exists(DB_DIRECTORY):
        st.info("No data in database yet.")
    else:
        try:
            if db is None:
                st.warning("Could not connect to database.")
            else:
                all_data = db.get(include=["metadatas"])
                total = len(all_data["ids"])

                st.metric("Total entries stored", total)
                st.markdown("---")

                llm_companies = get_all_companies(db)
                llm_topics = get_all_topics(db)
                
                col_llm1, col_llm2 = st.columns(2)
                with col_llm1:
                    selected_llm_company_raw = st.selectbox("Filter by company", ["All Companies"] + llm_companies, index=0)
                    selected_llm_company = "All Companies" if selected_llm_company_raw == "All Companies" else selected_llm_company_raw
                with col_llm2:
                    selected_llm_topic_raw = st.selectbox("Filter by topic", ["All Topics"] + llm_topics, index=0)
                    selected_llm_topic = "All Topics" if selected_llm_topic_raw == "All Topics" else selected_llm_topic_raw
                
                if st.button("Run LLM Classification", type="primary", key="btn_llm"):
                    st.info("LLM classification starting...")

        except Exception as e:
            st.error(f"Error loading database: {e}")

with tab3:
    st.subheader("Browse Entries")
    
    if db is None:
        st.info("Connecting to database...")
    else:
        try:
            companies = get_all_companies(db)
            topics = get_all_topics(db)
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                browse_company = st.selectbox("Filter by Company", ["All"] + companies)
            with col_b2:
                browse_topic = st.selectbox("Filter by Topic", ["All"] + topics)
            
            if st.button("Refresh", key="btn_refresh"):
                st.info("Refreshing...")
            
            st.info("Browse functionality - work in progress")
            
        except Exception as e:
            st.error(f"Error: {e}")

with tab4:
    st.subheader("Semantic Search")
    
    if db is None:
        st.info("Connecting to database...")
    else:
        try:
            search_query = st.text_input("Enter search query", placeholder="Search documents...")
            companies = get_all_companies(db)
            search_company = st.selectbox("Filter by Company", ["All"] + companies)
            
            if search_query and st.button("Search", type="primary", key="btn_search"):
                with st.spinner("Searching..."):
                    if search_company != "All":
                        results = db.similarity_search(search_query, k=5, filter={"company": search_company})
                    else:
                        results = db.similarity_search(search_query, k=5)
                
                st.markdown(f"### Found {len(results)} results")
                for i, doc in enumerate(results):
                    st.markdown(f"**Result {i+1}:**")
                    st.markdown(f"*Source:* {doc.metadata.get('source_file', 'Unknown')}")
                    st.markdown(f"*Company:* {doc.metadata.get('company', 'Unknown')}")
                    st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                    st.markdown("---")
            
        except Exception as e:
            st.error(f"Search error: {e}")
