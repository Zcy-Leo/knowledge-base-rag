# Knowledge Base RAG System

A Hybrid Retrieval-Augmented Generation (RAG) system for enterprise knowledge management, featuring FAISS vector search, BM25 keyword search, CrossEncoder re-ranking, and Gemini LLM integration.

## Features

### Data Ingestion
- **PDF Parsing**: Using Marker and PyMuPDF for high-quality text extraction
- **Document Parsing**: Support for Word, Excel, PowerPoint, Email, EPUB, and more
- **Website URL**: Extract content from web pages using trafilatura
- **JSON Import**: Pre-parsed knowledge entries in JSON format

### Search Modes
1. **Pure Vector Search**: FAISS-based semantic search
2. **Hybrid Search**: Combined vector + BM25 search with RRF fusion
3. **Hybrid + CrossEncoder**: Re-ranked results using CrossEncoder model
4. **Hybrid + Gemini**: Re-ranked results using Gemini LLM
5. **CrossEncoder + AI Assistant**: Full RAG pipeline with LLM QA

### System Architecture
```
Data Ingestion → Processing → Vectorization → ChromaDB Storage
                      ↓
                 Query → FAISS + BM25 → RRF Fusion → Re-ranking → LLM Generation
```

## Quick Start

### Prerequisites
- Python 3.9+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Zcy-Leo/knowledge-base-rag.git
   cd knowledge-base-rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app_v2.py
   ```

5. **Access the UI**
   Open your browser and navigate to `http://localhost:8501`

## Project Structure

```
knowledge-base-rag/
├── app_v2.py              # Main application (Streamlit UI)
├── app_v3.py              # Lightweight version (deprecated)
├── faiss_search.py        # FAISS vector search and hybrid retrieval
├── reranker.py            # CrossEncoder and Gemini re-ranking
├── llm_qa.py              # LLM QA pipeline (Gemini integration)
├── llm_classify.py        # LLM-based document classification
├── unified_extractor.py   # Unified document extraction (PDF, Word, etc.)
├── marker_extractor.py    # Marker PDF extraction
├── bm25_retriever.py      # BM25 keyword retrieval
├── knowledge_schema.py    # Knowledge entry data model
├── logger.py              # Logging utilities
├── requirements.txt       # Dependencies
├── .env.example           # Environment variable template
└── .gitignore             # Git ignore rules
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |

### Model Configuration

The system uses the following models (auto-downloaded on first run):
- **Embedding**: `bge-small-en-v1.5` (SentenceTransformers)
- **CrossEncoder**: `ms-marco-MiniLM-L-6-v2` (SentenceTransformers)
- **LLM**: `gemini-1.5-flash` (Google Gemini)

## Usage

### Ingesting Documents

1. Go to the **Ingest** tab
2. Select source type:
   - Upload PDF (marker parse)
   - Upload Document (unstructured)
   - Upload JSON (pre-parsed)
   - Website URL
3. Select company and topic (optional)
4. Click **Extract and Ingest**

### Searching

1. Go to the **Search** tab
2. Enter your query
3. Select search mode
4. Click **Search** or **Ask AI** (for RAG mode)

## Dependencies

| Category | Packages |
|----------|----------|
| Core | streamlit, chromadb, langchain, sentence-transformers |
| Search | faiss-cpu, rank-bm25 |
| ML | torch, transformers, scikit-learn |
| Parsing | PyMuPDF, python-docx, openpyxl, beautifulsoup4 |
| API | google-generativeai, requests |
| Web Scraping | trafilatura |

## License

This project is for academic purposes.

## Authors

- Zcy-Leo