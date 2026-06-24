from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("Starting automated data ingestion pipeline...\n")

# ==========================================
# Step 1: Read PDF
# ==========================================
print("[Step 1/4] Reading PDF file...")
pdf_path = "sample_manual.pdf"
loader = PyMuPDFLoader(pdf_path)
docs = loader.load()
print(f"   Done. Found {len(docs)} pages.\n")

# ==========================================
# Step 2: Text Chunking
# ==========================================
print("[Step 2/4] Splitting text into chunks...")
# Each chunk max 500 chars, with 50 char overlap to preserve sentence continuity
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
print(f"   Done. Document split into {len(chunks)} chunks.\n")

# ==========================================
# Step 3: Embedding
# ==========================================
print("[Step 3/4] Loading embedding model...")
print("   (Note: first run will download the model from HuggingFace, may take a minute)")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
print("   Model loaded.\n")

# ==========================================
# Step 4: Store in Database
# ==========================================
print("[Step 4/4] Storing vectors in local Chroma database...")
# persist_directory specifies where the database files are saved locally
db_directory = "./my_local_database"
db = Chroma.from_documents(chunks, embeddings, persist_directory=db_directory)
print("   Data stored successfully.\n")

print("Pipeline complete. Check the 'my_local_database' folder in the project directory.")