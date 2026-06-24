from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("Connecting to local knowledge base...\n")

# 1. Load the same embedding model used during ingestion
# Must match the model used when storing data, otherwise vectors won't align
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 2. Specify the database directory path
db_directory = "./my_local_database"

# 3. Connect to database (no need to reload the PDF here)
db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
print("Database connected. Ready to query.\n")

# ==========================================
# Core: Query the knowledge base
# ==========================================
query = "How to reset the device to factory settings?"

print(f"Query: {query}")
print("Searching for relevant passages...\n")

# Similarity search, k=3 means return top 3 most relevant chunks
results = db.similarity_search(query, k=3)

# Print results
for i, res in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(res.page_content)
    print("\n")