import sqlite3
import os
import numpy as np
import json
import faiss

class SimpleVectorStore:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._load_embeddings()
    
    def _load_embeddings(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, embedding FROM langchain_embeddings;")
        rows = cursor.fetchall()
        
        if not rows:
            cursor.execute("SELECT id, embedding FROM embeddings;")
            rows = cursor.fetchall()
        
        self.embeddings = {}
        for row in rows:
            doc_id, emb_blob = row
            if isinstance(emb_blob, bytes):
                self.embeddings[doc_id] = np.frombuffer(emb_blob, dtype=np.float32)
        
        self.embedding_list = list(self.embeddings.values())
        self.id_list = list(self.embeddings.keys())
        
        if self.embedding_list:
            self.index = faiss.IndexFlatL2(len(self.embedding_list[0]))
            self.index.add(np.array(self.embedding_list))
            print(f"Loaded {len(self.embedding_list)} embeddings")
    
    def search(self, query_embedding, k=10):
        if not hasattr(self, 'index'):
            return []
        
        query = np.array([query_embedding]).astype(np.float32)
        distances, indices = self.index.search(query, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            doc_id = self.id_list[idx]
            results.append({
                'id': doc_id,
                'distance': float(distances[0][i]),
                'similarity': 1.0 / (1.0 + float(distances[0][i]))
            })
        
        return results
    
    def get_document(self, doc_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT document, metadata FROM langchain_documents WHERE id = ?;", (doc_id,))
        row = cursor.fetchone()
        if row:
            doc, meta = row
            return {
                'content': doc,
                'metadata': json.loads(meta) if isinstance(meta, str) else meta
            }
        return None
    
    def close(self):
        self.conn.close()

if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_local_database", "chroma.sqlite3")
    store = SimpleVectorStore(db_path)
    
    test_embedding = np.random.rand(384).tolist()
    results = store.search(test_embedding, k=3)
    
    print("\nSearch results:")
    for i, res in enumerate(results):
        doc = store.get_document(res['id'])
        if doc:
            title = doc['metadata'].get('title', 'Untitled')
            print(f"\n{i+1}. {title[:50]}")
            print(f"   Similarity: {res['similarity']:.4f}")
            print(f"   Content: {doc['content'][:200]}...")
    
    store.close()