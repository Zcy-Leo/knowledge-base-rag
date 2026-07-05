import os
import sys
import json
import sqlite3
import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "my_local_database", "chroma.sqlite3")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "my_local_database", "faiss_index.bin")
FAISS_INDEX_PQ_PATH = os.path.join(BASE_DIR, "my_local_database", "faiss_index_pq.bin")

class FAISSSearch:
    def __init__(self, use_compression=True, nlist=100, nprobe=10):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        model_path = "C:/Users/HP/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
        self.embeddings = HuggingFaceEmbeddings(model_name=model_path, model_kwargs={"device": "cpu"})
        self.index = None
        self.doc_ids = []
        self.docs = {}
        self.use_compression = use_compression
        self.nlist = nlist
        self.nprobe = nprobe
        self._load_or_build_index()
        self._rerankers = {}
        self._bm25_retriever = None
    
    def _load_or_build_index(self):
        pq_index_path = FAISS_INDEX_PQ_PATH if self.use_compression else FAISS_INDEX_PATH
        
        if os.path.exists(pq_index_path):
            print("Loading FAISS index from disk...")
            self.index = faiss.read_index(pq_index_path)
            
            if self.use_compression and hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe
            
            with open(pq_index_path + ".ids", "r", encoding="utf-8") as f:
                self.doc_ids = json.load(f)
            
            with open(pq_index_path + ".docs", "r", encoding="utf-8") as f:
                docs_list = json.load(f)
                self.docs = {doc_id: content for doc_id, content in docs_list}
            
            print(f"Loaded {len(self.doc_ids)} documents")
        else:
            print("Building FAISS index from SQLite...")
            self._build_from_sqlite()
    
    def _build_from_sqlite(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, c0 FROM embedding_fulltext_search_content;")
        rows = cursor.fetchall()
        conn.close()
        
        self.doc_ids = [str(r[0]) for r in rows]
        self.docs = {str(r[0]): r[1] for r in rows}
        
        print(f"Encoding {len(self.docs)} documents...")
        doc_embeddings = self.embeddings.embed_documents(list(self.docs.values()))
        doc_embeddings_np = np.array(doc_embeddings).astype(np.float32)
        
        if self.use_compression and len(self.docs) > 100:
            print("Using IVF+PQ compressed index for memory efficiency...")
            quantizer = faiss.IndexFlatL2(doc_embeddings_np.shape[1])
            self.index = faiss.IndexIVFPQ(quantizer, doc_embeddings_np.shape[1], 
                                          self.nlist, 16, 8)
            self.index.train(doc_embeddings_np)
            self.index.add(doc_embeddings_np)
            self.index.nprobe = self.nprobe
            save_path = FAISS_INDEX_PQ_PATH
        else:
            print("Using flat index (small dataset)...")
            self.index = faiss.IndexFlatL2(doc_embeddings_np.shape[1])
            self.index.add(doc_embeddings_np)
            save_path = FAISS_INDEX_PATH
        
        faiss.write_index(self.index, save_path)
        with open(save_path + ".ids", "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f)
        with open(save_path + ".docs", "w", encoding="utf-8") as f:
            docs_list = [[doc_id, content] for doc_id, content in self.docs.items()]
            json.dump(docs_list, f)
        
        print(f"Index built and saved with {len(self.doc_ids)} documents")
    
    def get_memory_usage(self):
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        return f"Current memory usage: {mem_mb:.2f} MB"
    
    def search(self, query, k=10, filter_dict=None):
        query_embedding = self.embeddings.embed_query(query)
        query_np = np.array([query_embedding]).astype(np.float32)
        
        distances, indices = self.index.search(query_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            doc_id = self.doc_ids[idx]
            content = self.docs.get(doc_id, "")
            metadata = self._parse_metadata(content)
            
            if filter_dict:
                match = True
                for key, value in filter_dict.items():
                    if metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            results.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "distance": float(distances[0][i]),
                "similarity": 1.0 / (1.0 + float(distances[0][i]))
            })
        
        return results
    
    def hybrid_search(self, query, k=10, filter_dict=None, use_bm25=True, reranker_type='crossencoder', reranker_model=None):
        candidate_k = k * 3
        
        vector_results = self.search(query, k=candidate_k, filter_dict=filter_dict)
        vector_results_formatted = []
        for rank, res in enumerate(vector_results):
            vector_results_formatted.append({
                'doc_id': res['id'],
                'content': res['content'],
                'metadata': res['metadata'],
                'vector_rank': rank,
                'similarity': res['similarity']
            })
        
        bm25_results = []
        if use_bm25:
            try:
                if self._bm25_retriever is None:
                    from bm25_retriever import BM25Retriever
                    self._bm25_retriever = BM25Retriever()
                    self._bm25_retriever.load_index()
                
                if self._bm25_retriever:
                    bm25_results = self._bm25_retriever.search(query, k=candidate_k, filter_dict=filter_dict)
            except Exception as e:
                print(f"BM25 error: {e}")
        
        fused = self._rrf_fusion(vector_results_formatted, bm25_results)
        
        if reranker_type:
            try:
                from reranker import get_reranker
                
                if reranker_type == 'gemini':
                    model_key = 'gemini'
                else:
                    model_key = reranker_model or 'cross-encoder/ms-marco-MiniLM-L-6-v2'
                
                if model_key not in self._rerankers:
                    self._rerankers[model_key] = get_reranker(model_id=model_key, device='cpu')
                    self._rerankers[model_key].initialize()
                
                reranked = self._rerankers[model_key].rerank(query, fused, top_k=k)
                return reranked
            except Exception as e:
                print(f"Rerank error ({reranker_type}): {e}. Using RRF results only.")
        
        return fused[:k]
    
    def _get_doc_key(self, res):
        doc_id = res.get('doc_id', '') or res.get('id', '')
        if doc_id:
            return str(doc_id)
        content = res.get('content', '')
        if content:
            import hashlib
            return hashlib.md5(content.encode('utf-8')).hexdigest()
        return str(id(res))
    
    def _get_content_key(self, content):
        import hashlib
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _rrf_fusion(self, vector_results, bm25_results, k=60):
        score_map = {}
        key_to_res = {}
        content_keys = set()
        
        all_results = []
        for rank, res in enumerate(vector_results):
            all_results.append({**res, 'source': 'vector', 'rank': rank})
        for rank, res in enumerate(bm25_results):
            all_results.append({**res, 'source': 'bm25', 'rank': rank})
        
        for res in all_results:
            content = res.get('content', '')
            content_key = self._get_content_key(content)
            
            if content_key in content_keys:
                continue
            content_keys.add(content_key)
            
            doc_key = res.get('doc_id', '') or res.get('id', '') or content_key
            source = res.get('source', '')
            rank = res.get('rank', -1)
            
            score_map[doc_key] = score_map.get(doc_key, 0) + 1 / (k + rank + 1)
            if doc_key not in key_to_res:
                key_to_res[doc_key] = {
                    'content': content,
                    'metadata': res.get('metadata', {}),
                }
            if source == 'vector':
                key_to_res[doc_key]['vector_rank'] = rank
            else:
                key_to_res[doc_key]['bm25_rank'] = rank
        
        results = []
        for doc_key, res in key_to_res.items():
            results.append({
                'doc_id': doc_key,
                'content': res['content'],
                'metadata': res['metadata'],
                'rrf_score': score_map.get(doc_key, 0),
                'vector_rank': res.get('vector_rank', -1),
                'bm25_rank': res.get('bm25_rank', -1)
            })
        
        results.sort(key=lambda x: x['rrf_score'], reverse=True)
        return results
    
    def _parse_metadata(self, content):
        metadata = {}
        lines = content.split("\n")
        for line in lines:
            if line.startswith("[Title]"):
                metadata["title"] = line.replace("[Title]", "").strip()
            elif line.startswith("[Keywords]"):
                metadata["keywords"] = line.replace("[Keywords]", "").strip()
            elif line.startswith("[Type]"):
                metadata["type"] = line.replace("[Type]", "").strip()
            elif line.startswith("[Company]"):
                metadata["company"] = line.replace("[Company]", "").strip()
        return metadata

if __name__ == "__main__":
    searcher = FAISSSearch()
    
    queries = ["how to reset password", "installation guide", "HP printer"]
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = searcher.search(query, k=3)
        for i, res in enumerate(results):
            print(f"  {i+1}. {res['metadata'].get('title', 'Untitled')}")
            print(f"     Similarity: {res['similarity']:.4f}")