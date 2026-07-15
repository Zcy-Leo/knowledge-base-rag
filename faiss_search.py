import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

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
    def __init__(self, use_compression=False, nlist=100, nprobe=10, embedding_model=None):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.embeddings = embedding_model
        self.index = None
        self.doc_ids = []
        self.docs = {}
        self._all_metadata = {}
        self.use_compression = use_compression
        self.nlist = nlist
        self.nprobe = nprobe
        self._rerankers = {}
        self._bm25_retriever = None
        self._initialized = False
    
    def _lazy_load_embeddings(self):
        if self.embeddings is None:
            try:
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_DATASETS_OFFLINE"] = "1"
                os.environ["HF_HUB_OFFLINE"] = "1"
                
                model_path = os.path.join(os.environ.get("USERPROFILE", "~"), ".cache", "huggingface", "hub", "models--BAAI--bge-small-en-v1.5", "snapshots")
                if os.path.exists(model_path):
                    import glob
                    snapshots = glob.glob(os.path.join(model_path, "*"))
                    if snapshots:
                        model_path = snapshots[0]
                
                if os.path.exists(model_path):
                    self.embeddings = HuggingFaceEmbeddings(model_name=model_path, model_kwargs={"device": "cpu"})
                else:
                    self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"})
                print("Embedding model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load embedding model: {str(e)}")
        return self.embeddings
    
    def initialize(self):
        if self._initialized:
            return
        
        try:
            self._lazy_load_embeddings()
            self._load_or_build_index()
            self._initialized = True
            print("FAISSSearch initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FAISSSearch: {str(e)}")
    
    def _load_or_build_index(self):
        pq_index_path = FAISS_INDEX_PQ_PATH if self.use_compression else FAISS_INDEX_PATH
        
        if os.path.exists(pq_index_path):
            print("Loading FAISS index from disk...")
            try:
                self.index = faiss.read_index(pq_index_path)
                
                if self.use_compression and hasattr(self.index, 'nprobe'):
                    self.index.nprobe = self.nprobe
                
                with open(pq_index_path + ".ids", "r", encoding="utf-8") as f:
                    self.doc_ids = json.load(f)
                
                with open(pq_index_path + ".docs", "r", encoding="utf-8") as f:
                    docs_list = json.load(f)
                    self.docs = {doc_id: content for doc_id, content in docs_list}
                
                self._load_all_metadata()
                print(f"Loaded {len(self.doc_ids)} documents")
                return
            except Exception as e:
                print(f"Failed to load PQ index: {e}")
        
        if os.path.exists(FAISS_INDEX_PATH):
            print("Trying flat index...")
            try:
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                
                with open(FAISS_INDEX_PATH + ".ids", "r", encoding="utf-8") as f:
                    self.doc_ids = json.load(f)
                
                with open(FAISS_INDEX_PATH + ".docs", "r", encoding="utf-8") as f:
                    docs_list = json.load(f)
                    self.docs = {doc_id: content for doc_id, content in docs_list}
                
                self._load_all_metadata()
                print(f"Loaded {len(self.doc_ids)} documents from flat index")
                return
            except Exception as e:
                print(f"Failed to load flat index: {e}")
        
        print("Building FAISS index from SQLite...")
        self._build_from_sqlite()
    
    def _load_all_metadata(self):
        print("Loading all metadata from SQLite...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, key, string_value, int_value, float_value FROM embedding_metadata;")
        meta_rows = cursor.fetchall()
        conn.close()
        
        self._all_metadata = {}
        for doc_id, key, string_val, int_val, float_val in meta_rows:
            doc_id_str = str(doc_id)
            if doc_id_str not in self._all_metadata:
                self._all_metadata[doc_id_str] = {}
            if string_val is not None:
                self._all_metadata[doc_id_str][key] = string_val
            elif int_val is not None:
                self._all_metadata[doc_id_str][key] = int_val
            elif float_val is not None:
                self._all_metadata[doc_id_str][key] = float_val
        print(f"Loaded metadata for {len(self._all_metadata)} documents")
    
    def _build_from_sqlite(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, c0 FROM embedding_fulltext_search_content;")
        rows = cursor.fetchall()
        
        self.doc_ids = [str(r[0]) for r in rows]
        self.docs = {str(r[0]): r[1] for r in rows}
        
        print("Loading all metadata from SQLite...")
        cursor.execute("SELECT id, key, string_value, int_value, float_value FROM embedding_metadata;")
        meta_rows = cursor.fetchall()
        conn.close()
        
        self._all_metadata = {}
        for doc_id, key, string_val, int_val, float_val in meta_rows:
            doc_id_str = str(doc_id)
            if doc_id_str not in self._all_metadata:
                self._all_metadata[doc_id_str] = {}
            if string_val is not None:
                self._all_metadata[doc_id_str][key] = string_val
            elif int_val is not None:
                self._all_metadata[doc_id_str][key] = int_val
            elif float_val is not None:
                self._all_metadata[doc_id_str][key] = float_val
        print(f"Loaded metadata for {len(self._all_metadata)} documents")
        
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
    
    def _get_metadata_from_db(self, doc_id):
        if doc_id in self._all_metadata:
            return self._all_metadata[doc_id]
        return None

    def search(self, query, k=10, filter_dict=None, timestamp=None):
        if timestamp:
            return self.temporal_query(query, timestamp=timestamp, k=k)
        
        query_embedding = self.embeddings.embed_query(query)
        query_np = np.array([query_embedding]).astype(np.float32)
        
        distances, indices = self.index.search(query_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            doc_id = self.doc_ids[idx]
            content = self.docs.get(doc_id, "")
            metadata = self._get_metadata_from_db(doc_id)
            
            if metadata is None:
                continue
            
            content_metadata = self._parse_metadata(content)
            metadata.update(content_metadata)
            
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
    
    def bm25_search(self, query, k=10, filter_dict=None):
        try:
            if self._bm25_retriever is None:
                from bm25_retriever import get_bm25_retriever
                persist_dir = os.path.join(BASE_DIR, "bm25_index")
                self._bm25_retriever = get_bm25_retriever(persist_dir)
            
            if self._bm25_retriever:
                return self._bm25_retriever.search(query, k=k, filter_dict=filter_dict)
        except Exception as e:
            print(f"BM25 search error: {e}")
        return []
    
    def hybrid_search(self, query, k=10, filter_dict=None, use_bm25=True, reranker_type='crossencoder', reranker_model=None, timestamp=None):
        candidate_k = k * 3
        
        vector_results = self.search(query, k=candidate_k, filter_dict=filter_dict, timestamp=timestamp)
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
                from llm_config import get_current_provider
                
                if reranker_type == 'llm':
                    model_key = get_current_provider()
                elif reranker_type == 'gemini':
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
            elif line.startswith("[Source]"):
                metadata["source_file"] = line.replace("[Source]", "").strip()
            elif line.startswith("[Page]"):
                try:
                    metadata["source_page"] = int(line.replace("[Page]", "").strip())
                except:
                    metadata["source_page"] = 0
        return metadata

    def temporal_query(self, query, timestamp=None, k=10):
        from chunk_change_detector import ChunkChangeDetector
        
        detector = ChunkChangeDetector()
        
        if timestamp is None:
            timestamp = "9999-12-31T23:59:59.999999"
        
        conn = detector._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT cv.chunk_id, cv.doc_id, cv.position, cv.content, cv.embedding,
                   cv.valid_from, cv.valid_to, cv.version_number
            FROM chunk_versions cv
            WHERE cv.valid_from <= ? AND (cv.valid_to IS NULL OR cv.valid_to > ?)
            AND cv.status = 'active'
            ORDER BY cv.doc_id, cv.position
        ''', (timestamp, timestamp))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        doc_contents = {}
        for row in rows:
            chunk_id, doc_id, position, content, embedding, valid_from, valid_to, version = row
            if doc_id not in doc_contents:
                doc_contents[doc_id] = []
            doc_contents[doc_id].append({
                'position': position,
                'content': content,
                'chunk_id': chunk_id
            })
        
        full_docs = {}
        for doc_id, chunks in doc_contents.items():
            chunks.sort(key=lambda x: x['position'])
            full_docs[doc_id] = '\n\n'.join(c['content'] for c in chunks)
        
        query_embedding = self.embeddings.embed_query(query)
        query_np = np.array([query_embedding]).astype(np.float32)
        
        rows_with_embeddings = [row for row in rows if row[4] is not None]
        rows_without_embeddings = [row for row in rows if row[4] is None]
        
        if rows_with_embeddings:
            embedding_dim = len(np.frombuffer(rows_with_embeddings[0][4], dtype=np.float32))
            temp_index = faiss.IndexFlatL2(embedding_dim)
            
            temp_doc_ids = []
            temp_embeddings = []
            
            for row in rows_with_embeddings:
                chunk_id, doc_id, position, content, embedding, valid_from, valid_to, version = row
                emb = np.frombuffer(embedding, dtype=np.float32)
                temp_embeddings.append(emb)
                temp_doc_ids.append(doc_id)
            
            temp_embeddings_np = np.array(temp_embeddings).astype(np.float32)
            temp_index.add(temp_embeddings_np)
            
            distances, indices = temp_index.search(query_np, k)
            
            results = []
            seen_docs = set()
            for i, idx in enumerate(indices[0]):
                doc_id = temp_doc_ids[idx]
                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)
                
                results.append({
                    'id': doc_id,
                    'content': full_docs.get(doc_id, ""),
                    'metadata': {
                        'valid_from': rows_with_embeddings[idx][5],
                        'valid_to': rows_with_embeddings[idx][6],
                        'version_number': rows_with_embeddings[idx][7],
                        'timestamp_query': timestamp
                    },
                    'distance': float(distances[0][i]),
                    'similarity': 1.0 / (1.0 + float(distances[0][i]))
                })
            
            return results
        else:
            temp_index = faiss.IndexFlatL2(len(query_embedding))
            
            temp_doc_ids = []
            temp_embeddings = []
            
            for row in rows_without_embeddings:
                chunk_id, doc_id, position, content, embedding, valid_from, valid_to, version = row
                emb = self.embeddings.embed_query(content)
                temp_embeddings.append(emb)
                temp_doc_ids.append(doc_id)
            
            if temp_embeddings:
                temp_embeddings_np = np.array(temp_embeddings).astype(np.float32)
                temp_index.add(temp_embeddings_np)
                
                distances, indices = temp_index.search(query_np, k)
                
                results = []
                seen_docs = set()
                for i, idx in enumerate(indices[0]):
                    doc_id = temp_doc_ids[idx]
                    if doc_id in seen_docs:
                        continue
                    seen_docs.add(doc_id)
                    
                    results.append({
                        'id': doc_id,
                        'content': full_docs.get(doc_id, ""),
                        'metadata': {
                            'valid_from': rows_without_embeddings[idx][5],
                            'valid_to': rows_without_embeddings[idx][6],
                            'version_number': rows_without_embeddings[idx][7],
                            'timestamp_query': timestamp
                        },
                        'distance': float(distances[0][i]),
                        'similarity': 1.0 / (1.0 + float(distances[0][i]))
                    })
                
                return results
        
        return []
    
    def rebuild_index_incremental(self):
        from chunk_change_detector import ChunkChangeDetector
        
        detector = ChunkChangeDetector()
        
        conn = detector._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT cv.chunk_id, cv.doc_id, cv.position, cv.content, cv.embedding
            FROM chunk_versions cv
            WHERE cv.status = 'active'
            ORDER BY cv.doc_id, cv.position
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            print("No active chunks found")
            return
        
        doc_contents = {}
        for row in rows:
            chunk_id, doc_id, position, content, embedding = row
            if doc_id not in doc_contents:
                doc_contents[doc_id] = {'chunks': [], 'embeddings': []}
            doc_contents[doc_id]['chunks'].append(content)
            if embedding:
                doc_contents[doc_id]['embeddings'].append(np.frombuffer(embedding, dtype=np.float32))
        
        docs_to_embed = []
        doc_ids_list = []
        
        for doc_id, data in doc_contents.items():
            full_content = '\n\n'.join(data['chunks'])
            docs_to_embed.append(full_content)
            doc_ids_list.append(doc_id)
        
        print(f"Embedding {len(docs_to_embed)} documents...")
        doc_embeddings = self.embeddings.embed_documents(docs_to_embed)
        doc_embeddings_np = np.array(doc_embeddings).astype(np.float32)
        
        if self.use_compression and len(docs_to_embed) > 100:
            print("Using IVF+PQ compressed index...")
            quantizer = faiss.IndexFlatL2(doc_embeddings_np.shape[1])
            self.index = faiss.IndexIVFPQ(quantizer, doc_embeddings_np.shape[1], 
                                          self.nlist, 16, 8)
            self.index.train(doc_embeddings_np)
            self.index.add(doc_embeddings_np)
            self.index.nprobe = self.nprobe
            save_path = FAISS_INDEX_PQ_PATH
        else:
            print("Using flat index...")
            self.index = faiss.IndexFlatL2(doc_embeddings_np.shape[1])
            self.index.add(doc_embeddings_np)
            save_path = FAISS_INDEX_PATH
        
        faiss.write_index(self.index, save_path)
        with open(save_path + ".ids", "w", encoding="utf-8") as f:
            json.dump(doc_ids_list, f)
        with open(save_path + ".docs", "w", encoding="utf-8") as f:
            docs_list = [[doc_id, content] for doc_id, content in zip(doc_ids_list, docs_to_embed)]
            json.dump(docs_list, f)
        
        self.doc_ids = doc_ids_list
        self.docs = {doc_id: content for doc_id, content in zip(doc_ids_list, docs_to_embed)}
        
        print(f"Incremental index rebuilt with {len(self.doc_ids)} documents")


if __name__ == "__main__":
    searcher = FAISSSearch()
    searcher.initialize()
    
    queries = ["how to reset password", "installation guide", "HP printer"]
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = searcher.search(query, k=3)
        for i, res in enumerate(results):
            print(f"  {i+1}. {res['metadata'].get('title', 'Untitled')}")
            print(f"     Similarity: {res['similarity']:.4f}")