import os
import pickle
import re
import uuid
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))

class BM25Retriever:
    def __init__(self, persist_dir=None):
        self.persist_dir = persist_dir or "./bm25_index"
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self.metadata = []
        self.tokenized_corpus = []
    
    def _tokenize(self, text):
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        return tokens
    
    def build_index(self, documents, doc_ids=None, metadata=None):
        self.documents = documents
        self.doc_ids = doc_ids or [str(uuid.uuid4()) for _ in range(len(documents))]
        self.metadata = metadata or [{}] * len(documents)
        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save_index()
    
    def _save_index(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        index_data = {
            'documents': self.documents,
            'doc_ids': self.doc_ids,
            'metadata': self.metadata,
            'tokenized_corpus': self.tokenized_corpus,
            'bm25': self.bm25
        }
        with open(os.path.join(self.persist_dir, "bm25_index.pkl"), "wb") as f:
            pickle.dump(index_data, f)
    
    def load_index(self):
        index_path = os.path.join(self.persist_dir, "bm25_index.pkl")
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                index_data = pickle.load(f)
                self.documents = index_data['documents']
                self.doc_ids = index_data['doc_ids']
                self.metadata = index_data['metadata']
                self.tokenized_corpus = index_data['tokenized_corpus']
                self.bm25 = index_data['bm25']
            return True
        return False
    
    def add_documents(self, documents, doc_ids=None, metadata=None):
        start_idx = len(self.documents)
        self.documents.extend(documents)
        new_ids = doc_ids or list(range(start_idx, start_idx + len(documents)))
        self.doc_ids.extend(new_ids)
        new_meta = metadata or [{}] * len(documents)
        self.metadata.extend(new_meta)
        
        new_tokenized = [self._tokenize(doc) for doc in documents]
        self.tokenized_corpus.extend(new_tokenized)
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save_index()
    
    def search(self, query, k=10, filter_dict=None):
        if not self.bm25:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        results = []
        for i, score in enumerate(scores):
            if filter_dict:
                meta = self.metadata[i]
                match = True
                for key, value in filter_dict.items():
                    if meta.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            results.append({
                'doc_id': self.doc_ids[i],
                'content': self.documents[i],
                'metadata': self.metadata[i],
                'score': float(score),
                'rank': -1
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        for idx, r in enumerate(results):
            r['rank'] = idx
        
        return results[:k]
    
    def get_index_size(self):
        return len(self.documents)
    
    def clear_index(self):
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self.metadata = []
        self.tokenized_corpus = []
        if os.path.exists(self.persist_dir):
            for f in os.listdir(self.persist_dir):
                os.remove(os.path.join(self.persist_dir, f))


def get_bm25_retriever(persist_dir=None):
    retriever = BM25Retriever(persist_dir)
    retriever.load_index()
    return retriever


def sync_bm25_with_chroma(db, persist_dir=None):
    try:
        all_docs = db.get()
        if not all_docs or len(all_docs['documents']) == 0:
            return None
        
        retriever = get_bm25_retriever(persist_dir)
        
        chroma_ids = set(all_docs['ids'])
        if retriever.get_index_size() > 0:
            existing_ids = set(retriever.doc_ids)
            new_ids = chroma_ids - existing_ids
            
            if new_ids:
                new_idx = [i for i, doc_id in enumerate(all_docs['ids']) if doc_id in new_ids]
                new_docs = [all_docs['documents'][i] for i in new_idx]
                new_ids_list = [all_docs['ids'][i] for i in new_idx]
                new_meta = [all_docs['metadatas'][i] for i in new_idx]
                retriever.add_documents(new_docs, new_ids_list, new_meta)
        else:
            retriever.build_index(
                documents=all_docs['documents'],
                doc_ids=all_docs['ids'],
                metadata=all_docs['metadatas']
            )
        
        return retriever
    except Exception as e:
        print(f"Error syncing BM25: {e}")
        return None
