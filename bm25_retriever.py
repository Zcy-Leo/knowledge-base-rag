import os
import pickle
import re
import uuid

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    try:
        STOPWORDS = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        STOPWORDS = set(stopwords.words('english'))
except Exception:
    import re
    STOPWORDS = set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'but', 'and', 'if', 'or', 'because', 'until', 'while', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'])
    
    def word_tokenize(text):
        return re.findall(r'\w+', text.lower())

from rank_bm25 import BM25Okapi

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


def sync_bm25_with_chroma(db_or_reader, persist_dir=None):
    try:
        if hasattr(db_or_reader, 'get_documents_with_metadatas'):
            all_docs_dict = db_or_reader.get_documents_with_metadatas()
            all_data = {
                'ids': list(all_docs_dict.keys()),
                'documents': [all_docs_dict[id].get('content', '') for id in all_docs_dict.keys()],
                'metadatas': [all_docs_dict[id].get('metadata', {}) for id in all_docs_dict.keys()]
            }
        else:
            all_data = db_or_reader.get()
        
        if not all_data or len(all_data['documents']) == 0:
            return None
        
        retriever = get_bm25_retriever(persist_dir)
        
        chroma_ids = set(all_data['ids'])
        if retriever.get_index_size() > 0:
            existing_ids = set(retriever.doc_ids)
            new_ids = chroma_ids - existing_ids
            
            if new_ids:
                new_idx = [i for i, doc_id in enumerate(all_data['ids']) if doc_id in new_ids]
                new_docs = [all_data['documents'][i] for i in new_idx]
                new_ids_list = [all_data['ids'][i] for i in new_idx]
                new_meta = [all_data['metadatas'][i] for i in new_idx]
                retriever.add_documents(new_docs, new_ids_list, new_meta)
        else:
            retriever.build_index(
                documents=all_data['documents'],
                doc_ids=all_data['ids'],
                metadata=all_data['metadatas']
            )
        
        return retriever
    except Exception as e:
        print(f"Error syncing BM25: {e}")
        return None
