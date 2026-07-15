import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import pickle
import json
import time
import requests
from typing import List, Dict, Optional, Union
from dotenv import load_dotenv
from llm_provider import LLMProvider, call_llm

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path, override=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.5-flash"

session = requests.Session()
session.verify = False

MODEL_REGISTRY = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {
        "name": "MS MARCO MiniLM L6",
        "type": "crossencoder",
        "language": "english",
        "description": "Lightweight English reranker, fast",
        "model_class": "CrossEncoderReranker"
    },
    "cross-encoder/ms-marco-MiniLM-L-12-v2": {
        "name": "MS MARCO MiniLM L12",
        "type": "crossencoder",
        "language": "english",
        "description": "Medium English reranker, higher accuracy",
        "model_class": "CrossEncoderReranker"
    },
    "cross-encoder/ms-marco-MultiBERT-L-12": {
        "name": "MS MARCO MultiBERT L12",
        "type": "crossencoder",
        "language": "english",
        "description": "Large English reranker, best accuracy",
        "model_class": "CrossEncoderReranker"
    },
    "gemini": {
        "name": "Gemini 2.5 Flash",
        "type": "llm",
        "language": "english",
        "description": "Cloud LLM reranker, requires internet",
        "model_class": "GeminiReranker"
    },
    "deepseek": {
        "name": "DeepSeek",
        "type": "llm",
        "language": "english",
        "description": "Domestic LLM reranker, free tier available",
        "model_class": "LLMReranker"
    },
    "zhipu": {
        "name": "ZhiPu GLM",
        "type": "llm",
        "language": "english",
        "description": "Domestic LLM reranker, free tier available",
        "model_class": "LLMReranker"
    },
    "qianwen": {
        "name": "Aliyun Qianwen",
        "type": "llm",
        "language": "english",
        "description": "Domestic LLM reranker, free tier available",
        "model_class": "LLMReranker"
    },
    "doubao": {
        "name": "ByteDance Doubao",
        "type": "llm",
        "language": "english",
        "description": "Domestic LLM reranker, free tier available",
        "model_class": "LLMReranker"
    }
}

MODEL_CACHE = {}

def get_available_models(language: str = None) -> List[Dict]:
    models = []
    for model_id, info in MODEL_REGISTRY.items():
        if language is None or info["language"] in [language, "multilingual"]:
            models.append({
                "id": model_id,
                "name": info["name"],
                "type": info["type"],
                "language": info["language"],
                "description": info["description"]
            })
    return models

class BaseReranker:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_info = MODEL_REGISTRY.get(model_id, {})
        self._initialized = False
    
    def initialize(self):
        raise NotImplementedError
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 10) -> List[Dict]:
        raise NotImplementedError
    
    def predict_scores(self, query: str, documents: List[Dict]) -> List[float]:
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        return self.model_info.get("name", self.model_id)
    
    @property
    def language(self) -> str:
        return self.model_info.get("language", "english")

class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        super().__init__(model_id)
        self.device = device
        self.model = None
        self._model_name = model_id
    
    def initialize(self):
        if not self._initialized:
            try:
                import os
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_DATASETS_OFFLINE"] = "1"
                os.environ["HF_HUB_OFFLINE"] = "1"
                from sentence_transformers import CrossEncoder
                print(f"Loading CrossEncoder model: {self._model_name} (device: {self.device})")
                start_time = time.time()
                self.model = CrossEncoder(self._model_name, device=self.device)
                load_time = time.time() - start_time
                print(f"Model loaded in {load_time:.2f}s")
                self._initialized = True
            except ImportError:
                raise RuntimeError("sentence_transformers is not installed. Install with: pip install sentence-transformers")
            except Exception as e:
                raise RuntimeError(f"Failed to load CrossEncoder model '{self._model_name}': {str(e)}. Make sure the model name is correct.")
        return self.model
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 10) -> List[Dict]:
        if not documents:
            return []
        
        if not query or not query.strip():
            return documents[:top_k]
        
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            pairs = [(query, doc.get('content', '')) for doc in documents]
            scores = self.model.predict(pairs)
            
            for i, doc in enumerate(documents):
                doc['rerank_score'] = float(scores[i])
            
            ranked = sorted(documents, key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            rerank_time = time.time() - start_time
            print(f"Reranked {len(documents)} documents in {rerank_time:.2f}s")
            
            return ranked[:top_k]
        except Exception as e:
            print(f"CrossEncoder rerank error: {str(e)}. Returning original order.")
            return documents[:top_k]
    
    def predict_scores(self, query: str, documents: List[Dict]) -> List[float]:
        if not self._initialized:
            self.initialize()
        
        pairs = [(query, doc.get('content', '')) for doc in documents]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]

class LLMReranker(BaseReranker):
    def __init__(self, model_id: str = None):
        from llm_config import get_current_provider
        super().__init__(model_id or get_current_provider())
        self.provider = model_id or get_current_provider()
        self._initialized = True
        self.last_call_time = 0
    
    def initialize(self):
        return self
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 10) -> List[Dict]:
        if not documents:
            return []
        
        start_time = time.time()
        
        docs_text = ""
        for i, doc in enumerate(documents):
            content = doc.get('content', '')[:500]
            docs_text += f"\nDocument {i+1}:\n{content}\n"
        
        prompt = f"""Given the query, rank the documents from most relevant to least relevant. Return ONLY a JSON array of document numbers (1-based).

Query: {query}

Documents:{docs_text}

Return: [most_relevant_doc_num, next_doc_num, ...]"""
        
        try:
            text = call_llm(prompt, max_tokens=512, response_format="json")
            
            if text:
                text = text.strip()
                if text.startswith('```json'):
                    text = text.replace('```json', '').replace('```', '')
                elif text.startswith('```'):
                    text = text.replace('```', '')
                
                ranked_indices = json.loads(text)
                
                reranked_docs = []
                for position, idx in enumerate(ranked_indices):
                    if 1 <= idx <= len(documents):
                        doc = documents[idx - 1]
                        doc['rerank_score'] = float(len(documents) - position)
                        reranked_docs.append(doc)
                
                rerank_time = time.time() - start_time
                print(f"{self.provider.upper()} reranked {len(documents)} documents in {rerank_time:.2f}s")
                
                return reranked_docs[:top_k]
        except Exception as e:
            print(f"{self.provider.upper()} rerank error: {e}")
        
        return sorted(documents, key=lambda x: x.get('rrf_score', 0), reverse=True)[:top_k]
    
    def predict_scores(self, query: str, documents: List[Dict]) -> List[float]:
        reranked = self.rerank(query, documents, top_k=len(documents))
        scores = [doc.get('rerank_score', 0.0) for doc in documents]
        return scores

class GeminiReranker(LLMReranker):
    def __init__(self, model_id: str = "gemini", api_key: Optional[str] = None, model_name: Optional[str] = None):
        super().__init__(model_id="gemini")

def get_reranker(model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", use_gemini: bool = False, device: str = "cpu") -> BaseReranker:
    if use_gemini:
        model_id = "gemini"
    
    if model_id in MODEL_CACHE:
        print(f"Using cached reranker: {model_id}")
        return MODEL_CACHE[model_id]
    
    model_info = MODEL_REGISTRY.get(model_id)
    if not model_info:
        raise ValueError(f"Unknown model: {model_id}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = model_info["model_class"]
    if model_class == "CrossEncoderReranker":
        reranker = CrossEncoderReranker(model_id=model_id, device=device)
    elif model_class == "GeminiReranker":
        reranker = GeminiReranker(model_id=model_id)
    elif model_class == "LLMReranker":
        reranker = LLMReranker(model_id=model_id)
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    MODEL_CACHE[model_id] = reranker
    return reranker

def rrf_fusion(vector_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
    score_map = {}
    key_to_res = {}
    
    for rank, res in enumerate(vector_results):
        doc_key = res.get('doc_id', res.get('id', str(rank)))
        score_map[doc_key] = score_map.get(doc_key, 0) + 1 / (k + rank + 1)
        if doc_key not in key_to_res:
            key_to_res[doc_key] = {
                'doc_id': doc_key,
                'content': res.get('content', res.get('page_content', '')),
                'metadata': res.get('metadata', {}),
                'vector_rank': rank
            }
    
    for rank, res in enumerate(bm25_results):
        doc_key = res.get('doc_id', res.get('id', str(rank)))
        score_map[doc_key] = score_map.get(doc_key, 0) + 1 / (k + rank + 1)
        if doc_key not in key_to_res:
            key_to_res[doc_key] = {
                'doc_id': doc_key,
                'content': res.get('content', res.get('page_content', '')),
                'metadata': res.get('metadata', {}),
                'bm25_rank': rank
            }
    
    results = []
    for doc_key, res in key_to_res.items():
        results.append({
            **res,
            'rrf_score': score_map.get(doc_key, 0)
        })
    
    results.sort(key=lambda x: x['rrf_score'], reverse=True)
    return results

def hybrid_search(db, bm25_retriever, query: str, top_k: int = 10, filter_dict: Optional[Dict] = None, 
                  use_rerank: bool = False, reranker: Optional[BaseReranker] = None) -> List[Dict]:
    candidate_k = top_k * 3
    
    vector_results_raw = db.similarity_search(query, k=candidate_k, filter=filter_dict) if filter_dict else db.similarity_search(query, k=candidate_k)
    
    vector_results = []
    for i, res in enumerate(vector_results_raw):
        meta = res.metadata
        doc_id = meta.get('id', meta.get('source_file', '') + str(meta.get('source_page', '')) + str(i))
        if not doc_id:
            doc_id = str(i)
        vector_results.append({
            'doc_id': doc_id,
            'content': res.page_content,
            'metadata': meta,
            'vector_rank': i
        })
    
    bm25_results = bm25_retriever.search(query, k=candidate_k, filter_dict=filter_dict)
    
    fused = rrf_fusion(vector_results, bm25_results)
    
    if use_rerank and reranker:
        reranked = reranker.rerank(query, fused, top_k=top_k)
        return reranked
    
    return fused[:top_k]