import os
import time
import requests
from typing import List, Dict, Optional, Generator
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

session = requests.Session()
session.verify = False

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_K = 5


def _clean_content(content: str) -> str:
    if "[Content] " in content:
        content = content.split("[Content] ", 1)[1]
    if "[Keywords]" in content:
        content = content.split("[Keywords]")[0].strip()
    lines = content.split("\n")
    clean_lines = []
    for line in lines:
        if not line.startswith("["):
            clean_lines.append(line)
    return "\n".join(clean_lines).strip()

def build_rag_prompt(query: str, search_results: List[Dict], top_k: int = DEFAULT_TOP_K) -> str:
    context_parts = []
    for i, result in enumerate(search_results[:top_k]):
        raw_content = result.get('content', '')[:2000]
        content = _clean_content(raw_content)
        source = result.get('metadata', {}).get('source_file', 'Unknown')
        context_parts.append(f"[Document {i+1}]({source}):\n{content}\n")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a helpful research assistant. Answer the user's question based on the provided context documents.

Question: {query}

Context:
{context}

Instructions:
- Answer only based on the information in the context documents
- If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."
- Provide citations by referencing the document numbers in square brackets
- Keep the answer concise and well-structured
- Use markdown format for better readability

Answer:"""
    
    return prompt


def generate_answer(query: str, search_results: List[Dict], 
                    max_tokens: int = DEFAULT_MAX_TOKENS, 
                    temperature: float = DEFAULT_TEMPERATURE,
                    top_k: int = DEFAULT_TOP_K) -> Dict:
    if not search_results:
        return {
            'answer': "No relevant documents found.",
            'sources': [],
            'latency': 0,
            'success': True
        }
    
    if not GEMINI_API_KEY:
        return {
            'answer': "Gemini API key not configured. Please set GEMINI_API_KEY environment variable.",
            'sources': [],
            'latency': 0,
            'success': False
        }
    
    prompt = build_rag_prompt(query, search_results, top_k)
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "topP": 0.9
        }
    }
    
    start_time = time.time()
    
    try:
        response = session.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        answer = data['candidates'][0]['content']['parts'][0]['text']
        
        latency = time.time() - start_time
        
        sources = [
            {
                'document': i + 1,
                'source': result.get('metadata', {}).get('source_file', 'Unknown'),
                'relevance': result.get('rerank_score', result.get('rrf_score', result.get('similarity', 0)))
            }
            for i, result in enumerate(search_results[:top_k])
        ]
        
        return {
            'answer': answer,
            'sources': sources,
            'latency': round(latency, 2),
            'success': True
        }
    
    except requests.exceptions.RequestException as e:
        latency = time.time() - start_time
        return {
            'answer': f"LLM API error: {str(e)}",
            'sources': [],
            'latency': round(latency, 2),
            'success': False
        }


def generate_answer_streaming(query: str, search_results: List[Dict],
                              max_tokens: int = DEFAULT_MAX_TOKENS,
                              temperature: float = DEFAULT_TEMPERATURE,
                              top_k: int = DEFAULT_TOP_K) -> Generator[str, None, None]:
    if not search_results:
        yield "No relevant documents found."
        return
    
    if not GEMINI_API_KEY:
        yield "Gemini API key not configured."
        return
    
    prompt = build_rag_prompt(query, search_results, top_k)
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "topP": 0.9
        },
        "stream": True
    }
    
    try:
        response = session.post(API_URL, json=payload, timeout=120, stream=True)
        response.raise_for_status()
        
        buffer = ""
        for line in response.iter_lines():
            if line:
                try:
                    import json
                    data = json.loads(line.decode('utf-8').replace('data: ', ''))
                    if 'candidates' in data and data['candidates']:
                        parts = data['candidates'][0].get('content', {}).get('parts', [])
                        if parts:
                            text = parts[0].get('text', '')
                            if text:
                                buffer += text
                                if len(buffer) >= 50 or any(p in buffer for p in ['.', '!', '?', '\n']):
                                    yield buffer
                                    buffer = ""
                except (json.JSONDecodeError, KeyError):
                    continue
        
        if buffer:
            yield buffer
    
    except requests.exceptions.RequestException as e:
        yield f"LLM API error: {str(e)}"


def extract_citations(answer: str) -> List[str]:
    import re
    citations = re.findall(r'\[(\d+)\]', answer)
    return sorted(set(citations))


if __name__ == "__main__":
    test_results = [
        {
            'content': "Machine learning is a subset of artificial intelligence that uses algorithms to learn from data.",
            'source': "ml_intro.pdf",
            'score': 0.95
        },
        {
            'content': "Deep learning is a type of machine learning that uses neural networks with multiple layers.",
            'source': "deep_learning.pdf",
            'score': 0.88
        }
    ]
    
    print("=== Non-streaming test ===")
    result = generate_answer("What is deep learning?", test_results)
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Latency: {result['latency']}s")
    
    print("\n=== Streaming test ===")
    for chunk in generate_answer_streaming("What is machine learning?", test_results):
        print(chunk, end='', flush=True)
    print()