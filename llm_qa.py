import os
import time
import re
from typing import List, Dict, Optional, Generator
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path, override=True)

from llm_provider import LLMProvider, call_llm

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
                    top_k: int = DEFAULT_TOP_K,
                    provider: str = None) -> Dict:
    if not search_results:
        return {
            'answer': "No relevant documents found.",
            'sources': [],
            'latency': 0,
            'success': True
        }
    
    prompt = build_rag_prompt(query, search_results, top_k)
    
    start_time = time.time()
    
    try:
        answer = call_llm(prompt, max_tokens=max_tokens, temperature=temperature)
        
        if answer is None:
            latency = time.time() - start_time
            return {
                'answer': "LLM API key not configured.",
                'sources': [],
                'latency': round(latency, 2),
                'success': False
            }
        
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
    
    except Exception as e:
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
                              top_k: int = DEFAULT_TOP_K,
                              provider: str = None) -> Generator[str, None, None]:
    if not search_results:
        yield "No relevant documents found."
        return
    
    prompt = build_rag_prompt(query, search_results, top_k)
    
    try:
        answer = call_llm(prompt, max_tokens=max_tokens, temperature=temperature)
        
        if answer is None:
            yield "LLM API key not configured."
            return
        
        buffer = ""
        for char in answer:
            buffer += char
            if len(buffer) >= 50 or char in ['.', '!', '?', '\n']:
                yield buffer
                buffer = ""
        
        if buffer:
            yield buffer
    
    except Exception as e:
        yield f"LLM API error: {str(e)}"

def extract_citations(answer: str) -> List[str]:
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