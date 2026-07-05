import os
import time
import math
from typing import Generator, List

DEFAULT_CHUNK_SIZE = 50
MAX_MEMORY_MB = 1024


def split_pdf_by_pages(pdf_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Generator[List[int], None, None]:
    """
    Split PDF into page chunks for streaming processing.
    
    Args:
        pdf_path: Path to PDF file
        chunk_size: Number of pages per chunk
    
    Yields:
        List of page indices for each chunk
    """
    try:
        import fitz
        doc = fitz.Document(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        for i in range(0, total_pages, chunk_size):
            yield list(range(i, min(i + chunk_size, total_pages)))
    except Exception as e:
        print(f"[ERROR] Failed to get page count: {e}")
        yield [0]


def estimate_memory_usage(num_docs: int, embedding_dim: int = 384, faiss_index_size: float = 0.5) -> float:
    """
    Estimate FAISS memory usage in MB.
    
    Args:
        num_docs: Number of documents
        embedding_dim: Embedding dimension (default: 384)
        faiss_index_size: Estimated index overhead per document (MB)
    
    Returns:
        Estimated memory usage in MB
    """
    embedding_memory_mb = (num_docs * embedding_dim * 4) / (1024 * 1024)
    index_memory_mb = num_docs * faiss_index_size
    return embedding_memory_mb + index_memory_mb


def should_chunk_file(file_path: str, max_memory_mb: int = MAX_MEMORY_MB) -> bool:
    """
    Determine if a file should be chunked based on size and memory constraints.
    
    Args:
        file_path: Path to file
        max_memory_mb: Maximum allowed memory usage
    
    Returns:
        True if file should be chunked
    """
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    if file_size_mb > max_memory_mb * 0.5:
        return True
    
    try:
        import fitz
        doc = fitz.Document(file_path)
        num_pages = len(doc)
        doc.close()
        
        estimated_memory = estimate_memory_usage(num_pages * 10)
        if estimated_memory > max_memory_mb:
            return True
    except:
        pass
    
    return False


def process_pdf_in_chunks(pdf_path: str, 
                          process_func, 
                          chunk_size: int = DEFAULT_CHUNK_SIZE,
                          **kwargs) -> list:
    """
    Process a large PDF in chunks to avoid memory overflow.
    
    Args:
        pdf_path: Path to PDF file
        process_func: Function to process each chunk (receives page indices)
        chunk_size: Number of pages per chunk
        **kwargs: Additional arguments to pass to process_func
    
    Returns:
        List of results from each chunk
    """
    results = []
    
    for page_indices in split_pdf_by_pages(pdf_path, chunk_size):
        print(f"Processing pages {page_indices[0] + 1}-{page_indices[-1] + 1}...")
        
        try:
            result = process_func(pdf_path, page_indices, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed to process pages {page_indices[0] + 1}-{page_indices[-1] + 1}: {e}")
            results.append(None)
    
    return results


def chunk_text_by_size(text: str, max_chunk_chars: int = 5000, 
                       overlap_chars: int = 500) -> List[str]:
    """
    Split text into chunks by character count with overlap.
    
    Args:
        text: Input text
        max_chunk_chars: Maximum characters per chunk
        overlap_chars: Overlap between consecutive chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + max_chunk_chars, text_length)
        
        if end < text_length:
            last_period = text.rfind('.', start, end)
            if last_period != -1 and last_period > start + max_chunk_chars // 2:
                end = last_period + 1
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= text_length:
            break
        
        start = end - overlap_chars
        if start < 0:
            start = 0
    
    return chunks


def chunk_text_by_paragraphs(text: str, max_paragraphs_per_chunk: int = 10) -> List[str]:
    """
    Split text into chunks by paragraphs.
    
    Args:
        text: Input text
        max_paragraphs_per_chunk: Maximum paragraphs per chunk
    
    Returns:
        List of text chunks
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    
    for i in range(0, len(paragraphs), max_paragraphs_per_chunk):
        chunk = '\n\n'.join(paragraphs[i:i + max_paragraphs_per_chunk])
        chunks.append(chunk)
    
    return chunks
