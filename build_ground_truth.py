import os
import json
import sqlite3
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "my_local_database", "chroma.sqlite3")
OUTPUT_DIR = os.path.join(BASE_DIR, "experiment_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_all_documents():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, segment_id, embedding_id, seq_id FROM embeddings")
    embeddings = cursor.fetchall()
    
    cursor.execute("SELECT id, key, string_value, int_value, float_value, bool_value FROM embedding_metadata")
    metadata_rows = cursor.fetchall()
    
    documents = {}
    for row in embeddings:
        doc_id = row['id']
        documents[doc_id] = {
            'id': doc_id,
            'segment_id': row['segment_id'],
            'embedding_id': row['embedding_id'],
            'metadata': {}
        }
    
    for row in metadata_rows:
        doc_id = row['id']
        key = row['key']
        if row['string_value'] is not None:
            documents[doc_id]['metadata'][key] = row['string_value']
        elif row['int_value'] is not None:
            documents[doc_id]['metadata'][key] = row['int_value']
        elif row['float_value'] is not None:
            documents[doc_id]['metadata'][key] = row['float_value']
        elif row['bool_value'] is not None:
            documents[doc_id]['metadata'][key] = row['bool_value']
    
    conn.close()
    return list(documents.values())

def analyze_documents(docs):
    sources = defaultdict(list)
    topics = set()
    companies = set()
    
    for doc in docs:
        source = doc['metadata'].get('source_file', 'unknown')
        topic = doc['metadata'].get('topic', 'unknown')
        company = doc['metadata'].get('company', 'unknown')
        title = doc['metadata'].get('title', '')
        
        sources[source].append({
            'id': doc['id'],
            'title': title,
            'topic': topic
        })
        topics.add(topic)
        companies.add(company)
    
    print(f"\n=== Document Analysis ===")
    print(f"Total documents: {len(docs)}")
    print(f"Unique sources: {len(sources)}")
    print(f"Unique topics: {topics}")
    print(f"Unique companies: {companies}")
    
    print("\n=== Documents by Source ===")
    for source, items in sources.items():
        print(f"\n{source} ({len(items)} docs):")
        for item in items[:3]:
            title = item['title'][:60] if item['title'] else 'No title'
            print(f"  - {item['id']}: {title}")
    
    return sources, topics, companies

TEST_QUERIES = [
    {"query": "how to reset the device", "type": "semantic", "difficulty": "easy"},
    {"query": "scan to email", "type": "keyword", "difficulty": "easy"},
    {"query": "print from mobile phone", "type": "semantic", "difficulty": "medium"},
    {"query": "connect printer to wifi network", "type": "semantic", "difficulty": "medium"},
    {"query": "replace toner cartridge", "type": "keyword", "difficulty": "easy"},
    {"query": "clean the printhead", "type": "semantic", "difficulty": "medium"},
    {"query": "paper jam solution", "type": "keyword", "difficulty": "easy"},
    {"query": "send fax to multiple recipients", "type": "semantic", "difficulty": "hard"},
    {"query": "factory reset printer", "type": "semantic", "difficulty": "medium"},
    {"query": "software update for printer", "type": "semantic", "difficulty": "medium"},
    {"query": "configure network settings", "type": "semantic", "difficulty": "hard"},
    {"query": "check ink level", "type": "keyword", "difficulty": "easy"},
    {"query": "duplex printing setup", "type": "semantic", "difficulty": "medium"},
    {"query": "scan document to USB drive", "type": "semantic", "difficulty": "medium"},
    {"query": "setup email notification", "type": "semantic", "difficulty": "hard"},
    {"query": "how to install printer driver", "type": "semantic", "difficulty": "medium"},
    {"query": "wireless connection setup", "type": "semantic", "difficulty": "medium"},
    {"query": "maintenance schedule", "type": "keyword", "difficulty": "medium"},
    {"query": "troubleshoot print quality", "type": "semantic", "difficulty": "hard"},
    {"query": "energy saving mode", "type": "keyword", "difficulty": "easy"},
    {"query": "how to scan multiple pages", "type": "semantic", "difficulty": "medium"},
    {"query": "change paper size", "type": "keyword", "difficulty": "easy"},
    {"query": "set up scan to network folder", "type": "semantic", "difficulty": "hard"},
    {"query": "firmware upgrade instructions", "type": "semantic", "difficulty": "medium"},
    {"query": "default settings", "type": "keyword", "difficulty": "easy"},
    {"query": "how to use automatic document feeder", "type": "semantic", "difficulty": "medium"},
    {"query": "adjust print density", "type": "semantic", "difficulty": "medium"},
    {"query": "cancel print job", "type": "keyword", "difficulty": "easy"},
    {"query": "network troubleshooting guide", "type": "semantic", "difficulty": "hard"},
    {"query": "recommended paper types", "type": "keyword", "difficulty": "easy"},
]

def build_ground_truth(docs):
    ground_truth = {}
    
    for query_info in TEST_QUERIES:
        query = query_info['query']
        relevant_ids = []
        
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        
        for doc in docs:
            title = doc['metadata'].get('title', '').lower()
            content = doc['metadata'].get('content', '').lower()
            keywords = doc['metadata'].get('keywords', '').lower()
            
            match_score = 0
            
            for token in query_tokens:
                if token in title:
                    match_score += 3
                if token in content:
                    match_score += 1
                if token in keywords:
                    match_score += 2
            
            if match_score > 0:
                relevant_ids.append({'id': doc['id'], 'score': match_score})
        
        relevant_ids.sort(key=lambda x: x['score'], reverse=True)
        top_ids = [str(r['id']) for r in relevant_ids[:5]]
        
        ground_truth[query] = {
            'relevant_ids': top_ids,
            'type': query_info['type'],
            'difficulty': query_info['difficulty'],
            'num_relevant': len(top_ids)
        }
        
        print(f"Query: '{query[:40]}...' -> {len(top_ids)} relevant docs")
    
    ground_truth_path = os.path.join(OUTPUT_DIR, "ground_truth.json")
    with open(ground_truth_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"\n✅ Ground truth saved to {ground_truth_path}")
    return ground_truth

if __name__ == "__main__":
    docs = get_all_documents()
    sources, topics, companies = analyze_documents(docs)
    build_ground_truth(docs)