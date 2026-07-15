import os
import sys
import json
import time
import csv
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from faiss_search import FAISSSearch
from reranker import GeminiReranker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiment_results")

DOCUMENT_TYPE_GROUPS = {
    'TechDocs': ['user_guide', 'manual', 'guide', 'handbook'],
    'Academic': ['paper', 'research', 'article', 'thesis', 'journal'],
    'Healthcare': ['medical', 'health', 'disease', 'treatment', 'clinical'],
    'Legal': ['law', 'legal', 'policy', 'regulation', 'compliance'],
    'Business': ['business', 'finance', 'marketing', 'strategy'],
    'Product': ['product', 'specification', 'datasheet', 'whitepaper'],
    'Other': []
}

def get_document_type(source_file):
    if not source_file:
        return 'Other'
    source_lower = str(source_file).lower()
    for doc_type, keywords in DOCUMENT_TYPE_GROUPS.items():
        if any(kw in source_lower for kw in keywords):
            return doc_type
    return 'Other'

def load_ground_truth():
    gt_path = os.path.join(OUTPUT_DIR, "ground_truth.json")
    with open(gt_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_metadata(doc_ids):
    db_path = os.path.join(BASE_DIR, "db", "chroma.sqlite3")
    metadata = {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for doc_id in doc_ids:
            try:
                int_id = int(doc_id)
                cursor.execute("SELECT string_value FROM embedding_metadata WHERE id = ? AND key = 'source_file'", (int_id,))
                row = cursor.fetchone()
                source_file = row[0] if row else None
                cursor.execute("SELECT int_value FROM embedding_metadata WHERE id = ? AND key = 'source_page'", (int_id,))
                row = cursor.fetchone()
                source_page = row[0] if row else None
                metadata[str(doc_id)] = {'source_file': source_file, 'source_page': source_page}
            except:
                metadata[str(doc_id)] = {'source_file': None, 'source_page': None}
        conn.close()
    except Exception as e:
        print(f"Error loading metadata: {e}")
    return metadata

def ndcg_at_k(scores, k):
    scores = scores[:k]
    if not scores:
        return 0.0
    dcg = sum(scores[i] / (i + 1) for i in range(len(scores)))
    idcg = sum(1.0 / (i + 1) for i in range(len(scores)))
    return dcg / idcg

def compute_metrics(results, relevant_ids):
    result_ids = [str(r.get('doc_id', '')) for r in results]
    scores = []
    for rid in relevant_ids:
        if rid in result_ids:
            pos = result_ids.index(rid)
            scores.append(1.0 / (pos + 1))
        else:
            scores.append(0.0)
    ndcg5 = ndcg_at_k(scores, 5)
    ndcg10 = ndcg_at_k(scores, 10)
    mrr = max(scores) if scores else 0.0
    recall5 = sum(1 for rid in relevant_ids if rid in result_ids[:5]) / len(relevant_ids)
    recall10 = sum(1 for rid in relevant_ids if rid in result_ids[:10]) / len(relevant_ids)
    precision5 = sum(1 for rid in result_ids[:5] if rid in relevant_ids) / min(5, len(result_ids)) if result_ids else 0
    precision10 = sum(1 for rid in result_ids[:10] if rid in relevant_ids) / min(10, len(result_ids)) if result_ids else 0
    f1_5 = 2 * precision5 * recall5 / (precision5 + recall5) if (precision5 + recall5) > 0 else 0
    f1_10 = 2 * precision10 * recall10 / (precision10 + recall10) if (precision10 + recall10) > 0 else 0
    return {
        'ndcg@5': ndcg5,
        'ndcg@10': ndcg10,
        'mrr': mrr,
        'recall@5': recall5,
        'recall@10': recall10,
        'precision@5': precision5,
        'precision@10': precision10,
        'f1@5': f1_5,
        'f1@10': f1_10
    }

def run_gemini_experiment():
    print("=== Initializing FAISS Search ===")
    searcher = FAISSSearch()
    searcher.initialize()
    
    print("=== Initializing Gemini Reranker ===")
    gemini_reranker = GeminiReranker()
    
    ground_truth = load_ground_truth()
    test_queries = [{'query': q, 'type': gt['type'], 'difficulty': gt['difficulty']} 
                    for q, gt in ground_truth.items()]
    
    all_metrics = []
    all_results = []
    
    print("\n=== Running Hybrid_Gemini ===")
    for query_info in test_queries:
        query = query_info['query']
        relevant_ids = ground_truth[query]['relevant_ids']
        
        start_time = time.time()
        results = searcher.hybrid_search(query, k=50, reranker_type='gemini')
        latency = time.time() - start_time
        
        metrics = compute_metrics(results, relevant_ids)
        
        result_doc_types = []
        doc_ids = [str(r.get('doc_id', '')) for r in results[:10]]
        metadata = load_metadata(doc_ids)
        for doc_id in doc_ids:
            source_file = metadata.get(doc_id, {}).get('source_file', '')
            doc_type = get_document_type(source_file)
            result_doc_types.append(doc_type)
        
        metrics.update({
            'mode': 'Hybrid_Gemini',
            'query': query,
            'query_type': query_info['type'],
            'difficulty': query_info['difficulty'],
            'latency': latency,
            'result_doc_types': json.dumps(result_doc_types)
        })
        all_metrics.append(metrics)
        
        result_entry = {'mode': 'Hybrid_Gemini', 'query': query, 'query_type': query_info['type'], 
                        'difficulty': query_info['difficulty'], 'latency': latency, 'num_results': len(results)}
        result_entry.update(metrics)
        for i, res in enumerate(results[:5]):
            result_entry[f'result_{i}_id'] = res.get('doc_id', '')
            result_entry[f'result_{i}_score'] = res.get('rrf_score', res.get('similarity', res.get('rerank_score', '')))
        
        all_results.append(result_entry)
        print(f"  Query: '{query[:30]}...' | Latency: {latency:.3f}s | NDCG@10: {metrics['ndcg@10']:.4f} | MRR: {metrics['mrr']:.4f}")
        
        time.sleep(15)
    
    csv_path = os.path.join(OUTPUT_DIR, "gemini_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['mode', 'query', 'query_type', 'difficulty', 'latency', 'num_results', 'ndcg@5', 'ndcg@10', 'mrr'] + \
                     [f'result_{i}_id' for i in range(5)] + [f'result_{i}_score' for i in range(5)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    metrics_csv_path = os.path.join(OUTPUT_DIR, "gemini_metrics.csv")
    with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['mode', 'query', 'query_type', 'difficulty', 'latency', 'ndcg@5', 'ndcg@10', 'mrr', 
                      'recall@5', 'recall@10', 'precision@5', 'precision@10', 'f1@5', 'f1@10', 'result_doc_types']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
    
    summary = {
        'avg_latency': sum(m['latency'] for m in all_metrics) / len(all_metrics),
        'avg_ndcg@5': sum(m['ndcg@5'] for m in all_metrics) / len(all_metrics),
        'avg_ndcg@10': sum(m['ndcg@10'] for m in all_metrics) / len(all_metrics),
        'avg_mrr': sum(m['mrr'] for m in all_metrics) / len(all_metrics),
        'avg_recall@5': sum(m['recall@5'] for m in all_metrics) / len(all_metrics),
        'avg_recall@10': sum(m['recall@10'] for m in all_metrics) / len(all_metrics),
        'avg_precision@5': sum(m['precision@5'] for m in all_metrics) / len(all_metrics),
        'avg_precision@10': sum(m['precision@10'] for m in all_metrics) / len(all_metrics),
        'avg_f1@5': sum(m['f1@5'] for m in all_metrics) / len(all_metrics),
        'avg_f1@10': sum(m['f1@10'] for m in all_metrics) / len(all_metrics),
        'total_queries': len(all_metrics)
    }
    
    summary_path = os.path.join(OUTPUT_DIR, "gemini_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'Hybrid_Gemini': summary}, f, indent=2)
    
    print(f"\n✅ Gemini results saved to {csv_path}")
    print(f"✅ Gemini metrics saved to {metrics_csv_path}")
    print(f"✅ Gemini summary saved to {summary_path}")

if __name__ == "__main__":
    run_gemini_experiment()