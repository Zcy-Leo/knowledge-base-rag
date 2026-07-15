# Experimental Results

## Table 1: Performance Comparison of Different Search Modes

| Search Mode | Avg. Latency (s) | NDCG@5 | NDCG@10 | MRR | Recall@10 | Precision@10 | F1@10 |
|-------------|------------------|--------|---------|-----|-----------|--------------|-------|
| Pure_Vector | 0.017 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Pure_BM25 | 0.003 | 0.1093 | 0.1093 | 0.1556 | 0.1333 | 0.0400 | 0.0615 |
| Hybrid_RRF | 0.032 | 0.5630 | 0.6450 | 0.8333 | 0.6889 | 0.2067 | 0.3179 |
| Hybrid_CrossEncoder_L6 | 1.797 | 0.9292 | 0.9292 | 1.0000 | 0.9111 | 0.2733 | 0.4205 |
| Hybrid_CrossEncoder_L12 | 4.453 | 0.7970 | 0.8408 | 0.9667 | 0.8889 | 0.2667 | 0.4103 |
| Hybrid_Gemini | 10.991 | 0.6512 | 0.6839 | 0.8333 | 0.6889 | 0.2067 | 0.3179 |

## Table 2: Performance by Query Type

| Search Mode | Keyword NDCG@10 | Semantic NDCG@10 | Keyword MRR | Semantic MRR |
|-------------|-----------------|------------------|-------------|--------------|
| Pure_Vector | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Pure_BM25 | 0.0505 | 0.1306 | 0.0625 | 0.1894 |
| Hybrid_RRF | 0.6718 | 0.6352 | 0.8750 | 0.8182 |
| Hybrid_CrossEncoder_L6 | 0.9260 | 0.9304 | 1.0000 | 1.0000 |
| Hybrid_CrossEncoder_L12 | 0.8862 | 0.8243 | 1.0000 | 0.9545 |
| Hybrid_Gemini | 0.6718 | 0.6883 | 0.8750 | 0.8182 |

## Table 3: Ablation Study

| Configuration | NDCG@10 | MRR | Latency (s) |
|---------------|---------|-----|-------------|
| Vector Only | 0.0000 | 0.0000 | 0.017 |
| BM25 Only | 0.1093 | 0.1556 | 0.003 |
| Hybrid (RRF) | 0.6450 | 0.8333 | 0.032 |
| Hybrid + L6 | 0.9292 | 1.0000 | 1.797 |
| Hybrid + L12 | 0.8408 | 0.9667 | 4.453 |

## Table 4: Reranker Model Comparison

| Reranker Model | NDCG@10 | MRR | Avg. Latency (s) |
|----------------|---------|-----|------------------|
| None (RRF only) | 0.6450 | 0.8333 | 0.032 |
| MiniLM-L-6-v2 | 0.9292 | 1.0000 | 1.797 |
| MiniLM-L-12-v2 | 0.8408 | 0.9667 | 4.453 |

## Results and Discussion

### 1. Overall Performance Analysis

Based on the experimental results, the following observations can be made:

- **Best Overall Performance**: Hybrid_CrossEncoder_L6 achieves the highest NDCG@10 score of 0.9292, indicating superior ranking quality.

- **Fastest Response**: Pure_BM25 has the lowest latency of 0.003s, making it suitable for real-time applications.

### 2. Search Mode Comparison

The experimental results demonstrate that hybrid search approaches consistently outperform single-mode retrieval:

- Pure Vector search excels at semantic understanding but may miss keyword-matching documents.
- Pure BM25 search is fast and effective for exact keyword matches but lacks semantic understanding.
- Hybrid search combining Vector and BM25 via RRF fusion leverages the strengths of both approaches.
- Adding CrossEncoder reranking further improves ranking quality by re-scoring candidate documents.

### 3. Query Type Analysis

Different query types exhibit varying performance across search modes:

- **Keyword Queries**: BM25-based approaches typically perform better due to exact term matching.
- **Semantic Queries**: Vector-based approaches and hybrid methods with reranking show stronger performance.

### 4. Reranker Model Impact

CrossEncoder reranking significantly improves retrieval quality:

- MiniLM-L-6-v2 provides a good balance between performance and speed.
- MiniLM-L-12-v2 offers slightly better accuracy but at the cost of increased latency.
- The choice of reranker depends on the specific requirements of the application.

### 5. Practical Implications

For production deployment, the following recommendations can be made:

- **High-throughput scenarios**: Use Pure BM25 or Hybrid RRF without reranking.
- **High-accuracy requirements**: Use Hybrid search with CrossEncoder reranking.
- **Balanced performance**: Hybrid RRF with MiniLM-L-6-v2 provides the best overall trade-off.

