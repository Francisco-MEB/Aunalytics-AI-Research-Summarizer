# RAPTOR Clustering Algorithm: Technical Deep Dive

## Overview
The RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) system builds a hierarchical tree of document summaries through iterative clustering and summarization. This document explains how the clustering algorithm works in detail.

## Clustering Algorithm Details

### 1. **Automatic Cluster Count Determination**
```python
n_clusters = max(2, min(len(chunks) // 5, 10))
```

**Logic:**
- **Target:** 3-7 chunks per cluster (optimal for summarization quality)
- **Formula:** `chunks // 5` gives approximately 5 chunks per cluster
- **Constraints:**
  - **Minimum:** 2 clusters (always split content into at least 2 groups)
  - **Maximum:** 10 clusters (prevents over-fragmentation at higher levels)
- **Examples:**
  - 122 chunks → `122 // 5 = 24`, capped at `10` clusters
  - 10 chunks → `10 // 5 = 2` clusters
  - 5 chunks → `5 // 5 = 1`, minimum of `2` clusters

### 2. **K-Means Clustering**
```python
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
```

**Algorithm:** scikit-learn's KMeans implementation
- **Method:** Lloyd's algorithm with k-means++ initialization
- **Distance Metric:** Euclidean distance in embedding space (384-dimensional for all-MiniLM-L6-v2)
- **Initialization:** k-means++ (smart centroid placement to avoid local minima)
- **Parameters:**
  - `n_init=10`: Run algorithm 10 times with different seeds, pick best result
  - `random_state=42`: Reproducible results
  - `max_iter=300` (default): Maximum iterations for convergence

**How K-Means Groups Chunks:**
1. **Initialize:** Place k centroids using k-means++ (picks centroids far from each other)
2. **Assignment:** Assign each chunk to nearest centroid (Euclidean distance in embedding space)
3. **Update:** Recompute centroids as mean of assigned chunks
4. **Repeat:** Steps 2-3 until centroids don't move significantly (convergence)
5. **Result:** Each chunk belongs to exactly one cluster (hard clustering)

**Similarity Measure:**
- Embeddings from sentence-transformers (all-MiniLM-L6-v2)
- Cosine similarity between embeddings determines semantic closeness
- KMeans uses Euclidean distance, but on normalized embeddings this approximates cosine similarity

### 3. **Cluster Summarization**
```python
def summarize_cluster(self, chunk_texts: List[str]) -> str:
    combined = "\n\n---\n\n".join(chunk_texts)
    # Truncate to 8000 chars if needed
    # Send to Gemini 2.5 Flash for summarization
```

**Process:**
1. Concatenate all chunks in cluster with separator `\n\n---\n\n`
2. Truncate to 8000 chars if too long (avoids context limit)
3. Send to Gemini 2.5 Flash with prompt:
   ```
   Summarize this research content, preserving key details:
   [combined text]
   Provide a 2-3 paragraph summary:
   ```
4. Gemini generates abstractive summary (not extractive)
5. Summary becomes a new "chunk" for next level

### 4. **Recursive Hierarchy Building**
```python
while len(current_level) > 1:
    level_num += 1
    # Cluster current level
    # Summarize clusters → next_level
    # Store next_level in database
    current_level = next_level
```

**Example: 122 Chunks → 3 Levels**

**Level 0 (Base):** 122 raw document chunks
- Cluster into 10 groups
- Summarize each → 10 summaries

**Level 1:** 10 cluster summaries
- Cluster into 2 groups
- Summarize each → 2 summaries

**Level 2:** 2 cluster summaries
- Only 2 chunks left, cluster into 1 group
- Summarize → 1 summary

**Level 3 (Root):** 1 global summary
- **Stop:** Only 1 chunk, hierarchy complete

## Why This Clustering Approach?

### Advantages:
1. **Automatic Scaling:** No manual tuning, adapts to corpus size
2. **Semantic Grouping:** Similar content clusters together via embeddings
3. **Balanced Tree:** Target of 3-7 chunks per cluster prevents skewed trees
4. **Efficient:** KMeans is O(n·k·d·iterations), fast for moderate sizes
5. **Reproducible:** `random_state=42` ensures same hierarchy on re-runs

### Design Choices:
- **Hard Clustering (KMeans)** vs. Soft Clustering (GMM):
  - Simpler, faster, works well for document clustering
  - Each chunk belongs to exactly one parent in tree
- **K-Means++ Initialization:**
  - Better than random initialization
  - Reduces risk of poor local minima
- **Cluster Size Target (3-7 chunks):**
  - Small enough for coherent summaries
  - Large enough to avoid over-fragmentation
  - Based on empirical testing in RAPTOR paper

## Clustering in Action (Example)

**Initial State:**
```
Level 0: 122 chunks
├─ "Machine learning overview..."
├─ "Neural networks are..."
├─ "Transformers revolutionized..."
└─ [119 more chunks]
```

**After Level 1 Clustering:**
```
Level 1: 10 summaries
├─ Cluster 1 (13 chunks) → "Summary: ML fundamentals including supervised learning..."
├─ Cluster 2 (12 chunks) → "Summary: Deep learning architectures like CNNs and RNNs..."
├─ Cluster 3 (11 chunks) → "Summary: Transformer models and attention mechanisms..."
└─ [7 more clusters]
```

**After Level 2 Clustering:**
```
Level 2: 2 summaries
├─ Cluster A (5 summaries) → "High-level summary: Traditional ML and deep learning approaches..."
└─ Cluster B (5 summaries) → "High-level summary: Modern architectures with transformers..."
```

**After Level 3 Clustering:**
```
Level 3: 1 global summary
└─ "Overall summary: This document covers machine learning from basics to state-of-the-art..."
```

## Retrieval Strategies Using Hierarchy

### Different Question Types Use Different Levels:

1. **Factual Questions:** ("What is the learning rate in section 3?")
   - **Target:** Level 0 (raw chunks)
   - **Why:** Specific details are only in base chunks

2. **Summary Questions:** ("Summarize all documents")
   - **Target:** Level 2 or Level 3 (high-level summaries)
   - **Why:** Pre-computed abstractions, no re-summarization needed

3. **Analytical Questions:** ("How do these approaches compare?")
   - **Target:** Multi-level (Level 1 + Level 0)
   - **Why:** Need both high-level patterns and specific evidence

4. **Comparison Questions:** ("Compare method A and B")
   - **Target:** Multi-level (retrieve from Level 1, expand to Level 0)
   - **Why:** Find relevant clusters, then drill down to details

## Implementation Files

- **`embeddings/raptor_advanced.py`:** RaptorBuilder class with clustering logic
- **`embeddings/batch_ingest.py`:** Calls RaptorBuilder to build hierarchy
- **`app/raptor_qa.py`:** Retrieval logic using hierarchy (to be updated for hierarchy-aware retrieval)

## References

- **RAPTOR Paper:** "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval" (arXiv:2401.18059)
- **KMeans:** scikit-learn implementation (Lloyd's algorithm with k-means++)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (384-dimensional)
- **Summarization:** Google Gemini 2.5 Flash API

---

**Last Updated:** January 2025
**Status:** Hierarchy successfully built with 122 chunks → 10 → 2 → 1 (3 levels)
