# Incremental RAPTOR Hierarchy Updates: Research & Implementation Strategy

## Problem Statement
Currently, adding or deleting documents requires rebuilding the entire RAPTOR hierarchy from scratch. This is inefficient for large corpora.

**Desired Behavior:**
- Add new document → Insert into hierarchy without rebuilding all existing clusters
- Delete document → Remove from hierarchy without rebuilding all existing clusters

## Why Full Rebuild is Inefficient

**Current Process:**
1. Delete all Level 1+ summaries (keep Level 0 chunks)
2. Add new document chunks to Level 0
3. Re-cluster ALL Level 0 chunks (old + new)
4. Re-summarize ALL clusters
5. Recursively rebuild all levels

**Time Complexity:** O(n * d * k * iterations) for KMeans, where n = total chunks
- For 1000 chunks: ~30 seconds
- For 10,000 chunks: ~5 minutes
- For 100,000 chunks: ~1 hour

## Research: Incremental Clustering Approaches

### 1. **Online KMeans / Mini-Batch KMeans**
**Algorithm:** Process data in batches, update centroids incrementally

**How it works:**
```python
from sklearn.cluster import MiniBatchKMeans

# Initial clustering
mbkmeans = MiniBatchKMeans(n_clusters=10, random_state=42)
mbkmeans.fit(existing_embeddings)

# Add new data
mbkmeans.partial_fit(new_embeddings)  # Update centroids without full rebuild
```

**Pros:**
- Fast updates: O(batch_size * k) instead of O(n * k)
- Scikit-learn implementation available
- Memory efficient

**Cons:**
- Cluster quality may degrade over time (drift)
- New data can shift centroids, affecting old assignments
- Need periodic full re-clustering to maintain quality

**Applicability to RAPTOR:**
-  Can handle adding new chunks incrementally
- ️ Deleting chunks is problematic (centroids may become invalid)
- ️ Summaries would need regeneration when clusters change

### 2. **Soft Clustering (Nearest Cluster Assignment)**
**Algorithm:** Assign new chunks to nearest existing cluster without re-clustering

**How it works:**
```python
def add_chunk_to_hierarchy(new_chunk_embedding, existing_clusters):
    # Find nearest Level 1 cluster
    distances = [cosine_distance(new_chunk_embedding, cluster_centroid) 
                 for cluster_centroid in cluster_centroids]
    nearest_cluster = np.argmin(distances)
    
    # Add chunk to that cluster
    clusters[nearest_cluster].append(new_chunk)
    
    # Re-summarize ONLY that cluster
    clusters[nearest_cluster]['summary'] = summarize_cluster(clusters[nearest_cluster])
    
    # Propagate up the hierarchy (re-summarize parent nodes)
    propagate_update_to_parents(nearest_cluster)
```

**Pros:**
- Very fast: O(k) for nearest cluster search
- No full re-clustering needed
- Preserves existing cluster structure

**Cons:**
- Cluster imbalance: popular clusters grow indefinitely
- No guarantee of optimal clustering
- May violate "3-7 chunks per cluster" target

**Applicability to RAPTOR:**
-  Adding new documents is fast
- ️ Deleting still requires parent re-summarization
- ️ Cluster quality degrades over time

### 3. **Lazy/Dirty Rebuild (Mark and Rebuild Affected Branches)**
**Algorithm:** Mark affected clusters as "dirty", rebuild only those branches

**How it works:**
```python
def add_document(new_doc_chunks):
    # Add to Level 0
    for chunk in new_doc_chunks:
        db.insert(chunk, level=0)
        # Mark parent cluster as dirty
        parent_cluster = find_parent_cluster(chunk)
        mark_dirty(parent_cluster)
    
    # Rebuild only dirty branches
    rebuild_dirty_nodes()

def mark_dirty(node_id):
    # Mark node and all ancestors as dirty
    db.update(node_id, dirty=True)
    parent = get_parent(node_id)
    if parent:
        mark_dirty(parent)

def rebuild_dirty_nodes():
    # Rebuild from bottom to top
    for level in range(1, max_level + 1):
        dirty_nodes = db.query("SELECT * FROM documents WHERE level = ? AND dirty = TRUE", level)
        for node in dirty_nodes:
            children = get_children(node)
            re_cluster_and_summarize(children)
            mark_clean(node)
```

**Pros:**
- Only rebuilds affected parts of tree
- Maintains cluster quality in unaffected regions
- Clear audit trail of what changed

**Cons:**
- Complex implementation (need "dirty" tracking)
- Adding many documents may cascade to full rebuild
- Database schema changes required

**Applicability to RAPTOR:**
-  Efficient for small updates (1-2 docs)
- ️ Complex to implement
- ️ Large updates may cascade to near-full rebuild

### 4. **Fixed Capacity Clusters (Pre-allocated Slots)**
**Algorithm:** Pre-allocate cluster capacity, add to cluster with space

**How it works:**
```python
# Pre-define cluster structure
MAX_CHUNKS_PER_CLUSTER = 10
clusters = create_initial_clusters(num_clusters=20, capacity=10)

def add_chunk(new_chunk):
    # Find nearest cluster with available capacity
    nearest_cluster = find_nearest_cluster_with_space(new_chunk)
    
    if nearest_cluster:
        clusters[nearest_cluster].append(new_chunk)
        re_summarize(nearest_cluster)
    else:
        # Create overflow cluster
        create_new_cluster(new_chunk)
```

**Pros:**
- Predictable structure
- Easy to implement
- Bounded cluster sizes

**Cons:**
- Wasted capacity if clusters are unevenly filled
- Overflow handling is complex
- May violate semantic clustering

**Applicability to RAPTOR:**
- ️ Not well-suited for semantic clustering
- ️ Capacity planning is difficult
-  Breaks KMeans assumptions

### 5. **Periodic Full Rebuild (Accept Staleness)**
**Algorithm:** Fast incremental updates, periodic full rebuilds

**How it works:**
```python
def add_document(new_doc):
    # Quick add: assign to nearest clusters
    for chunk in new_doc_chunks:
        nearest_cluster = find_nearest_cluster(chunk)
        clusters[nearest_cluster].append(chunk)
        mark_stale(nearest_cluster)
    
    # Check if rebuild needed
    stale_percentage = count_stale_nodes() / total_nodes()
    if stale_percentage > 0.3:  # 30% stale
        trigger_full_rebuild()
```

**Pros:**
- Simple to implement
- Fast most of the time
- Maintains cluster quality long-term

**Cons:**
- Periodic slow rebuilds
- Temporary suboptimal clustering
- Need to define "staleness" threshold

**Applicability to RAPTOR:**
-  Simple and practical
-  Works for both add and delete
-  Balances speed and quality

## CORRECTION: Active Research Area (2024-2025 Findings)

### **The Truth About Incremental Hierarchical Updates**

After extensive research of 2024-2025 papers, here's what I found:

1. **RAPTOR Paper (2024):** Makes NO mention of incremental updates
2. **Active Research (2024-2025):** Multiple papers on incremental hierarchical clustering:
   - **"Hierarchical Clustering Without Pairwise Distances by Incremental Similarity Search"** (SISAP 2024) - Schubert
   - **"Data stream clustering: introducing recursively extendable aggregation functions"** (IEEE 2025)
   - **"ICE: Incremental Subspace Clustering"** (2025)
   - **"Incremental hierarchical text clustering methods: a review"** (2023) - 26 pages surveying methods
3. **Industry Reality:** Companies still do **full rebuilds** periodically, but research is advancing

### **Key Research Insight: This IS an Open Problem Worth Solving**

The 2023 review paper "Incremental hierarchical text clustering methods: a review" confirms:
- Active research area with multiple approaches
- No single "best" solution
- Trade-offs between speed, quality, and complexity

---

## Research Opportunity: RAPTOR-Specific Challenge

**The Novel Problem:** None of the 2024-2025 papers address **incremental updates to recursive summarization + clustering trees** like RAPTOR.

**What 2024-2025 Research Focuses On:**
- **Density-based clustering** (HDBSCAN variants) - Schubert 2024
- **Data stream clustering** (online/streaming data) - IEEE 2025
- **Categorical data clustering** - ICE 2025
- **Applications:** Customer segmentation, IoT, cybersecurity

**What's Missing from Research:**
Incremental updates to **hierarchical summarization trees** where:
- Level 0 = raw text chunks (embeddings)
- Level 1+ = LLM-generated summaries of clusters (new embeddings)
- New documents must integrate without destroying existing clustering quality
- Full rebuild is O(n³) with KMeans

**Why RAPTOR is Different:**
- Traditional clustering: Just rearrange points into clusters
- RAPTOR clustering: Each cluster triggers **LLM summarization** (expensive!)
- Summary becomes new embedding → feeds into next level
- Adding 1 chunk can cascade changes up entire tree

**Potential Research Contributions:**
1. **Lazy rebuild algorithm** - Mark dirty subtrees, rebuild only affected branches
2. **Streaming RAPTOR** - Incremental summarization with quality guarantees
3. **Hybrid approach** - Fast Level 0 add + intelligent partial rebuilds
4. **Benchmark dataset** - Evaluate incremental vs full rebuild quality

**Publication Venues:**
- ACL, EMNLP (NLP conferences) - retrieval-augmented generation
- SIGIR, WWW (IR conferences) - hierarchical document clustering
- NeurIPS, ICML (ML conferences) - online learning algorithms

---

**Theoretical Issues:**
- Hierarchical clustering is **globally optimal** - adding one chunk can change the entire structure
- K-means assumes you're clustering ALL data at once
- "Incremental" approaches degrade to random assignment over time

**Practical Reality:**
- Google's vector search: Rebuilds indexes nightly
- Pinecone/Weaviate: Recommend batch updates
- OpenAI embeddings: No incremental hierarchy support

### **Recommended Solution: Optimize Full Rebuild**

Instead of trying to avoid rebuilds, make them **fast enough that you don't care**:

#### **Option 1: Fast Incremental Index (No Hierarchy for New Docs)**
```python
def add_document_fast(new_doc):
    # Step 1: Add to Level 0 immediately (no hierarchy)
    chunks = chunk_document(new_doc)
    embeddings = embed_chunks(chunks)
    db.insert(chunks, level=0, in_hierarchy=False)
    
    # Step 2: Users can query immediately (searches Level 0)
    # New chunks are searchable but not in hierarchy yet
    
    # Step 3: Schedule full rebuild for later
    if get_docs_not_in_hierarchy() > 10:  # Threshold
        schedule_async_rebuild()  # Background job

# Queries work normally - they search ALL Level 0 chunks
# Hierarchy is rebuilt periodically to include new docs
```

**Benefits:**
-  Add docs in ~2 seconds (just embed + store)
-  Queries work immediately on new docs
-  Hierarchy stays clean and optimal
-  Rebuild happens in background (10-20 mins for 1000s of docs)

#### **Option 2: Parallel Hierarchies**
```python
# Maintain TWO hierarchies
hierarchy_stable = RaptorTree(docs_1_to_100)  # Main hierarchy
hierarchy_new = RaptorTree(docs_101_to_110)    # New docs hierarchy

def query(question):
    # Query both hierarchies
    results_stable = hierarchy_stable.retrieve(question)
    results_new = hierarchy_new.retrieve(question)
    return merge_results([results_stable, results_new])

# Periodically: Merge hierarchies
if len(hierarchy_new) > 20:
    hierarchy_stable = RaptorTree(all_docs)
    hierarchy_new = RaptorTree([])
```

**Benefits:**
-  New docs have their own hierarchy immediately
-  No degradation of main hierarchy quality
-  Merge happens periodically

#### **Option 3: No Hierarchy (Flat Vector Search)**
```python
# Controversial take: Do you actually NEED hierarchy?

# For small corpora (<1000 docs):
# - Flat vector search is FASTER than hierarchy traversal
# - Quality is often BETTER (no abstraction loss)
# - pgvector can search 10K chunks in <100ms

# Use hierarchy only for:
# - Very large corpora (>10K docs)
# - Complex multi-document synthesis
# - When you need different abstraction levels

# Alternative: BM25 + Vector Hybrid
results_vector = vector_search(query, top_k=100)
results_bm25 = keyword_search(query, top_k=100)
results = rerank(results_vector + results_bm25)
```

**Reality Check:**
- Your 2 documents (~122 chunks) don't benefit from hierarchy
- Flat search would be faster and simpler
- Consider hierarchy when you have 50+ documents

### **Phase 2: Periodic Full Rebuild (Background Job)**
- Run full rebuild nightly or when >10 new docs added
- Use queue system (Celery, Redis Queue, etc.)
- Queries continue using old hierarchy until new one is ready
- Atomic swap when rebuild completes

## Recommended: Fast Incremental with Periodic Rebuild

### **The Practical Solution**

```python
class ProductionRaptorSystem:
    """
    Real-world approach: Fast adds + scheduled rebuilds
    """
    
    def add_document(self, doc_path):
        """Add document in ~2 seconds, hierarchy rebuilt later"""
        # 1. Chunk and embed
        chunks = chunk_document(doc_path)
        embeddings = embed_chunks(chunks)
        
        # 2. Store at Level 0 (NOT in hierarchy yet)
        for chunk, emb in zip(chunks, embeddings):
            db.insert(
                chunk=chunk,
                embedding=emb,
                level=0,
                in_hierarchy=False,  # Flag: not yet in tree
                doc_id=doc_path
            )
        
        print(f" Added {len(chunks)} chunks. Searchable immediately!")
        print(f"⏳ Hierarchy will rebuild when 10+ new docs added")
        
        # 3. Check if rebuild needed
        new_docs_count = db.count_where(in_hierarchy=False)
        if new_docs_count >= 10:
            self.schedule_rebuild()
    
    
    def schedule_rebuild(self):
        """Queue background rebuild job"""
        # Option 1: Immediate async
        threading.Thread(target=self.rebuild_hierarchy_async).start()
        
        # Option 2: Queue for worker (production)
        # redis_queue.enqueue(self.rebuild_hierarchy_async)
        
        # Option 3: Scheduled (cron job runs nightly)
        # Just log that rebuild is needed
        print(" Rebuild queued for background processing")
    
    
    def rebuild_hierarchy_async(self):
        """Full rebuild in background (10-30 mins for large corpus)"""
        print(" Starting full hierarchy rebuild...")
        
        # Get ALL Level 0 chunks (old + new)
        all_chunks = db.query("SELECT * FROM documents WHERE level = 0")
        
        # Clear old hierarchy (keep Level 0)
        db.execute("DELETE FROM documents WHERE level > 0")
        
        # Rebuild hierarchy
        raptor = RaptorBuilder()
        raptor.build_hierarchy_with_clustering(
            all_chunks, 
            source_file="corpus", 
            user_id=self.user_id
        )
        
        # Mark all as in hierarchy
        db.execute("UPDATE documents SET in_hierarchy = TRUE WHERE level = 0")
        
        print(" Hierarchy rebuild complete!")
    
    
    def query(self, question):
        """Query works on ALL Level 0 chunks (hierarchy + non-hierarchy)"""
        # Level 0 searches work immediately on new docs
        if self.classify_question(question) == "FACTUAL":
            return self.retrieve_from_level(question, level=0)
        
        # Summary questions use hierarchy (excludes new docs until rebuild)
        elif self.classify_question(question) == "SUMMARY":
            # This uses only docs in hierarchy
            # New docs not yet in hierarchy won't be in high-level summaries
            # But that's OK - they're still searchable at Level 0
            return self.retrieve_from_level(question, level=2)
```

### **User Experience**

```bash
# User adds 1 document
$ python add_doc.py new_paper.pdf
 Added 25 chunks. Searchable immediately!
⏳ Hierarchy will rebuild when 10+ new docs added

# Queries work right away on new doc
$ python raptor_qa.py
Question: What does the new paper say about X?
[Searches Level 0, includes new doc] 

Question: Summarize all documents
[Uses hierarchy Level 2, excludes new doc] ️
[Note: New doc not in hierarchy yet, but will be after rebuild]

# After 10 docs added
 Added 22 chunks. Searchable immediately!
 Rebuild queued for background processing
[Rebuild happens in background, takes 10 mins]
 Hierarchy rebuild complete!

# Now all docs are in hierarchy
Question: Summarize all documents
[Uses hierarchy Level 2, includes ALL docs] 
```

### **Performance:**
- **Add 1 doc:** ~2 seconds (embed + store)
- **Add 10 docs:** ~20 seconds (embed + store all)
- **Full rebuild (background):** 10-30 minutes for 1000s of docs
- **Query during rebuild:** Still works (uses old hierarchy)

### **Scalability:**
| Corpus Size | Rebuild Time | Rebuild Frequency |
|-------------|--------------|-------------------|
| <100 docs | ~30 seconds | After every 10 adds |
| 100-1000 docs | ~5 minutes | Nightly |
| 1000-10K docs | ~30 minutes | Weekly |
| >10K docs | ~2 hours | Consider flat search instead |

## Database Schema Updates Required

Add columns to track cluster state:

```sql
ALTER TABLE documents 
ADD COLUMN stale BOOLEAN DEFAULT FALSE,
ADD COLUMN last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN child_count INTEGER DEFAULT 0;

CREATE INDEX idx_stale ON documents(user_id, hierarchy_level, stale);
```

## Testing Strategy

```python
# Test 1: Add 1 document
add_document_incremental(new_doc)
assert hierarchy_valid()
assert all_docs_covered()

# Test 2: Add 10 documents
for doc in new_docs:
    add_document_incremental(doc)
assert hierarchy_valid()
assert cluster_balance_ok()

# Test 3: Delete 1 document
delete_document_incremental(doc_id)
assert hierarchy_valid()
assert no_orphaned_clusters()

# Test 4: Trigger rebalancing
add_many_documents()  # Force rebalancing
assert full_rebuild_triggered()
```

## Performance Estimates

**Current (Full Rebuild):**
- Add 1 doc to 1000 chunks: ~30 seconds
- Add 10 docs: ~35 seconds
- Delete 1 doc: ~30 seconds

**With Incremental Updates:**
- Add 1 doc: ~2 seconds (soft clustering)
- Add 10 docs: ~20 seconds (soft clustering)
- Delete 1 doc: ~1 second
- Full rebuild (triggered): ~30 seconds (same as before)

**Expected Improvement:** 10-15x faster for small updates

## Trade-offs Summary

| Strategy | Add Speed | Delete Speed | Quality | Complexity |
|----------|-----------|--------------|---------|------------|
| Full Rebuild (current) |  Slow |  Slow |  Optimal |  Simple |
| Online KMeans | ️ Medium |  Hard | ️ Degrades | ️ Medium |
| Soft Clustering |  Fast |  Fast | ️ Degrades |  Simple |
| Lazy Rebuild |  Fast |  Fast |  Good |  Complex |
| Periodic Rebuild |  Fast |  Fast |  Good |  Simple |
| **Hybrid (Recommended)** |  Fast |  Fast |  Good | ️ Medium |

## Next Steps

1.  Document incremental strategies (this document)
2. Implement `IncrementalRaptorBuilder` class
3. Add database schema updates
4. Test with real workloads
5. Monitor cluster quality metrics
6. Tune rebalancing thresholds

---

**Status:** Research complete, implementation pending
**Estimated Implementation Time:** 8-12 hours
**Recommended Approach:** Hybrid (Soft Clustering + Periodic Rebuild)
