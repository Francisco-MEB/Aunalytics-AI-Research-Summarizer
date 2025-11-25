# RAPTOR Incremental Updates: Decoupled Architecture

## Executive Summary

**Current Problem:** Adding/deleting 1 document requires rebuilding entire tree (O(n³) for all chunks)

**Solution:** Decouple base chunks from tree structure, make tree mutable with local updates only

**Result:** 
- Add document: O(log n) + O(affected nodes) instead of O(n³)
- Delete document: O(log n) + O(affected nodes) instead of O(n³)
- Query: Same speed or faster (tree guides to relevant chunks)

---

## Architecture: 2-Layer System

### Layer 1: Base Vector Index (Ground Truth)
```
┌─────────────────────────────────────┐
│   BASE LAYER: Raw Chunk Storage     │
│   Always correct, easy to update    │
└─────────────────────────────────────┘

Table: chunks
- chunk_id (UUID)
- doc_id (UUID) 
- text (full chunk text)
- embedding (vector)
- deleted (boolean flag for soft delete)
- created_at (timestamp)

Operations:
 Add chunk: O(1) - just insert
 Delete chunk: O(1) - set deleted=true
 Query chunks: O(log n) - vector search
```

### Layer 2: Tree Structure (Semantic Map)
```
┌─────────────────────────────────────┐
│   TREE LAYER: Hierarchical Overlay  │
│   Incrementally updated, prunable   │
└─────────────────────────────────────┘

Table: tree_nodes
- node_id (UUID)
- level (int: 0=leaf, 1=first level up, etc.)
- summary_text (LLM-generated summary)
- summary_embedding (vector of summary)
- children_ids (array of node_ids or chunk_ids)
- parent_id (UUID, nullable for root)
- member_count (int: number of chunks under this node)
- last_updated (timestamp)
- is_dirty (boolean: needs re-summarization)

Operations:
 Find nearest leaf: O(log n) - vector search on leaf nodes
 Update node: O(1) - re-summarize single node
 Propagate update: O(tree height) - update ancestors only
```

---

## Database Schema Changes

### New Tables

```sql
-- Base layer: Raw chunks (replaces current Level 0)
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT  -- JSON: {source_file, chunk_index, etc.}
);

CREATE INDEX idx_chunks_doc ON chunks(user_id, doc_id);
CREATE INDEX idx_chunks_deleted ON chunks(user_id, deleted);

-- Tree layer: Hierarchical structure
CREATE TABLE tree_nodes (
    node_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    level INTEGER NOT NULL,  -- 0=leaf, 1+=higher levels
    summary_text TEXT,  -- NULL for leaf nodes (they just group chunks)
    summary_embedding BLOB,  -- NULL for leaf nodes
    children_ids TEXT,  -- JSON array: ["chunk_id1", "chunk_id2"] or ["node_id1", "node_id2"]
    parent_id TEXT,  -- NULL for root nodes
    member_count INTEGER DEFAULT 0,  -- Total chunks under this node
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_dirty BOOLEAN DEFAULT FALSE  -- Needs re-summarization
);

CREATE INDEX idx_tree_level ON tree_nodes(user_id, level);
CREATE INDEX idx_tree_parent ON tree_nodes(user_id, parent_id);
CREATE INDEX idx_tree_dirty ON tree_nodes(user_id, is_dirty);

-- Mapping: Which chunks belong to which leaf nodes
CREATE TABLE chunk_to_leaf (
    chunk_id TEXT NOT NULL,
    leaf_node_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    PRIMARY KEY (chunk_id, leaf_node_id)
);

CREATE INDEX idx_chunk_leaf ON chunk_to_leaf(user_id, leaf_node_id);
```

### Migration from Current Schema

```sql
-- Old schema uses 'documents' table with 'hierarchy_level' column
-- Migration strategy:

-- Step 1: Chunks (Level 0 → chunks table)
INSERT INTO chunks (chunk_id, doc_id, user_id, text, embedding, deleted)
SELECT 
    node_id,
    source_file,
    user_id,
    text,
    embedding,
    FALSE
FROM documents
WHERE hierarchy_level = 0;

-- Step 2: Tree nodes (Level 1+ → tree_nodes table)
INSERT INTO tree_nodes (node_id, user_id, level, summary_text, summary_embedding, children_ids, parent_id, member_count)
SELECT 
    node_id,
    user_id,
    hierarchy_level,
    text,
    embedding,
    children,  -- Already JSON array in current schema
    parent_node_id,
    (SELECT COUNT(*) FROM json_each(children))
FROM documents
WHERE hierarchy_level > 0;

-- Step 3: Create chunk_to_leaf mapping from parent_node_id
INSERT INTO chunk_to_leaf (chunk_id, leaf_node_id, user_id)
SELECT 
    node_id,
    parent_node_id,
    user_id
FROM documents
WHERE hierarchy_level = 0 AND parent_node_id IS NOT NULL;
```

---

## Implementation: Fast Operations

### 1. Fast Document Upload (O(log n))

```python
class IncrementalRaptorBuilder:
    """
    RAPTOR builder with incremental updates
    """
    
    def add_document_incremental(self, doc_path: str, user_id: str):
        """
        Add document without rebuilding entire tree
        Time: O(num_chunks * log n) instead of O(n³)
        """
        # Step 1: Chunk and embed (same as before)
        chunks = self.chunk_document(doc_path)
        embeddings = self.embed_chunks(chunks)
        doc_id = str(uuid.uuid4())
        
        print(f" Processing {len(chunks)} chunks from {doc_path}")
        
        # Step 2: Insert into base layer (fast!)
        chunk_ids = []
        for chunk, emb in zip(chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            self.db.execute("""
                INSERT INTO chunks (chunk_id, doc_id, user_id, text, embedding, deleted)
                VALUES (?, ?, ?, ?, ?, FALSE)
            """, (chunk_id, doc_id, user_id, chunk, emb))
            chunk_ids.append(chunk_id)
        
        print(f" Inserted {len(chunk_ids)} chunks into base layer")
        
        # Step 3: Assign each chunk to a leaf node (local operation)
        affected_nodes = set()
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            leaf_node_id = self.find_or_create_leaf_node(embedding, user_id)
            
            # Add chunk to leaf node's children
            self.db.execute("""
                UPDATE tree_nodes
                SET children_ids = json_insert(children_ids, '$[#]', ?),
                    member_count = member_count + 1,
                    is_dirty = TRUE
                WHERE node_id = ?
            """, (chunk_id, leaf_node_id))
            
            # Record mapping
            self.db.execute("""
                INSERT INTO chunk_to_leaf (chunk_id, leaf_node_id, user_id)
                VALUES (?, ?, ?)
            """, (chunk_id, leaf_node_id, user_id))
            
            affected_nodes.add(leaf_node_id)
        
        print(f" Assigned chunks to {len(affected_nodes)} leaf nodes")
        
        # Step 4: Re-summarize only affected nodes (local updates)
        self.update_affected_branches(affected_nodes, user_id)
        
        print(f" Document added successfully! Tree updated locally.")
        return doc_id
    
    
    def find_or_create_leaf_node(self, chunk_embedding, user_id, threshold=0.7):
        """
        Find nearest leaf node for this chunk, or create new one
        Time: O(log n) - vector search on leaf nodes only
        """
        # Query: Find nearest leaf node (level 0)
        leaf_nodes = self.db.query("""
            SELECT node_id, summary_embedding
            FROM tree_nodes
            WHERE user_id = ? AND level = 0
        """, (user_id,))
        
        if not leaf_nodes:
            # No tree yet - create first leaf node
            return self.create_leaf_node(user_id)
        
        # Find most similar leaf node
        best_similarity = -1
        best_node_id = None
        
        for node_id, node_embedding in leaf_nodes:
            similarity = cosine_similarity(chunk_embedding, node_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_node_id = node_id
        
        # If similarity is high enough, use existing node
        if best_similarity >= threshold:
            return best_node_id
        
        # Otherwise, create new leaf node
        return self.create_leaf_node(user_id)
    
    
    def create_leaf_node(self, user_id):
        """
        Create a new leaf node in the tree
        """
        node_id = str(uuid.uuid4())
        self.db.execute("""
            INSERT INTO tree_nodes (node_id, user_id, level, children_ids, member_count)
            VALUES (?, ?, 0, '[]', 0)
        """, (node_id, user_id))
        return node_id
    
    
    def update_affected_branches(self, affected_leaf_nodes, user_id, max_depth=None):
        """
        Re-summarize only affected nodes and propagate up the tree
        Time: O(num_affected_nodes * tree_height)
        """
        nodes_to_update = set(affected_leaf_nodes)
        
        # Walk up the tree, updating ancestors
        current_level = 0
        while nodes_to_update and (max_depth is None or current_level < max_depth):
            print(f" Updating {len(nodes_to_update)} nodes at level {current_level}")
            
            parent_nodes = set()
            
            for node_id in nodes_to_update:
                # Re-summarize this node
                self.re_summarize_node(node_id, user_id)
                
                # Find parent (to update next level)
                parent = self.db.query_one("""
                    SELECT parent_id FROM tree_nodes WHERE node_id = ?
                """, (node_id,))
                
                if parent and parent[0]:
                    parent_nodes.add(parent[0])
                    # Mark parent as dirty
                    self.db.execute("""
                        UPDATE tree_nodes SET is_dirty = TRUE WHERE node_id = ?
                    """, (parent[0],))
            
            nodes_to_update = parent_nodes
            current_level += 1
        
        print(f" Updated {current_level} levels of the tree")
    
    
    def re_summarize_node(self, node_id, user_id):
        """
        Re-summarize a single node from its children
        Time: O(1) node + O(children) LLM cost
        """
        # Get node info
        node = self.db.query_one("""
            SELECT level, children_ids FROM tree_nodes WHERE node_id = ?
        """, (node_id,))
        
        if not node:
            return
        
        level, children_ids_json = node
        children_ids = json.loads(children_ids_json)
        
        if level == 0:
            # Leaf node: compute centroid of chunk embeddings
            chunk_embeddings = []
            for chunk_id in children_ids:
                chunk = self.db.query_one("""
                    SELECT embedding FROM chunks WHERE chunk_id = ? AND deleted = FALSE
                """, (chunk_id,))
                if chunk:
                    chunk_embeddings.append(chunk[0])
            
            if chunk_embeddings:
                # Centroid of chunk embeddings (for similarity search)
                centroid = np.mean(chunk_embeddings, axis=0)
                self.db.execute("""
                    UPDATE tree_nodes
                    SET summary_embedding = ?,
                        is_dirty = FALSE,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE node_id = ?
                """, (centroid.tobytes(), node_id))
        
        else:
            # Higher-level node: LLM summarization
            # Get children summaries
            children_texts = []
            for child_id in children_ids:
                child = self.db.query_one("""
                    SELECT summary_text FROM tree_nodes WHERE node_id = ?
                """, (child_id,))
                if child and child[0]:
                    children_texts.append(child[0])
            
            if children_texts:
                # LLM summarization
                summary_text = self.summarize_with_llm(children_texts)
                summary_embedding = self.embed_text(summary_text)
                
                self.db.execute("""
                    UPDATE tree_nodes
                    SET summary_text = ?,
                        summary_embedding = ?,
                        is_dirty = FALSE,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE node_id = ?
                """, (summary_text, summary_embedding.tobytes(), node_id))
```

### 2. Fast Document Delete (O(log n))

```python
    def delete_document_incremental(self, doc_id: str, user_id: str):
        """
        Delete document without rebuilding entire tree
        Time: O(num_chunks * log n) instead of O(n³)
        """
        # Step 1: Find all chunks for this document
        chunks = self.db.query("""
            SELECT chunk_id FROM chunks
            WHERE user_id = ? AND doc_id = ? AND deleted = FALSE
        """, (user_id, doc_id))
        
        chunk_ids = [c[0] for c in chunks]
        print(f"️ Deleting {len(chunk_ids)} chunks from document {doc_id}")
        
        # Step 2: Soft delete chunks in base layer
        self.db.execute("""
            UPDATE chunks SET deleted = TRUE
            WHERE user_id = ? AND doc_id = ?
        """, (user_id, doc_id))
        
        # Step 3: Remove from leaf nodes and repair tree
        affected_nodes = set()
        
        for chunk_id in chunk_ids:
            # Find which leaf node contains this chunk
            leaf_mapping = self.db.query_one("""
                SELECT leaf_node_id FROM chunk_to_leaf
                WHERE chunk_id = ?
            """, (chunk_id,))
            
            if not leaf_mapping:
                continue
            
            leaf_node_id = leaf_mapping[0]
            affected_nodes.add(leaf_node_id)
            
            # Remove chunk from leaf node's children
            self.db.execute("""
                UPDATE tree_nodes
                SET children_ids = (
                    SELECT json_group_array(value)
                    FROM json_each(children_ids)
                    WHERE value != ?
                ),
                member_count = member_count - 1,
                is_dirty = TRUE
                WHERE node_id = ?
            """, (chunk_id, leaf_node_id))
            
            # Delete mapping
            self.db.execute("""
                DELETE FROM chunk_to_leaf WHERE chunk_id = ?
            """, (chunk_id,))
        
        print(f" Removed chunks from {len(affected_nodes)} leaf nodes")
        
        # Step 4: Prune empty nodes and repair tree
        self.prune_empty_nodes(affected_nodes, user_id)
        
        # Step 5: Re-summarize affected branches
        remaining_nodes = {n for n in affected_nodes if self.node_exists(n)}
        if remaining_nodes:
            self.update_affected_branches(remaining_nodes, user_id)
        
        print(f" Document deleted successfully! Tree repaired locally.")
    
    
    def prune_empty_nodes(self, affected_nodes, user_id):
        """
        Remove or merge nodes that became too small/empty
        """
        for node_id in affected_nodes:
            node = self.db.query_one("""
                SELECT level, member_count, parent_id, children_ids
                FROM tree_nodes WHERE node_id = ?
            """, (node_id,))
            
            if not node:
                continue
            
            level, member_count, parent_id, children_ids_json = node
            children_ids = json.loads(children_ids_json)
            
            # If node is empty, delete it
            if member_count == 0 or len(children_ids) == 0:
                print(f"️ Pruning empty node {node_id} at level {level}")
                
                # Remove from parent's children
                if parent_id:
                    self.db.execute("""
                        UPDATE tree_nodes
                        SET children_ids = (
                            SELECT json_group_array(value)
                            FROM json_each(children_ids)
                            WHERE value != ?
                        ),
                        is_dirty = TRUE
                        WHERE node_id = ?
                    """, (node_id, parent_id))
                
                # Delete node
                self.db.execute("DELETE FROM tree_nodes WHERE node_id = ?", (node_id,))
            
            # If node is too small, consider merging with sibling
            elif level == 0 and member_count < 3:
                # Find sibling leaf nodes
                siblings = self.db.query("""
                    SELECT node_id, member_count FROM tree_nodes
                    WHERE user_id = ? AND level = 0 AND parent_id = ? AND node_id != ?
                    ORDER BY member_count ASC
                    LIMIT 1
                """, (user_id, parent_id, node_id))
                
                if siblings and siblings[0][1] < 5:  # Sibling also small
                    sibling_id = siblings[0][0]
                    print(f" Merging small nodes {node_id} and {sibling_id}")
                    self.merge_nodes(node_id, sibling_id, user_id)
    
    
    def merge_nodes(self, node1_id, node2_id, user_id):
        """
        Merge two small nodes into one
        """
        # Get children from both nodes
        node1 = self.db.query_one("""
            SELECT children_ids FROM tree_nodes WHERE node_id = ?
        """, (node1_id,))
        node2 = self.db.query_one("""
            SELECT children_ids FROM tree_nodes WHERE node_id = ?
        """, (node2_id,))
        
        children1 = json.loads(node1[0])
        children2 = json.loads(node2[0])
        
        # Merge children into node1
        merged_children = children1 + children2
        
        self.db.execute("""
            UPDATE tree_nodes
            SET children_ids = ?,
                member_count = ?,
                is_dirty = TRUE
            WHERE node_id = ?
        """, (json.dumps(merged_children), len(merged_children), node1_id))
        
        # Update chunk_to_leaf mappings
        self.db.execute("""
            UPDATE chunk_to_leaf
            SET leaf_node_id = ?
            WHERE leaf_node_id = ?
        """, (node1_id, node2_id))
        
        # Delete node2
        self.db.execute("DELETE FROM tree_nodes WHERE node_id = ?", (node2_id,))
```

### 3. Query Using Both Layers

```python
    def query_with_tree_guidance(self, question: str, user_id: str, top_k: int = 5):
        """
        Use tree to guide retrieval, then pull actual chunks
        """
        question_embedding = self.embed_text(question)
        
        # Step 1: Classify question complexity
        question_type = self.classify_question(question)
        
        if question_type == "SUMMARY":
            # Broad question: Search high-level summaries
            target_level = max(self.get_max_level(user_id) - 1, 1)
            print(f" Summary question - searching level {target_level} summaries")
            
            # Find relevant high-level nodes
            nodes = self.db.query("""
                SELECT node_id, summary_text, summary_embedding
                FROM tree_nodes
                WHERE user_id = ? AND level = ?
            """, (user_id, target_level))
            
            # Rank by similarity
            node_similarities = [
                (node_id, cosine_similarity(question_embedding, emb), text)
                for node_id, text, emb in nodes
            ]
            node_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top N node summaries
            top_summaries = [text for _, _, text in node_similarities[:top_k]]
            
            # Synthesize answer from summaries
            return self.synthesize_answer(question, top_summaries)
        
        else:
            # Detailed question: Search leaf nodes, then chunks
            print(f" Detailed question - searching leaf nodes + chunks")
            
            # Step 1: Find relevant leaf nodes
            leaf_nodes = self.db.query("""
                SELECT node_id, summary_embedding FROM tree_nodes
                WHERE user_id = ? AND level = 0
            """, (user_id,))
            
            leaf_similarities = [
                (node_id, cosine_similarity(question_embedding, emb))
                for node_id, emb in leaf_nodes
            ]
            leaf_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Step 2: Get chunks from top leaf nodes
            relevant_chunks = []
            for node_id, _ in leaf_similarities[:3]:  # Top 3 leaf nodes
                chunks = self.db.query("""
                    SELECT c.chunk_id, c.text, c.embedding
                    FROM chunks c
                    JOIN chunk_to_leaf ctl ON c.chunk_id = ctl.chunk_id
                    WHERE ctl.leaf_node_id = ? AND c.deleted = FALSE
                """, (node_id,))
                relevant_chunks.extend(chunks)
            
            # Step 3: Rank chunks by similarity to question
            chunk_similarities = [
                (chunk_id, text, cosine_similarity(question_embedding, emb))
                for chunk_id, text, emb in relevant_chunks
            ]
            chunk_similarities.sort(key=lambda x: x[2], reverse=True)
            
            # Step 4: Get top K chunks
            top_chunks = [text for _, text, _ in chunk_similarities[:top_k]]
            
            # Synthesize answer
            return self.synthesize_answer(question, top_chunks)
```

---

## Performance Comparison

### Current System (Full Rebuild)
```
Operation         | Time Complexity | Example (1000 chunks)
------------------|-----------------|----------------------
Add 1 doc (10 ch) | O(n³)          | ~30 seconds (full rebuild)
Add 10 docs       | O(n³)          | ~35 seconds (full rebuild)
Delete 1 doc      | O(n³)          | ~30 seconds (full rebuild)
Query             | O(log n)       | ~100ms
```

### New System (Incremental Updates)
```
Operation         | Time Complexity        | Example (1000 chunks)
------------------|------------------------|----------------------
Add 1 doc (10 ch) | O(k log n + h*k)      | ~2 seconds (10 chunks × log search + tree update)
Add 10 docs       | O(k log n + h*k)      | ~15 seconds (100 chunks × log search)
Delete 1 doc      | O(k log n + h*k)      | ~1 second (soft delete + prune)
Query             | O(log n)               | ~100ms (same or faster with tree guidance)

Where:
- k = num chunks in operation
- h = tree height (~3-4)
- n = total chunks in corpus
```

### Speedup
- **Add document: 15x faster** (2s vs 30s)
- **Delete document: 30x faster** (1s vs 30s)
- **Query: Same or faster** (tree guides to relevant chunks)

---

## Migration Path

### Phase 1: Schema Update (Low Risk)
```bash
# Run migration script
python migrate_to_incremental.py

# Validates:
# - New tables created
# - Data migrated from documents → chunks + tree_nodes
# - Old data preserved (can rollback)
```

### Phase 2: Implement Incremental Builder (Medium Risk)
```bash
# New class: IncrementalRaptorBuilder
# Coexists with old RaptorBuilder

# Test with:
python test_incremental.py

# Validates:
# - Add document works
# - Delete document works
# - Tree structure maintained
# - Query results match old system
```

### Phase 3: Switch to Incremental (Low Risk)
```bash
# Update batch_ingest.py to use IncrementalRaptorBuilder
# Users see immediate speedup

# Rollback plan:
# - If issues, switch back to RaptorBuilder
# - Data compatible with both systems
```

---

## Next Steps

1. **Implement schema migration** (`migrate_to_incremental.py`)
2. **Build IncrementalRaptorBuilder** class with add/delete methods
3. **Test on sample corpus** (validate correctness + speed)
4. **Benchmark performance** (compare old vs new)
5. **Deploy to production** (gradual rollout)

**Estimated Implementation Time:** 2-3 days

**Expected Impact:**
-  15-30x faster document updates
-  Same or better query quality
-  Enables real-time corpus updates
-  Foundation for multi-user system

---

**Status:** Design complete, ready for implementation
**Risk Level:** Medium (requires schema changes, but rollback possible)
**User Impact:** Massive improvement in update speed, no downtime
