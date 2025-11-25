"""
Incremental RAPTOR Implementation with 2-Layer Architecture
Enables fast document add/delete without full tree rebuild
"""
import numpy as np
from typing import List, Dict, Set, Optional, Tuple
import uuid
import json
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from tqdm import tqdm

load_dotenv()


class IncrementalRaptorBuilder:
    """
    Fast incremental RAPTOR updates using decoupled architecture:
    - Layer 1: chunks (base truth, raw embeddings)
    - Layer 2: tree_nodes (hierarchical overlay)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 gemini_api_key: str = None):
        self.model = SentenceTransformer(model_name)
        
        self.db_config = {
            'user': os.getenv("user"),
            'password': os.getenv("password"),
            'host': os.getenv("host"),
            'port': int(os.getenv("port", "5432")),
            'dbname': os.getenv("dbname")
        }
        
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    
    
    def _get_conn(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)
    
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    
    # ============================================================
    # FAST DOCUMENT ADD (O(log n))
    # ============================================================
    
    def add_document_incremental(self, chunks: List[Dict], doc_id: str, user_id: str, 
                                  source_file: str = None,
                                  similarity_threshold: float = 0.7) -> str:
        """
        Add document without rebuilding entire tree
        Time: O(num_chunks * log n) instead of O(n³)
        
        Args:
            chunks: List of dicts with 'id' and 'text' keys
            doc_id: Document UUID
            user_id: User UUID
            source_file: Optional filename for metadata (defaults to doc_id if not provided)
            similarity_threshold: Threshold for assigning to existing leaf nodes
        
        Returns:
            doc_id
        """
        print(f"\n Adding document {doc_id} with {len(chunks)} chunks...")
        
        # Step 1: Embed all chunks
        print("   Generating embeddings...")
        texts = [c['text'] for c in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        # Step 2: Insert into base layer (chunks table)
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                chunk_data = []
                for chunk, emb in zip(chunks, embeddings):
                    chunk_data.append((
                        chunk['id'],
                        doc_id,
                        user_id,
                        chunk['text'],
                        emb.tolist(),
                        False,  # not deleted
                        json.dumps({'source_file': source_file or doc_id})
                    ))
                
                execute_values(cur, """
                    INSERT INTO chunks (chunk_id, doc_id, user_id, text, embedding, deleted, metadata)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO NOTHING
                """, chunk_data)
                
                conn.commit()
                print(f" Inserted {len(chunks)} chunks into base layer")
            
            # Step 3: Assign each chunk to a leaf node (local operation)
            affected_nodes = set()
            
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                print("   Assigning chunks to tree nodes...")
                for chunk, emb in tqdm(zip(chunks, embeddings), total=len(chunks), desc="  Assigning"):
                    leaf_node_id = self.find_or_create_leaf_node(
                        cur, emb, user_id, similarity_threshold
                    )
                    
                    # Add chunk to leaf node's children
                    cur.execute("""
                        UPDATE tree_nodes
                        SET children_ids = children_ids || %s::jsonb,
                            member_count = member_count + 1,
                            is_dirty = TRUE
                        WHERE node_id = %s
                    """, (json.dumps([chunk['id']]), leaf_node_id))
                    
                    # Record mapping
                    cur.execute("""
                        INSERT INTO chunk_to_leaf (chunk_id, leaf_node_id, user_id)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (chunk_id, leaf_node_id) DO NOTHING
                    """, (chunk['id'], leaf_node_id, user_id))
                    
                    affected_nodes.add(leaf_node_id)
                
                conn.commit()
                print(f" Assigned chunks to {len(affected_nodes)} leaf nodes")
            
            # Step 4: Re-summarize only affected branches (local updates)
            self.update_affected_branches(conn, affected_nodes, user_id)
            
            print(f" Document added successfully! Tree updated locally.")
            
        finally:
            conn.close()
        
        return doc_id
    
    
    def find_or_create_leaf_node(self, cur, chunk_embedding: np.ndarray, 
                                  user_id: str, threshold: float = 0.7) -> str:
        """
        Find nearest leaf node for this chunk, or create new one
        Time: O(num_leaf_nodes) - could be optimized with vector index
        
        Args:
            cur: Database cursor
            chunk_embedding: Embedding vector
            user_id: User UUID
            threshold: Similarity threshold
        
        Returns:
            leaf_node_id (UUID)
        """
        # Query all leaf nodes (level 0)
        cur.execute("""
            SELECT node_id, summary_embedding
            FROM tree_nodes
            WHERE user_id = %s AND level = 0
                AND summary_embedding IS NOT NULL
        """, (user_id,))
        
        leaf_nodes = cur.fetchall()
        
        if not leaf_nodes:
            # No tree yet - create first leaf node
            return self.create_leaf_node(cur, user_id)
        
        # Find most similar leaf node
        best_similarity = -1
        best_node_id = None
        
        for row in leaf_nodes:
            node_id = row['node_id']
            node_emb_bytes = row['summary_embedding']
            
            # Convert pgvector to numpy array
            # Supabase stores as string like '[0.1, 0.2, ...]'
            try:
                if isinstance(node_emb_bytes, str):
                    node_emb = np.array(json.loads(node_emb_bytes))
                elif isinstance(node_emb_bytes, list):
                    node_emb = np.array(node_emb_bytes)
                else:
                    # bytes or memoryview
                    node_emb = np.frombuffer(node_emb_bytes, dtype=np.float32)
            except Exception as e:
                print(f"Warning: Could not parse embedding for node {node_id}: {e}")
                continue
            
            similarity = self.cosine_similarity(chunk_embedding, node_emb)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_node_id = node_id
        
        # If similarity is high enough, use existing node
        if best_similarity >= threshold:
            return best_node_id
        
        # Otherwise, create new leaf node
        return self.create_leaf_node(cur, user_id)
    
    
    def create_leaf_node(self, cur, user_id: str, parent_id: Optional[str] = None) -> str:
        """
        Create a new leaf node in the tree
        
        Args:
            cur: Database cursor
            user_id: User UUID
            parent_id: Optional parent node UUID
        
        Returns:
            node_id (UUID)
        """
        node_id = str(uuid.uuid4())
        
        cur.execute("""
            INSERT INTO tree_nodes (node_id, user_id, level, children_ids, member_count, parent_id)
            VALUES (%s, %s, 0, '[]'::jsonb, 0, %s)
        """, (node_id, user_id, parent_id))
        
        return node_id
    
    
    def update_affected_branches(self, conn, affected_leaf_nodes: Set[str], 
                                  user_id: str, max_depth: Optional[int] = None):
        """
        Re-summarize only affected nodes and propagate up the tree
        Time: O(num_affected_nodes * tree_height)
        
        Args:
            conn: Database connection
            affected_leaf_nodes: Set of leaf node UUIDs that changed
            user_id: User UUID
            max_depth: Optional max levels to propagate (None = all levels)
        """
        nodes_to_update = set(affected_leaf_nodes)
        current_level = 0
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            while nodes_to_update and (max_depth is None or current_level < max_depth):
                print(f" Updating {len(nodes_to_update)} nodes at level {current_level}")
                
                parent_nodes = set()
                
                for node_id in nodes_to_update:
                    # Re-summarize this node
                    parent_id = self.re_summarize_node(cur, node_id, user_id)
                    
                    # Add parent to next level's update list
                    if parent_id:
                        parent_nodes.add(parent_id)
                        # Mark parent as dirty
                        cur.execute("""
                            UPDATE tree_nodes SET is_dirty = TRUE WHERE node_id = %s
                        """, (parent_id,))
                
                conn.commit()
                nodes_to_update = parent_nodes
                current_level += 1
            
            print(f" Updated {current_level} levels of the tree")
    
    
    def re_summarize_node(self, cur, node_id: str, user_id: str) -> Optional[str]:
        """
        Re-summarize a single node from its children
        Time: O(1) query + O(children) LLM cost
        
        Args:
            cur: Database cursor
            node_id: Node UUID
            user_id: User UUID
        
        Returns:
            parent_id (UUID) or None
        """
        # Get node info
        cur.execute("""
            SELECT level, children_ids, parent_id FROM tree_nodes WHERE node_id = %s
        """, (node_id,))
        
        row = cur.fetchone()
        if not row:
            return None
        
        level = row['level']
        children_ids = row['children_ids']
        parent_id = row['parent_id']
        
        if level == 0:
            # Leaf node: compute centroid of chunk embeddings
            if not children_ids:
                return parent_id
            
            chunk_embeddings = []
            for chunk_id in children_ids:
                cur.execute("""
                    SELECT embedding FROM chunks 
                    WHERE chunk_id = %s AND deleted = FALSE
                """, (chunk_id,))
                chunk_row = cur.fetchone()
                if chunk_row and chunk_row['embedding']:
                    emb = chunk_row['embedding']
                    # Convert to numpy
                    try:
                        if isinstance(emb, str):
                            emb_array = np.array(json.loads(emb))
                        elif isinstance(emb, list):
                            emb_array = np.array(emb)
                        else:
                            emb_array = np.frombuffer(emb, dtype=np.float32)
                        chunk_embeddings.append(emb_array)
                    except Exception as e:
                        print(f"Warning: Could not parse embedding for chunk {chunk_id}: {e}")
                        continue
            
            if chunk_embeddings:
                # Centroid of chunk embeddings
                centroid = np.mean(chunk_embeddings, axis=0)
                cur.execute("""
                    UPDATE tree_nodes
                    SET summary_embedding = %s,
                        is_dirty = FALSE,
                        last_updated = NOW()
                    WHERE node_id = %s
                """, (centroid.tolist(), node_id))
        
        else:
            # Higher-level node: LLM summarization
            if not children_ids:
                return parent_id
            
            children_texts = []
            for child_id in children_ids:
                cur.execute("""
                    SELECT summary_text FROM tree_nodes WHERE node_id = %s
                """, (child_id,))
                child_row = cur.fetchone()
                if child_row and child_row['summary_text']:
                    children_texts.append(child_row['summary_text'])
            
            if children_texts:
                # LLM summarization
                summary_text = self.summarize_cluster(children_texts)
                summary_embedding = self.model.encode([summary_text], show_progress_bar=False)[0]
                
                cur.execute("""
                    UPDATE tree_nodes
                    SET summary_text = %s,
                        summary_embedding = %s,
                        is_dirty = FALSE,
                        last_updated = NOW()
                    WHERE node_id = %s
                """, (summary_text, summary_embedding.tolist(), node_id))
        
        return parent_id
    
    
    def summarize_cluster(self, chunk_texts: List[str]) -> str:
        """
        Summarize cluster with Gemini (same as original RAPTOR)
        """
        combined = "\n\n---\n\n".join(chunk_texts)
        
        if len(combined) > 8000:
            combined = combined[:8000] + "...[truncated]"
        
        prompt = f"""Summarize this research content, preserving key details:

{combined}

Provide a 2-3 paragraph summary:"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Warning: Summarization error: {e}")
            return combined[:2000]
    
    
    # ============================================================
    # FAST DOCUMENT DELETE (O(log n))
    # ============================================================
    
    def delete_document_incremental(self, doc_id: str, user_id: str):
        """
        Delete document without rebuilding entire tree
        Time: O(num_chunks * log n) instead of O(n³)
        
        Args:
            doc_id: Document UUID
            user_id: User UUID
        """
        print(f"\n️ Deleting document {doc_id}...")
        
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Step 1: Find all chunks for this document
                cur.execute("""
                    SELECT chunk_id FROM chunks
                    WHERE user_id = %s AND doc_id = %s AND deleted = FALSE
                """, (user_id, doc_id))
                
                chunks = cur.fetchall()
                chunk_ids = [c['chunk_id'] for c in chunks]
                
                if not chunk_ids:
                    print(f"️ No chunks found for document {doc_id}")
                    return
                
                print(f"️ Deleting {len(chunk_ids)} chunks...")
                
                # Step 2: Soft delete chunks in base layer
                cur.execute("""
                    UPDATE chunks SET deleted = TRUE
                    WHERE user_id = %s AND doc_id = %s
                """, (user_id, doc_id))
                
                # Step 3: Remove from leaf nodes and collect affected nodes
                affected_nodes = set()
                
                for chunk_id in chunk_ids:
                    # Find which leaf node contains this chunk
                    cur.execute("""
                        SELECT leaf_node_id FROM chunk_to_leaf
                        WHERE chunk_id = %s
                    """, (chunk_id,))
                    
                    leaf_row = cur.fetchone()
                    if not leaf_row:
                        continue
                    
                    leaf_node_id = leaf_row['leaf_node_id']
                    affected_nodes.add(leaf_node_id)
                    
                    # Remove chunk from leaf node's children
                    cur.execute("""
                        UPDATE tree_nodes
                        SET children_ids = (
                            SELECT jsonb_agg(elem)
                            FROM jsonb_array_elements(children_ids) elem
                            WHERE elem::text != %s::text
                        ),
                        member_count = member_count - 1,
                        is_dirty = TRUE
                        WHERE node_id = %s
                    """, (f'"{chunk_id}"', leaf_node_id))
                    
                    # Delete mapping
                    cur.execute("""
                        DELETE FROM chunk_to_leaf WHERE chunk_id = %s
                    """, (chunk_id,))
                
                conn.commit()
                print(f" Removed chunks from {len(affected_nodes)} leaf nodes")
                
                # Step 4: Prune empty nodes and repair tree
                remaining_nodes = self.prune_empty_nodes(cur, affected_nodes, user_id)
                conn.commit()
                
                # Step 5: Re-summarize affected branches
                if remaining_nodes:
                    self.update_affected_branches(conn, remaining_nodes, user_id)
                
                print(f" Document deleted successfully! Tree repaired locally.")
        
        finally:
            conn.close()
    
    
    def prune_empty_nodes(self, cur, affected_nodes: Set[str], user_id: str) -> Set[str]:
        """
        Remove or merge nodes that became too small/empty
        
        Args:
            cur: Database cursor
            affected_nodes: Set of node UUIDs that were affected
            user_id: User UUID
        
        Returns:
            Set of remaining nodes that need re-summarization
        """
        remaining_nodes = set()
        
        for node_id in affected_nodes:
            cur.execute("""
                SELECT level, member_count, parent_id, children_ids
                FROM tree_nodes WHERE node_id = %s
            """, (node_id,))
            
            row = cur.fetchone()
            if not row:
                continue
            
            level = row['level']
            member_count = row['member_count']
            parent_id = row['parent_id']
            children_ids = row['children_ids']
            
            # If node is empty, delete it
            if member_count == 0 or not children_ids:
                print(f"️ Pruning empty node {node_id} at level {level}")
                
                # Remove from parent's children
                if parent_id:
                    cur.execute("""
                        UPDATE tree_nodes
                        SET children_ids = (
                            SELECT jsonb_agg(elem)
                            FROM jsonb_array_elements(children_ids) elem
                            WHERE elem::text != %s::text
                        ),
                        is_dirty = TRUE
                        WHERE node_id = %s
                    """, (f'"{node_id}"', parent_id))
                
                # Delete node
                cur.execute("DELETE FROM tree_nodes WHERE node_id = %s", (node_id,))
            
            else:
                # Node still has members - keep it for re-summarization
                remaining_nodes.add(node_id)
        
        return remaining_nodes
    
    
    # ============================================================
    # HELPER: Get tree statistics
    # ============================================================
    
    def get_tree_stats(self, user_id: str) -> Dict:
        """
        Get statistics about the tree structure
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM get_tree_stats(%s)", (user_id,))
                row = cur.fetchone()
                return dict(row) if row else {}
        finally:
            conn.close()
