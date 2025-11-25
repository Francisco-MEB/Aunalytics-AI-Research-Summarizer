"""
Database Cleanup Utility for RAPTOR System
==========================================
Handles cleanup when document ingestion is interrupted mid-way,
leaving the database in an inconsistent state.

Use this tool when:
- Document ingestion was interrupted (Ctrl+C, connection loss, crash)
- Tree hierarchy seems corrupted
- Want to remove a specific document completely
- Want to reset the entire system for a user

Author: AI Research Summarizer Team
"""
import os
import sys
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from tqdm import tqdm

load_dotenv()


class DatabaseCleaner:
    """
    Cleanup utility for RAPTOR database
    
    Handles:
    - Orphaned chunks (chunks without tree node mappings)
    - Orphaned tree nodes (empty nodes with no children)
    - Partial document ingestion cleanup
    - Full user data reset
    - Integrity validation
    """
    
    def __init__(self):
        self.db_config = {
            'user': os.getenv("user"),
            'password': os.getenv("password"),
            'host': os.getenv("host"),
            'port': int(os.getenv("port", "5432")),
            'dbname': os.getenv("dbname")
        }
    
    def _get_conn(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)
    
    # ============================================================
    # VALIDATION - Check for Issues
    # ============================================================
    
    def validate_integrity(self, user_id: str) -> Dict:
        """
        Check database integrity and report issues
        
        Returns dict with:
        - orphaned_chunks: Chunks not mapped to any leaf node
        - empty_tree_nodes: Tree nodes with no children
        - dangling_mappings: Mappings pointing to non-existent chunks/nodes
        - dirty_nodes: Nodes that need re-summarization
        - partial_docs: Documents that may be incomplete
        """
        print("\n Validating database integrity...")
        
        issues = {
            'orphaned_chunks': [],
            'empty_tree_nodes': [],
            'dangling_mappings': [],
            'dirty_nodes': [],
            'partial_docs': [],
            'chunks_without_embeddings': []
        }
        
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # 1. Find orphaned chunks (not mapped to any leaf node)
                cur.execute("""
                    SELECT c.chunk_id, c.doc_id, c.metadata
                    FROM chunks c
                    LEFT JOIN chunk_to_leaf ctl ON c.chunk_id = ctl.chunk_id
                    WHERE c.user_id = %s 
                      AND c.deleted = FALSE 
                      AND ctl.chunk_id IS NULL
                """, (user_id,))
                issues['orphaned_chunks'] = [dict(r) for r in cur.fetchall()]
                
                # 2. Find empty tree nodes (member_count = 0 or no children)
                cur.execute("""
                    SELECT node_id, level, member_count, children_ids
                    FROM tree_nodes
                    WHERE user_id = %s 
                      AND (member_count = 0 OR children_ids = '[]'::jsonb OR children_ids IS NULL)
                """, (user_id,))
                issues['empty_tree_nodes'] = [dict(r) for r in cur.fetchall()]
                
                # 3. Find dangling mappings (pointing to deleted/non-existent chunks)
                cur.execute("""
                    SELECT ctl.chunk_id, ctl.leaf_node_id
                    FROM chunk_to_leaf ctl
                    LEFT JOIN chunks c ON ctl.chunk_id = c.chunk_id
                    WHERE ctl.user_id = %s 
                      AND (c.chunk_id IS NULL OR c.deleted = TRUE)
                """, (user_id,))
                issues['dangling_mappings'] = [dict(r) for r in cur.fetchall()]
                
                # 4. Find dirty nodes (need re-summarization)
                cur.execute("""
                    SELECT node_id, level, is_dirty, summary_text IS NULL as missing_summary
                    FROM tree_nodes
                    WHERE user_id = %s AND is_dirty = TRUE
                """, (user_id,))
                issues['dirty_nodes'] = [dict(r) for r in cur.fetchall()]
                
                # 5. Check for partial documents (based on recent inserts with issues)
                cur.execute("""
                    SELECT 
                        doc_id,
                        metadata->>'source_file' as source_file,
                        COUNT(*) as chunk_count,
                        COUNT(CASE WHEN embedding IS NULL THEN 1 END) as missing_embeddings
                    FROM chunks
                    WHERE user_id = %s AND deleted = FALSE
                    GROUP BY doc_id, metadata->>'source_file'
                    HAVING COUNT(CASE WHEN embedding IS NULL THEN 1 END) > 0
                """, (user_id,))
                issues['partial_docs'] = [dict(r) for r in cur.fetchall()]
                
                # 6. Chunks without embeddings
                cur.execute("""
                    SELECT chunk_id, doc_id, metadata
                    FROM chunks
                    WHERE user_id = %s 
                      AND deleted = FALSE 
                      AND embedding IS NULL
                """, (user_id,))
                issues['chunks_without_embeddings'] = [dict(r) for r in cur.fetchall()]
                
        finally:
            conn.close()
        
        # Print summary
        print("\n" + "="*60)
        print("INTEGRITY CHECK RESULTS")
        print("="*60)
        
        has_issues = False
        
        if issues['orphaned_chunks']:
            has_issues = True
            print(f"\n️  Orphaned Chunks: {len(issues['orphaned_chunks'])}")
            print("   (Chunks not linked to any tree node)")
            for c in issues['orphaned_chunks'][:5]:
                # Handle metadata that might already be a dict or a string
                if isinstance(c['metadata'], dict):
                    meta = c['metadata']
                elif c['metadata']:
                    meta = json.loads(c['metadata'])
                else:
                    meta = {}
                print(f"   - {c['chunk_id'][:8]}... from {meta.get('source_file', 'unknown')}")
            if len(issues['orphaned_chunks']) > 5:
                print(f"   ... and {len(issues['orphaned_chunks']) - 5} more")
        
        if issues['empty_tree_nodes']:
            has_issues = True
            print(f"\n️  Empty Tree Nodes: {len(issues['empty_tree_nodes'])}")
            print("   (Nodes with no children)")
            for n in issues['empty_tree_nodes'][:5]:
                print(f"   - Level {n['level']}: {n['node_id'][:8]}...")
            if len(issues['empty_tree_nodes']) > 5:
                print(f"   ... and {len(issues['empty_tree_nodes']) - 5} more")
        
        if issues['dangling_mappings']:
            has_issues = True
            print(f"\n️  Dangling Mappings: {len(issues['dangling_mappings'])}")
            print("   (References to deleted/missing chunks)")
        
        if issues['dirty_nodes']:
            has_issues = True
            print(f"\n️  Dirty Nodes: {len(issues['dirty_nodes'])}")
            print("   (Nodes needing re-summarization)")
            for n in issues['dirty_nodes'][:5]:
                status = "missing summary" if n['missing_summary'] else "outdated"
                print(f"   - Level {n['level']}: {n['node_id'][:8]}... ({status})")
        
        if issues['partial_docs']:
            has_issues = True
            print(f"\n️  Partial Documents: {len(issues['partial_docs'])}")
            print("   (Documents with missing embeddings)")
            for d in issues['partial_docs']:
                print(f"   - {d['source_file']}: {d['missing_embeddings']}/{d['chunk_count']} missing")
        
        if issues['chunks_without_embeddings']:
            has_issues = True
            print(f"\n️  Chunks Without Embeddings: {len(issues['chunks_without_embeddings'])}")
        
        if not has_issues:
            print("\n No integrity issues found!")
        
        return issues
    
    # ============================================================
    # CLEANUP OPERATIONS
    # ============================================================
    
    def cleanup_orphaned_chunks(self, user_id: str, dry_run: bool = True) -> int:
        """
        Remove or delete orphaned chunks (not mapped to any tree node)
        
        Args:
            user_id: User UUID
            dry_run: If True, only report what would be deleted
            
        Returns:
            Number of chunks affected
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT c.chunk_id, c.doc_id, c.metadata
                    FROM chunks c
                    LEFT JOIN chunk_to_leaf ctl ON c.chunk_id = ctl.chunk_id
                    WHERE c.user_id = %s 
                      AND c.deleted = FALSE 
                      AND ctl.chunk_id IS NULL
                """, (user_id,))
                
                orphans = cur.fetchall()
                
                if not orphans:
                    print(" No orphaned chunks found")
                    return 0
                
                print(f"\n{'Would delete' if dry_run else 'Deleting'} {len(orphans)} orphaned chunks...")
                
                if not dry_run:
                    chunk_ids = [o['chunk_id'] for o in orphans]
                    # Delete in batches to avoid issues with large arrays
                    # Cast to uuid type for comparison
                    for chunk_id in tqdm(chunk_ids, desc="  Deleting orphans", unit="chunk"):
                        cur.execute("""
                            UPDATE chunks SET deleted = TRUE
                            WHERE chunk_id = %s::uuid
                        """, (str(chunk_id),))
                    conn.commit()
                    print(f" Soft-deleted {len(orphans)} orphaned chunks")
                
                return len(orphans)
                
        finally:
            conn.close()
    
    def cleanup_empty_tree_nodes(self, user_id: str, dry_run: bool = True) -> int:
        """
        Remove empty tree nodes (no children)
        
        Args:
            user_id: User UUID
            dry_run: If True, only report what would be deleted
            
        Returns:
            Number of nodes removed
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT node_id, level, parent_id
                    FROM tree_nodes
                    WHERE user_id = %s 
                      AND (member_count = 0 OR children_ids = '[]'::jsonb OR children_ids IS NULL)
                """, (user_id,))
                
                empty_nodes = cur.fetchall()
                
                if not empty_nodes:
                    print(" No empty tree nodes found")
                    return 0
                
                print(f"\n{'Would delete' if dry_run else 'Deleting'} {len(empty_nodes)} empty tree nodes...")
                
                if not dry_run:
                    for node in tqdm(empty_nodes, desc="  Pruning nodes", unit="node"):
                        # Remove from parent's children list
                        if node['parent_id']:
                            cur.execute("""
                                UPDATE tree_nodes
                                SET children_ids = (
                                    SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
                                    FROM jsonb_array_elements(children_ids) elem
                                    WHERE elem::text != %s::text
                                ),
                                is_dirty = TRUE
                                WHERE node_id = %s
                            """, (f'"{node["node_id"]}"', node['parent_id']))
                        
                        # Delete the empty node
                        cur.execute("DELETE FROM tree_nodes WHERE node_id = %s", (node['node_id'],))
                    
                    conn.commit()
                    print(f" Deleted {len(empty_nodes)} empty tree nodes")
                
                return len(empty_nodes)
                
        finally:
            conn.close()
    
    def cleanup_dangling_mappings(self, user_id: str, dry_run: bool = True) -> int:
        """
        Remove mappings that point to deleted/non-existent chunks
        
        Args:
            user_id: User UUID
            dry_run: If True, only report what would be deleted
            
        Returns:
            Number of mappings removed
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT ctl.chunk_id, ctl.leaf_node_id
                    FROM chunk_to_leaf ctl
                    LEFT JOIN chunks c ON ctl.chunk_id = c.chunk_id
                    WHERE ctl.user_id = %s 
                      AND (c.chunk_id IS NULL OR c.deleted = TRUE)
                """, (user_id,))
                
                dangling = cur.fetchall()
                
                if not dangling:
                    print(" No dangling mappings found")
                    return 0
                
                print(f"\n{'Would delete' if dry_run else 'Deleting'} {len(dangling)} dangling mappings...")
                
                if not dry_run:
                    for m in tqdm(dangling, desc="  Cleaning mappings", unit="map"):
                        cur.execute("""
                            DELETE FROM chunk_to_leaf 
                            WHERE chunk_id = %s AND leaf_node_id = %s
                        """, (m['chunk_id'], m['leaf_node_id']))
                        
                        # Also remove from tree node's children
                        cur.execute("""
                            UPDATE tree_nodes
                            SET children_ids = (
                                SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
                                FROM jsonb_array_elements(children_ids) elem
                                WHERE elem::text != %s::text
                            ),
                            member_count = GREATEST(member_count - 1, 0),
                            is_dirty = TRUE
                            WHERE node_id = %s
                        """, (f'"{m["chunk_id"]}"', m['leaf_node_id']))
                    
                    conn.commit()
                    print(f" Deleted {len(dangling)} dangling mappings")
                
                return len(dangling)
                
        finally:
            conn.close()
    
    def cleanup_document(self, user_id: str, doc_id: str = None, 
                         source_file: str = None, dry_run: bool = True) -> int:
        """
        Completely remove a specific document (by doc_id or source_file)
        
        Use this when a document ingestion was interrupted and you want to
        clean it up and start fresh.
        
        Args:
            user_id: User UUID
            doc_id: Document UUID (optional if source_file provided)
            source_file: Source filename (optional if doc_id provided)
            dry_run: If True, only report what would be deleted
            
        Returns:
            Number of chunks removed
        """
        if not doc_id and not source_file:
            print(" Must provide either doc_id or source_file")
            return 0
        
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Find the doc_id if only source_file provided
                if not doc_id:
                    cur.execute("""
                        SELECT DISTINCT doc_id 
                        FROM chunks 
                        WHERE user_id = %s 
                          AND metadata->>'source_file' = %s
                          AND deleted = FALSE
                    """, (user_id, source_file))
                    rows = cur.fetchall()
                    if not rows:
                        print(f" No document found with source_file: {source_file}")
                        return 0
                    doc_ids = [r['doc_id'] for r in rows]
                else:
                    doc_ids = [doc_id]
                
                total_chunks = 0
                
                for did in doc_ids:
                    # Find all chunks
                    cur.execute("""
                        SELECT chunk_id FROM chunks
                        WHERE user_id = %s AND doc_id = %s AND deleted = FALSE
                    """, (user_id, did))
                    chunks = cur.fetchall()
                    chunk_ids = [c['chunk_id'] for c in chunks]
                    
                    if not chunk_ids:
                        continue
                    
                    print(f"\n{'Would delete' if dry_run else 'Deleting'} document {did[:8]}... ({len(chunk_ids)} chunks)")
                    
                    if not dry_run:
                        # 1. Get affected leaf nodes
                        cur.execute("""
                            SELECT DISTINCT leaf_node_id FROM chunk_to_leaf
                            WHERE chunk_id = ANY(%s)
                        """, (chunk_ids,))
                        leaf_nodes = [r['leaf_node_id'] for r in cur.fetchall()]
                        
                        # 2. Delete mappings
                        cur.execute("""
                            DELETE FROM chunk_to_leaf WHERE chunk_id = ANY(%s)
                        """, (chunk_ids,))
                        
                        # 3. Update tree nodes (remove chunks from children) with progress
                        total_ops = len(leaf_nodes) * len(chunk_ids)
                        if total_ops > 0:
                            with tqdm(total=total_ops, desc="  Updating tree", unit="op") as pbar:
                                for leaf_id in leaf_nodes:
                                    for chunk_id in chunk_ids:
                                        cur.execute("""
                                            UPDATE tree_nodes
                                            SET children_ids = (
                                                SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
                                                FROM jsonb_array_elements(children_ids) elem
                                                WHERE elem::text != %s::text
                                            ),
                                            member_count = GREATEST(member_count - 1, 0),
                                            is_dirty = TRUE
                                            WHERE node_id = %s
                                        """, (f'"{chunk_id}"', leaf_id))
                                        pbar.update(1)
                        
                        # 4. Soft delete chunks
                        for chunk_id in tqdm(chunk_ids, desc="  Deleting chunks", unit="chunk"):
                            cur.execute("""
                                UPDATE chunks SET deleted = TRUE
                                WHERE chunk_id = %s::uuid
                            """, (str(chunk_id),))
                        
                        conn.commit()
                        print(f" Deleted document {did[:8]}... ({len(chunk_ids)} chunks)")
                    
                    total_chunks += len(chunk_ids)
                
                return total_chunks
                
        finally:
            conn.close()
    
    def run_full_cleanup(self, user_id: str, dry_run: bool = True) -> Dict:
        """
        Run all cleanup operations in the correct order
        
        Args:
            user_id: User UUID
            dry_run: If True, only report what would be done
            
        Returns:
            Summary of cleanup actions
        """
        print("\n" + "="*60)
        print(f"FULL DATABASE CLEANUP {'(DRY RUN)' if dry_run else ''}")
        print("="*60)
        
        results = {
            'dangling_mappings': self.cleanup_dangling_mappings(user_id, dry_run),
            'orphaned_chunks': self.cleanup_orphaned_chunks(user_id, dry_run),
            'empty_tree_nodes': self.cleanup_empty_tree_nodes(user_id, dry_run),
        }
        
        print("\n" + "-"*60)
        print("CLEANUP SUMMARY")
        print("-"*60)
        print(f"Dangling mappings: {results['dangling_mappings']}")
        print(f"Orphaned chunks: {results['orphaned_chunks']}")
        print(f"Empty tree nodes: {results['empty_tree_nodes']}")
        
        if dry_run:
            print("\n️  This was a DRY RUN. No changes were made.")
            print("   Run with --execute to apply changes.")
        else:
            print("\n Cleanup complete!")
        
        return results
    
    def reset_user_data(self, user_id: str, confirm: bool = False) -> bool:
        """
        ️ DANGEROUS: Completely reset all data for a user
        
        Use this when you want to start completely fresh.
        
        Args:
            user_id: User UUID
            confirm: Must be True to actually execute
            
        Returns:
            True if reset was performed
        """
        if not confirm:
            print("\n️  WARNING: This will DELETE ALL DATA for this user!")
            print("   - All chunks")
            print("   - All tree nodes")
            print("   - All mappings")
            print("\n   To confirm, run with --confirm flag")
            return False
        
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                print("\n️ Deleting all user data...")
                
                # Delete in order to respect references
                cur.execute("DELETE FROM chunk_to_leaf WHERE user_id = %s", (user_id,))
                print("   - Deleted chunk_to_leaf mappings")
                
                cur.execute("DELETE FROM tree_nodes WHERE user_id = %s", (user_id,))
                print("   - Deleted tree_nodes")
                
                cur.execute("DELETE FROM chunks WHERE user_id = %s", (user_id,))
                print("   - Deleted chunks")
                
                conn.commit()
                print("\n All user data has been reset!")
                return True
                
        finally:
            conn.close()
    
    def list_documents(self, user_id: str) -> List[Dict]:
        """
        List all documents for a user with their status
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        doc_id,
                        metadata->>'source_file' as source_file,
                        COUNT(*) as chunk_count,
                        COUNT(CASE WHEN embedding IS NULL THEN 1 END) as missing_embeddings,
                        MIN(created_at) as created_at
                    FROM chunks
                    WHERE user_id = %s AND deleted = FALSE
                    GROUP BY doc_id, metadata->>'source_file'
                    ORDER BY MIN(created_at) DESC
                """, (user_id,))
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
    
    def purge_deleted_chunks(self, user_id: str, dry_run: bool = True) -> int:
        """
        HARD DELETE: Permanently remove soft-deleted chunks from database
        
        This frees up disk space but data cannot be recovered!
        
        Args:
            user_id: User UUID
            dry_run: If True, only report what would be deleted
            
        Returns:
            Number of chunks permanently deleted
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Count soft-deleted chunks
                cur.execute("""
                    SELECT COUNT(*) as count FROM chunks
                    WHERE user_id = %s AND deleted = TRUE
                """, (user_id,))
                count = cur.fetchone()['count']
                
                if count == 0:
                    print(" No soft-deleted chunks to purge")
                    return 0
                
                print(f"\n{'Would permanently delete' if dry_run else 'Permanently deleting'} {count} soft-deleted chunks...")
                
                if not dry_run:
                    # Hard delete - data is GONE forever
                    for _ in tqdm(range(1), desc="  Purging", unit="batch"):
                        cur.execute("""
                            DELETE FROM chunks
                            WHERE user_id = %s AND deleted = TRUE
                        """, (user_id,))
                    conn.commit()
                    print(f" Permanently deleted {count} chunks (data is unrecoverable)")
                
                return count
                
        finally:
            conn.close()
    
    def cleanup_legacy_documents_table(self, user_id: str, dry_run: bool = True) -> int:
        """
        Clean up the old 'documents' table (legacy system)
        
        The new incremental system uses 'chunks' + 'tree_nodes' tables.
        This removes data from the old 'documents' table.
        
        Args:
            user_id: User UUID
            dry_run: If True, only report what would be deleted
            
        Returns:
            Number of rows deleted from documents table
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if documents table exists and has data for this user
                cur.execute("""
                    SELECT COUNT(*) as count FROM documents
                    WHERE user_id = %s
                """, (user_id,))
                count = cur.fetchone()['count']
                
                if count == 0:
                    print(" No data in legacy 'documents' table for this user")
                    return 0
                
                print(f"\n{'Would delete' if dry_run else 'Deleting'} {count} rows from legacy 'documents' table...")
                
                if not dry_run:
                    cur.execute("""
                        DELETE FROM documents WHERE user_id = %s
                    """, (user_id,))
                    conn.commit()
                    print(f" Deleted {count} rows from legacy 'documents' table")
                
                return count
                
        finally:
            conn.close()
    
    def get_storage_stats(self, user_id: str) -> Dict:
        """
        Get storage statistics for a user
        
        Returns:
            Dict with counts for active chunks, deleted chunks, tree nodes, etc.
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                stats = {}
                
                # Active chunks
                cur.execute("SELECT COUNT(*) as count FROM chunks WHERE user_id = %s AND deleted = FALSE", (user_id,))
                stats['active_chunks'] = cur.fetchone()['count']
                
                # Soft-deleted chunks (recoverable)
                cur.execute("SELECT COUNT(*) as count FROM chunks WHERE user_id = %s AND deleted = TRUE", (user_id,))
                stats['deleted_chunks'] = cur.fetchone()['count']
                
                # Tree nodes
                cur.execute("SELECT COUNT(*) as count FROM tree_nodes WHERE user_id = %s", (user_id,))
                stats['tree_nodes'] = cur.fetchone()['count']
                
                # Mappings
                cur.execute("SELECT COUNT(*) as count FROM chunk_to_leaf WHERE user_id = %s", (user_id,))
                stats['mappings'] = cur.fetchone()['count']
                
                # Legacy documents table
                try:
                    cur.execute("SELECT COUNT(*) as count FROM documents WHERE user_id = %s", (user_id,))
                    stats['legacy_documents'] = cur.fetchone()['count']
                except:
                    stats['legacy_documents'] = 0
                    conn.rollback()
                
                return stats
                
        finally:
            conn.close()


def get_user_id_from_session() -> Optional[str]:
    """Get user_id from .session file"""
    try:
        with open('.session', 'r') as f:
            content = f.read().strip()
            # Try JSON format first
            try:
                data = json.loads(content)
                return data.get('user_id')
            except json.JSONDecodeError:
                # Fall back to key=value format
                for line in content.split('\n'):
                    if line.startswith('user_id='):
                        return line.strip().split('=')[1]
    except FileNotFoundError:
        pass
    return None


def interactive_cleanup():
    """Interactive cleanup menu"""
    cleaner = DatabaseCleaner()
    user_id = get_user_id_from_session()
    
    if not user_id:
        print(" No session found. Please login first (run raptor_qa_incremental.py)")
        return
    
    print("\n" + "="*60)
    print("DATABASE CLEANUP UTILITY")
    print("="*60)
    print(f"User ID: {user_id[:8]}...")
    
    while True:
        # Show current stats
        stats = cleaner.get_storage_stats(user_id)
        print("\n" + "-"*40)
        print("CURRENT STATUS:")
        print(f"  Active chunks: {stats['active_chunks']}")
        print(f"  Soft-deleted (recoverable): {stats['deleted_chunks']}")
        print(f"  Tree nodes: {stats['tree_nodes']}")
        print(f"  Legacy documents table: {stats['legacy_documents']}")
        print("-"*40)
        print("OPTIONS:")
        print("-"*40)
        print("1. Validate database integrity")
        print("2. List all documents")
        print("3. Cleanup orphans (dry run)")
        print("4. Cleanup orphans (execute)")
        print("5. Delete specific document")
        print("6. Purge soft-deleted data (free space)")
        print("7. Clean legacy 'documents' table")
        print("8. Reset ALL data (dangerous!)")
        print("q. Quit")
        print("-"*40)
        
        choice = input("\nChoice: ").strip().lower()
        
        if choice == '1':
            cleaner.validate_integrity(user_id)
        
        elif choice == '2':
            docs = cleaner.list_documents(user_id)
            if not docs:
                print("\nNo documents found.")
            else:
                print(f"\n{'Source File':<30} {'Chunks':<10} {'Missing':<10} {'Doc ID'}")
                print("-"*80)
                for d in docs:
                    status = "️" if d['missing_embeddings'] > 0 else ""
                    print(f"{status} {d['source_file'] or 'unknown':<28} {d['chunk_count']:<10} {d['missing_embeddings']:<10} {d['doc_id'][:8]}...")
        
        elif choice == '3':
            cleaner.run_full_cleanup(user_id, dry_run=True)
        
        elif choice == '4':
            confirm = input("\n️  This will modify the database. Continue? (yes/no): ").strip().lower()
            if confirm == 'yes':
                cleaner.run_full_cleanup(user_id, dry_run=False)
            else:
                print("Cancelled.")
        
        elif choice == '5':
            docs = cleaner.list_documents(user_id)
            if not docs:
                print("\nNo documents to delete.")
                continue
            
            print("\nDocuments:")
            for i, d in enumerate(docs, 1):
                print(f"  {i}. {d['source_file'] or 'unknown'} ({d['chunk_count']} chunks)")
            
            try:
                idx = int(input("\nDocument number to delete (0 to cancel): ")) - 1
                if idx < 0:
                    print("Cancelled.")
                    continue
                
                doc = docs[idx]
                confirm = input(f"\n️  Delete '{doc['source_file']}'? (yes/no): ").strip().lower()
                
                if confirm == 'yes':
                    cleaner.cleanup_document(user_id, doc_id=doc['doc_id'], dry_run=False)
                else:
                    print("Cancelled.")
            except (ValueError, IndexError):
                print("Invalid selection.")
        
        elif choice == '6':
            print("\n" + "!"*60)
            print("️  PURGE: This will PERMANENTLY delete soft-deleted chunks!")
            print("   Data will NOT be recoverable after this.")
            print("!"*60)
            confirm = input("\nType 'PURGE' to confirm: ").strip()
            if confirm == 'PURGE':
                cleaner.purge_deleted_chunks(user_id, dry_run=False)
            else:
                print("Cancelled.")
        
        elif choice == '7':
            print("\n" + "-"*60)
            print("LEGACY CLEANUP")
            print("-"*60)
            print("The old 'documents' table is from the legacy system.")
            print("The new system uses 'chunks' + 'tree_nodes' tables.")
            print("If you're fully migrated, you can remove legacy data.")
            
            confirm = input("\nDelete your data from legacy 'documents' table? (yes/no): ").strip().lower()
            if confirm == 'yes':
                cleaner.cleanup_legacy_documents_table(user_id, dry_run=False)
            else:
                print("Cancelled.")
        
        elif choice == '8':
            print("\n" + "!"*60)
            print("️  DANGER: This will DELETE ALL your data!")
            print("!"*60)
            confirm1 = input("\nType 'DELETE ALL DATA' to confirm: ").strip()
            if confirm1 == 'DELETE ALL DATA':
                cleaner.reset_user_data(user_id, confirm=True)
            else:
                print("Cancelled.")
        
        elif choice == 'q':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Cleanup Utility")
    parser.add_argument('--validate', action='store_true', help='Run integrity validation')
    parser.add_argument('--cleanup', action='store_true', help='Run cleanup (dry run)')
    parser.add_argument('--execute', action='store_true', help='Execute cleanup (not dry run)')
    parser.add_argument('--delete-doc', type=str, help='Delete specific document by source_file')
    parser.add_argument('--purge', action='store_true', help='Permanently delete soft-deleted chunks')
    parser.add_argument('--clean-legacy', action='store_true', help='Clean old documents table')
    parser.add_argument('--stats', action='store_true', help='Show storage statistics')
    parser.add_argument('--reset', action='store_true', help='Reset all data (DANGEROUS)')
    parser.add_argument('--confirm', action='store_true', help='Confirm dangerous operations')
    parser.add_argument('--user-id', type=str, help='User ID (defaults to session)')
    
    args = parser.parse_args()
    
    # Get user_id
    user_id = args.user_id or get_user_id_from_session()
    
    if not user_id:
        print(" No user_id found. Please login first or provide --user-id")
        sys.exit(1)
    
    cleaner = DatabaseCleaner()
    
    # Handle command-line operations
    if args.stats:
        stats = cleaner.get_storage_stats(user_id)
        print("\n" + "="*50)
        print("STORAGE STATISTICS")
        print("="*50)
        print(f"Active chunks:           {stats['active_chunks']}")
        print(f"Soft-deleted chunks:     {stats['deleted_chunks']} (recoverable)")
        print(f"Tree nodes:              {stats['tree_nodes']}")
        print(f"Chunk-to-leaf mappings:  {stats['mappings']}")
        print(f"Legacy documents table:  {stats['legacy_documents']}")
    elif args.validate:
        cleaner.validate_integrity(user_id)
    elif args.cleanup:
        cleaner.run_full_cleanup(user_id, dry_run=True)
    elif args.execute:
        cleaner.run_full_cleanup(user_id, dry_run=False)
    elif args.delete_doc:
        cleaner.cleanup_document(user_id, source_file=args.delete_doc, dry_run=not args.confirm)
    elif args.purge:
        if args.confirm:
            cleaner.purge_deleted_chunks(user_id, dry_run=False)
        else:
            cleaner.purge_deleted_chunks(user_id, dry_run=True)
            print("\n️  Use --confirm to actually purge the data.")
    elif args.clean_legacy:
        if args.confirm:
            cleaner.cleanup_legacy_documents_table(user_id, dry_run=False)
        else:
            cleaner.cleanup_legacy_documents_table(user_id, dry_run=True)
            print("\n️  Use --confirm to actually delete legacy data.")
    elif args.reset:
        cleaner.reset_user_data(user_id, confirm=args.confirm)
    else:
        # Interactive mode
        interactive_cleanup()
