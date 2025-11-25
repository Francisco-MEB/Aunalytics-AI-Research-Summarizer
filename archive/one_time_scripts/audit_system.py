"""
Complete system audit for incremental RAPTOR
Checks database schema, data integrity, and core functions
"""
import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import json

load_dotenv()

USER_ID = "5b11bef4-7ea1-4bf9-aac1-7f22f7c73705"

def get_connection():
    return psycopg2.connect(
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=int(os.getenv("port", "5432")),
        dbname=os.getenv("dbname")
    )


def audit_schema():
    """Check if all required tables and columns exist"""
    print("\n" + "="*60)
    print("1. DATABASE SCHEMA AUDIT")
    print("="*60)
    
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check chunks table
            cur.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'chunks'
                ORDER BY ordinal_position
            """)
            chunks_cols = cur.fetchall()
            
            print("\n CHUNKS TABLE:")
            if chunks_cols:
                for col in chunks_cols:
                    print(f"  - {col['column_name']}: {col['data_type']} (nullable: {col['is_nullable']})")
            else:
                print("   Table not found!")
            
            # Check tree_nodes table
            cur.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'tree_nodes'
                ORDER BY ordinal_position
            """)
            tree_cols = cur.fetchall()
            
            print("\n TREE_NODES TABLE:")
            if tree_cols:
                for col in tree_cols:
                    print(f"  - {col['column_name']}: {col['data_type']} (nullable: {col['is_nullable']})")
            else:
                print("   Table not found!")
            
            # Check chunk_to_leaf table
            cur.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'chunk_to_leaf'
                ORDER BY ordinal_position
            """)
            mapping_cols = cur.fetchall()
            
            print("\n CHUNK_TO_LEAF TABLE:")
            if mapping_cols:
                for col in mapping_cols:
                    print(f"  - {col['column_name']}: {col['data_type']} (nullable: {col['is_nullable']})")
            else:
                print("   Table not found!")
            
            # Check indexes
            print("\n INDEXES:")
            cur.execute("""
                SELECT tablename, indexname
                FROM pg_indexes
                WHERE tablename IN ('chunks', 'tree_nodes', 'chunk_to_leaf')
                ORDER BY tablename, indexname
            """)
            indexes = cur.fetchall()
            for idx in indexes:
                print(f"  - {idx['tablename']}.{idx['indexname']}")
            
    finally:
        conn.close()


def audit_data():
    """Check actual data in tables"""
    print("\n" + "="*60)
    print("2. DATA INTEGRITY AUDIT")
    print("="*60)
    
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check chunks data
            cur.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(DISTINCT doc_id) as unique_doc_ids,
                    COUNT(DISTINCT metadata->>'source_file') as unique_source_files,
                    COUNT(*) FILTER (WHERE deleted = TRUE) as deleted_chunks,
                    COUNT(*) FILTER (WHERE embedding IS NULL) as missing_embeddings
                FROM chunks
                WHERE user_id = %s
            """, (USER_ID,))
            chunk_stats = cur.fetchone()
            
            print("\n CHUNKS TABLE DATA:")
            print(f"  - Total chunks: {chunk_stats['total_chunks']}")
            print(f"  - Unique doc_ids: {chunk_stats['unique_doc_ids']}")
            print(f"  - Unique source_files: {chunk_stats['unique_source_files']}")
            print(f"  - Deleted chunks: {chunk_stats['deleted_chunks']}")
            print(f"  - Missing embeddings: {chunk_stats['missing_embeddings']}")
            
            # Sample documents
            cur.execute("""
                SELECT 
                    doc_id,
                    metadata->>'source_file' as source_file,
                    COUNT(*) as chunk_count,
                    MIN(created_at) as uploaded_at
                FROM chunks
                WHERE user_id = %s AND deleted = FALSE
                GROUP BY doc_id, metadata->>'source_file'
                ORDER BY chunk_count DESC
                LIMIT 5
            """, (USER_ID,))
            docs = cur.fetchall()
            
            print("\n SAMPLE DOCUMENTS:")
            for doc in docs:
                doc_id_short = str(doc['doc_id'])[:8]
                source = doc['source_file'] or 'Unknown'
                print(f"  - {source}")
                print(f"    Doc ID: {doc_id_short}... | {doc['chunk_count']} chunks | {doc['uploaded_at']}")
            
            # Check tree_nodes data
            cur.execute("""
                SELECT 
                    COUNT(*) as total_nodes,
                    COUNT(DISTINCT level) as levels,
                    COUNT(*) FILTER (WHERE level = 0) as leaf_nodes,
                    COUNT(*) FILTER (WHERE summary_text IS NULL) as no_summary,
                    COUNT(*) FILTER (WHERE summary_embedding IS NULL) as no_embedding
                FROM tree_nodes
                WHERE user_id = %s
            """, (USER_ID,))
            tree_stats = cur.fetchone()
            
            print("\n TREE_NODES TABLE DATA:")
            print(f"  - Total nodes: {tree_stats['total_nodes']}")
            print(f"  - Tree levels: {tree_stats['levels']}")
            print(f"  - Leaf nodes (level 0): {tree_stats['leaf_nodes']}")
            print(f"  - Nodes without summary: {tree_stats['no_summary']}")
            print(f"  - Nodes without embedding: {tree_stats['no_embedding']}")
            
            # Nodes per level
            cur.execute("""
                SELECT level, COUNT(*) as count
                FROM tree_nodes
                WHERE user_id = %s
                GROUP BY level
                ORDER BY level
            """, (USER_ID,))
            levels = cur.fetchall()
            
            print("\n NODES PER LEVEL:")
            for lvl in levels:
                print(f"  - Level {lvl['level']}: {lvl['count']} nodes")
            
            # Check chunk_to_leaf mappings
            cur.execute("""
                SELECT COUNT(*) as total_mappings
                FROM chunk_to_leaf
                WHERE user_id = %s
            """, (USER_ID,))
            mapping_stats = cur.fetchone()
            
            print("\n CHUNK_TO_LEAF MAPPINGS:")
            print(f"  - Total mappings: {mapping_stats['total_mappings']}")
            
    finally:
        conn.close()


def audit_functions():
    """Check if helper functions exist and work"""
    print("\n" + "="*60)
    print("3. FUNCTION AUDIT")
    print("="*60)
    
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check match_chunks function
            print("\n Testing match_chunks():")
            try:
                cur.execute("""
                    SELECT * FROM match_chunks(
                        ARRAY[0.1, 0.2, 0.3]::vector(3),
                        %s::uuid,
                        3,
                        FALSE
                    )
                """, (USER_ID,))
                print("   Function exists (note: used dummy 3-dim vector, may return no results)")
            except Exception as e:
                print(f"   Error: {e}")
            
            # Check match_tree_nodes function
            print("\n Testing match_tree_nodes():")
            try:
                cur.execute("""
                    SELECT * FROM match_tree_nodes(
                        ARRAY[0.1, 0.2, 0.3]::vector(3),
                        %s::uuid,
                        1,
                        3
                    )
                """, (USER_ID,))
                print("   Function exists (note: used dummy 3-dim vector, may return no results)")
            except Exception as e:
                print(f"   Error: {e}")
            
            # Check get_tree_stats function
            print("\n Testing get_tree_stats():")
            try:
                cur.execute("SELECT * FROM get_tree_stats(%s)", (USER_ID,))
                stats = cur.fetchone()
                if stats:
                    print(f"   Function works! Stats: {dict(stats)}")
                else:
                    print("  ️  Function returned NULL")
            except Exception as e:
                print(f"   Error: {e}")
            
    finally:
        conn.close()


def audit_doc_id_grouping():
    """Check if doc_id properly groups chunks by document"""
    print("\n" + "="*60)
    print("4. DOC_ID GROUPING AUDIT")
    print("="*60)
    
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check if same source_file has same doc_id
            cur.execute("""
                SELECT 
                    metadata->>'source_file' as source_file,
                    COUNT(DISTINCT doc_id) as distinct_doc_ids,
                    COUNT(*) as chunk_count
                FROM chunks
                WHERE user_id = %s AND deleted = FALSE
                  AND metadata->>'source_file' IS NOT NULL
                GROUP BY metadata->>'source_file'
                HAVING COUNT(DISTINCT doc_id) > 1
                ORDER BY chunk_count DESC
            """, (USER_ID,))
            
            problems = cur.fetchall()
            
            if problems:
                print("\n FOUND PROBLEMS:")
                print("These source files have multiple doc_ids (should be 1 per file):")
                for p in problems:
                    print(f"  - {p['source_file']}: {p['distinct_doc_ids']} different doc_ids ({p['chunk_count']} chunks)")
            else:
                print("\n ALL GOOD: Each source_file has exactly one doc_id")
            
    finally:
        conn.close()


def main():
    """Run complete audit"""
    print("\n" + "="*60)
    print("INCREMENTAL RAPTOR SYSTEM AUDIT")
    print("="*60)
    print(f"User ID: {USER_ID}")
    
    try:
        audit_schema()
        audit_data()
        audit_functions()
        audit_doc_id_grouping()
        
        print("\n" + "="*60)
        print("AUDIT COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\n AUDIT FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
