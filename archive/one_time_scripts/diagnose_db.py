"""Quick database diagnostic script"""
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

user_id = "5b11bef4-7ea1-4bf9-aac1-7f22f7c73705"

db_config = {
    'user': os.getenv('user'),
    'password': os.getenv('password'),
    'host': os.getenv('host'),
    'port': int(os.getenv('port', '5432')),
    'dbname': os.getenv('dbname')
}

conn = psycopg2.connect(**db_config)
try:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Count by level
        cur.execute(
            "SELECT hierarchy_level, COUNT(*) as count FROM documents WHERE user_id = %s GROUP BY hierarchy_level ORDER BY hierarchy_level",
            (user_id,)
        )
        print("=== Counts by Level ===")
        for row in cur.fetchall():
            print(f"  Level {row['hierarchy_level']}: {row['count']} chunks")
        
        # Distinct documents
        cur.execute(
            "SELECT COUNT(DISTINCT metadata->>'source_file') as distinct_docs FROM documents WHERE user_id = %s AND hierarchy_level = 0",
            (user_id,)
        )
        distinct = cur.fetchone()['distinct_docs']
        print(f"\n=== Distinct Documents: {distinct} ===")
        
        # List each document
        cur.execute(
            """SELECT metadata->>'source_file' as source_file,
                      metadata->>'file_hash' as file_hash,
                      COUNT(*) as chunk_count
               FROM documents
               WHERE user_id = %s AND hierarchy_level = 0
               GROUP BY metadata->>'source_file', metadata->>'file_hash'
               ORDER BY metadata->>'source_file'""",
            (user_id,)
        )
        print("\n=== Documents (Level 0) ===")
        for row in cur.fetchall():
            fname = os.path.basename(row['source_file']) if row['source_file'] else 'Unknown'
            fhash = row['file_hash'][:8] if row['file_hash'] else 'no-hash'
            print(f"  • {fname} ({row['chunk_count']} chunks) [hash: {fhash}...]")
        
        # Missing embeddings (fixed query for vector type)
        cur.execute(
            "SELECT COUNT(*) as missing FROM documents WHERE user_id = %s AND embedding IS NULL",
            (user_id,)
        )
        missing = cur.fetchone()['missing']
        print(f"\n=== Missing Embeddings: {missing} ===")
        
        # Test retrieval for each document
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_vector = model.encode("summarize", normalize_embeddings=True).tolist()
        
        cur.execute(
            "SELECT DISTINCT metadata->>'source_file' as source_file FROM documents WHERE user_id = %s AND hierarchy_level = 0 AND metadata->>'source_file' IS NOT NULL",
            (user_id,)
        )
        sources = [r['source_file'] for r in cur.fetchall()]
        
        print(f"\n=== Testing Retrieval Per Document (top 3 chunks) ===")
        for source in sources:
            cur.execute(
                """SELECT doc_id, content, 1 - (embedding <=> %s::vector) AS similarity
                   FROM documents
                   WHERE user_id = %s AND hierarchy_level = 0 AND metadata->>'source_file' = %s
                   ORDER BY embedding <=> %s::vector
                   LIMIT 3""",
                (query_vector, user_id, source, query_vector)
            )
            rows = cur.fetchall()
            fname = os.path.basename(source)
            print(f"\n  {fname}: Retrieved {len(rows)} chunks")
            for i, r in enumerate(rows, 1):
                sim = r['similarity']
                preview = r['content'][:80].replace('\n', ' ')
                print(f"    {i}. Sim={sim:.3f}: {preview}...")

finally:
    conn.close()
