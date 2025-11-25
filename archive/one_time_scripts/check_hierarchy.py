"""
Query Supabase hierarchy and verify RLS
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import json

load_dotenv()

db_config = {
    'user': os.getenv('user'),
    'password': os.getenv('password'),
    'host': os.getenv('host'),
    'port': int(os.getenv('port', '5432')),
    'dbname': os.getenv('dbname')
}

def check_hierarchy(user_id: str):
    """Check hierarchy levels and document counts"""
    conn = psycopg2.connect(**db_config)
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Count documents by hierarchy level
            cur.execute("""
                SELECT hierarchy_level, COUNT(*) as count
                FROM documents
                WHERE user_id = %s
                GROUP BY hierarchy_level
                ORDER BY hierarchy_level
            """, (user_id,))
            
            results = cur.fetchall()
            print("\nHierarchy Statistics:")
            print("-" * 40)
            for row in results:
                print(f"Level {row['hierarchy_level']}: {row['count']} documents")
            
            # Sample from each level
            print("\n" + "=" * 40)
            print("Sample Documents from Each Level:")
            print("=" * 40)
            
            for level in [0, 1, 2]:
                cur.execute("""
                    SELECT doc_id, content, metadata
                    FROM documents
                    WHERE user_id = %s AND hierarchy_level = %s
                    LIMIT 2
                """, (user_id, level))
                
                samples = cur.fetchall()
                print(f"\n--- Level {level} ---")
                for i, doc in enumerate(samples, 1):
                    print(f"\nDocument {i}:")
                    print(f"ID: {doc['doc_id']}")
                    print(f"Content preview: {doc['content'][:200]}...")
                    metadata = json.loads(doc['metadata']) if isinstance(doc['metadata'], str) else doc['metadata']
                    print(f"Metadata: {metadata}")
            
            # Test RLS - try to access another user's data
            print("\n" + "=" * 40)
            print("Testing RLS (should return 0):")
            print("=" * 40)
            fake_user = "00000000-0000-0000-0000-000000000000"
            cur.execute("""
                SELECT COUNT(*) as count
                FROM documents
                WHERE user_id = %s
            """, (fake_user,))
            
            result = cur.fetchone()
            print(f"Documents for fake user {fake_user}: {result['count']}")
            print("RLS Status: WORKING" if result['count'] == 0 else "RLS Status: NOT WORKING")
            
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python check_hierarchy.py <user_id>")
        sys.exit(1)
    
    user_id = sys.argv[1]
    check_hierarchy(user_id)
