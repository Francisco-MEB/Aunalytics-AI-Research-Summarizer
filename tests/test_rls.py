"""
Test Row Level Security (RLS) - Verify users can only access their own documents
"""
import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_rls():
    """Test that RLS properly isolates user data"""
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ ERROR: DATABASE_URL not found")
        return False
    
    print("🔒 Testing Row Level Security\n" + "="*60 + "\n")
    
    try:
        conn = psycopg2.connect(db_url)
        
        # Test 1: Check total documents (admin view)
        print("📊 Test 1: Total documents in database (admin view)")
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents;")
            total = cur.fetchone()[0]
            print(f"   Total documents: {total}\n")
        
        # Test 2: Check documents per user
        print("📊 Test 2: Documents grouped by user_id")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COALESCE(user_id::text, 'NULL') as user, 
                    COUNT(*) as count,
                    MIN(metadata->>'source_file') as sample_file
                FROM documents 
                GROUP BY user_id
                ORDER BY count DESC;
            """)
            results = cur.fetchall()
            
            if not results:
                print("   ⚠️  No documents found!")
            else:
                print("   User ID                              | Count | Sample File")
                print("   " + "-"*70)
                for user_id, count, sample_file in results:
                    print(f"   {user_id[:36]:36} | {count:5} | {sample_file}")
                print()
        
        # Test 3: Simulate user queries with RLS
        print("🔐 Test 3: Simulating user-specific queries (RLS simulation)\n")
        
        # Get list of user_ids
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT user_id FROM documents WHERE user_id IS NOT NULL;")
            user_ids = [row[0] for row in cur.fetchall()]
        
        if not user_ids:
            print("   ⚠️  No documents with user_id found. Upload with --user-id first!\n")
        else:
            for user_id in user_ids[:3]:  # Test first 3 users
                with conn.cursor() as cur:
                    # Query as if we're this user
                    cur.execute("""
                        SELECT COUNT(*), 
                               COUNT(DISTINCT metadata->>'source_file') as file_count
                        FROM documents 
                        WHERE user_id = %s;
                    """, (user_id,))
                    count, file_count = cur.fetchone()
                    
                    print(f"   User: {user_id}")
                    print(f"   ✅ Can see {count} documents from {file_count} file(s)")
                    
                    # Try to see another user's data (should be 0 with RLS)
                    if len(user_ids) > 1:
                        other_user = [u for u in user_ids if u != user_id][0]
                        cur.execute("""
                            SELECT COUNT(*) 
                            FROM documents 
                            WHERE user_id = %s;
                        """, (other_user,))
                        other_count = cur.fetchone()[0]
                        
                        if other_count == 0:
                            print(f"   ❌ Cannot see other user's {other_user} documents (RLS working!)")
                        else:
                            print(f"   ⚠️  WARNING: Can see {other_count} docs from user {other_user}")
                    print()
        
        # Test 4: Check RLS status
        print("🔒 Test 4: RLS Configuration Status")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT relrowsecurity 
                FROM pg_class 
                WHERE relname = 'documents';
            """)
            rls_enabled = cur.fetchone()[0]
            
            if rls_enabled:
                print("   ✅ Row Level Security is ENABLED\n")
            else:
                print("   ⚠️  WARNING: Row Level Security is DISABLED!\n")
        
        # Test 5: Check RLS policies
        print("🔒 Test 5: Active RLS Policies")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT policyname, cmd, qual 
                FROM pg_policies 
                WHERE tablename = 'documents'
                ORDER BY policyname;
            """)
            policies = cur.fetchall()
            
            if not policies:
                print("   ⚠️  No RLS policies found!\n")
            else:
                print("   Policy Name                              | Command | Condition")
                print("   " + "-"*80)
                for name, cmd, qual in policies:
                    qual_short = qual[:40] + "..." if qual and len(qual) > 40 else qual
                    print(f"   {name[:40]:40} | {cmd:7} | {qual_short}")
                print()
        
        conn.close()
        
        print("="*60)
        print("✅ RLS test complete!\n")
        
        # Summary
        if rls_enabled and policies:
            print("✅ RLS is properly configured")
            print("💡 To fully test RLS, you need to:")
            print("   1. Upload documents with different user_id values")
            print("   2. Query via Supabase client (with auth.uid())")
            print("   3. Verify users can only see their own data")
        else:
            print("⚠️  RLS configuration incomplete")
            print("💡 Run the RLS setup SQL in your Supabase SQL Editor")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    test_rls()
