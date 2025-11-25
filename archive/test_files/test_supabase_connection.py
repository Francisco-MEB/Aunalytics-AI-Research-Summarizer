"""
Quick test script to verify Supabase connection
"""
import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_connection():
    """Test the Supabase database connection"""
    
    # Get database URL
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        print(" ERROR: DATABASE_URL not found in .env file")
        print(" Make sure your .env file has: DATABASE_URL=postgresql://...")
        return False
    
    print(f" Attempting to connect to Supabase...")
    print(f" Database: {db_url.split('@')[1].split('/')[0] if '@' in db_url else 'hidden'}")
    
    try:
        # Try to connect
        conn = psycopg2.connect(db_url)
        print(" Connection successful!")
        
        # Check if pgvector extension exists
        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            result = cur.fetchone()
            
            if result:
                print(" pgvector extension is enabled")
            else:
                print("️  WARNING: pgvector extension NOT found")
                print(" Run this in Supabase SQL Editor: CREATE EXTENSION vector;")
        
        # Check if documents table exists
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'documents'
                );
            """)
            table_exists = cur.fetchone()[0]
            
            if table_exists:
                print(" 'documents' table exists")
                
                # Check table structure
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'documents'
                    ORDER BY ordinal_position;
                """)
                columns = cur.fetchall()
                print("\n Table structure:")
                for col_name, col_type in columns:
                    print(f"   - {col_name}: {col_type}")
                
                # Check if user_id column exists
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'documents' AND column_name = 'user_id';
                """)
                has_user_id = cur.fetchone()
                
                if has_user_id:
                    print("\n user_id column exists (RLS ready)")
                else:
                    print("\n️  WARNING: user_id column NOT found")
                    print(" Run this in Supabase SQL Editor:")
                    print("   ALTER TABLE documents ADD COLUMN user_id UUID;")
                
                # Check row count
                cur.execute("SELECT COUNT(*) FROM documents;")
                count = cur.fetchone()[0]
                print(f"\n Current row count: {count} documents")
                
            else:
                print("️  WARNING: 'documents' table does NOT exist")
                print(" Run this in Supabase SQL Editor:")
                print("""
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source_file TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(384) NOT NULL,
    user_id UUID,
    created_at TIMESTAMP DEFAULT NOW()
);
                """)
        
        # Check RLS status
        with conn.cursor() as cur:
            cur.execute("""
                SELECT relrowsecurity 
                FROM pg_class 
                WHERE relname = 'documents';
            """)
            result = cur.fetchone()
            
            if result and result[0]:
                print("\n Row Level Security (RLS) is ENABLED")
            else:
                print("\n️  Row Level Security (RLS) is DISABLED")
                print(" Enable it in Supabase SQL Editor:")
                print("   ALTER TABLE documents ENABLE ROW LEVEL SECURITY;")
        
        conn.close()
        print("\n All checks complete! Connection is working.")
        return True
        
    except psycopg2.OperationalError as e:
        print(f" Connection FAILED: {e}")
        print("\n Troubleshooting steps:")
        print("1. Check your DATABASE_URL in .env file")
        print("2. Verify your Supabase project is active")
        print("3. Check your database password is correct")
        print("4. Make sure you're using the connection string from Project Settings → Database")
        return False
        
    except Exception as e:
        print(f" Error: {e}")
        return False


if __name__ == "__main__":
    print(" Testing Supabase Connection\n" + "="*50 + "\n")
    success = test_connection()
    
    if success:
        print("\n" + "="*50)
        print(" Ready to upload documents!")
        print("\n Next step:")
        print("   python embeddings/store_to_supabase --in your_file.pdf --user-id \"your-uuid\"")
    else:
        print("\n" + "="*50)
        print(" Please fix the connection issues before uploading data")
