"""
Supabase Connection Test Script

Tests your Supabase database connection and verifies that everything is set up correctly.

Usage:
    python test_supabase_connection.py

Requirements:
    - .env file with DATABASE_URL configured
    - Supabase project with pgvector extension enabled
"""

import os
import sys
from dotenv import load_dotenv
import psycopg

def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def print_success(text):
    """Print a success message."""
    print(f"✅ {text}")

def print_error(text):
    """Print an error message."""
    print(f"❌ {text}")

def print_info(text):
    """Print an info message."""
    print(f"ℹ️  {text}")

def test_env_file():
    """Test if .env file exists and has DATABASE_URL."""
    print_header("Test 1: Environment Configuration")
    
    if not os.path.exists(".env"):
        print_error(".env file not found!")
        print_info("Create a .env file in the project root with your DATABASE_URL")
        return False
    
    print_success(".env file exists")
    
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        print_error("DATABASE_URL not found in .env file!")
        print_info("Add your Supabase connection string to .env")
        return False
    
    print_success("DATABASE_URL is configured")
    
    # Check if it looks like a Supabase URL
    if "supabase" in db_url.lower():
        print_success("URL appears to be a Supabase connection string")
    else:
        print_info("Warning: URL doesn't appear to be from Supabase")
    
    # Check if using connection pooler (port 5432)
    if ":5432/" in db_url:
        print_success("Using Connection Pooler (port 5432) ✓")
    elif ":5433/" in db_url:
        print_error("Using direct connection (port 5433)")
        print_info("⚠️  Change to Connection Pooler URL (port 5432) for better performance")
    
    return True

def test_database_connection():
    """Test if we can connect to the database."""
    print_header("Test 2: Database Connection")
    
    db_url = os.getenv("DATABASE_URL")
    
    try:
        conn = psycopg.connect(db_url)
        print_success("Connected to database successfully!")
        
        # Get database info
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            print_info(f"PostgreSQL version: {version.split(',')[0]}")
        
        conn.close()
        return True
        
    except psycopg.OperationalError as e:
        print_error(f"Connection failed: {e}")
        print_info("\nTroubleshooting:")
        print_info("1. Check your DATABASE_URL is correct")
        print_info("2. Verify your Supabase project is active (not paused)")
        print_info("3. Check your password is correct")
        print_info("4. Use Connection Pooler URL (port 5432)")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False

def test_pgvector_extension():
    """Test if pgvector extension is enabled."""
    print_header("Test 3: pgvector Extension")
    
    db_url = os.getenv("DATABASE_URL")
    
    try:
        conn = psycopg.connect(db_url)
        
        with conn.cursor() as cur:
            # Check if vector extension exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                );
            """)
            exists = cur.fetchone()[0]
            
            if exists:
                print_success("pgvector extension is enabled!")
                
                # Get version
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
                version = cur.fetchone()[0]
                print_info(f"pgvector version: {version}")
            else:
                print_error("pgvector extension is NOT enabled!")
                print_info("\nTo enable pgvector:")
                print_info("1. Go to Supabase SQL Editor")
                print_info("2. Run: CREATE EXTENSION IF NOT EXISTS vector;")
                conn.close()
                return False
        
        conn.close()
        return True
        
    except Exception as e:
        print_error(f"Error checking extension: {e}")
        return False

def test_documents_table():
    """Test if documents table exists and has correct schema."""
    print_header("Test 4: Documents Table")
    
    db_url = os.getenv("DATABASE_URL")
    
    try:
        conn = psycopg.connect(db_url)
        
        with conn.cursor() as cur:
            # Check if table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = 'documents'
                );
            """)
            exists = cur.fetchone()[0]
            
            if not exists:
                print_error("Documents table does NOT exist!")
                print_info("\nTo create the table:")
                print_info("1. Go to Supabase SQL Editor")
                print_info("2. Run the CREATE TABLE script from SUPABASE_SETUP.md")
                conn.close()
                return False
            
            print_success("Documents table exists!")
            
            # Check schema
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'documents'
                ORDER BY ordinal_position;
            """)
            columns = cur.fetchall()
            
            print_info("Table schema:")
            for col_name, col_type in columns:
                print(f"   - {col_name}: {col_type}")
            
            # Check for vector column
            has_vector = any(col[0] == 'embedding' for col in columns)
            if has_vector:
                print_success("Vector column (embedding) exists!")
            else:
                print_error("Vector column missing! Table schema is incorrect.")
                conn.close()
                return False
            
            # Check if table has data
            cur.execute("SELECT COUNT(*) FROM documents;")
            count = cur.fetchone()[0]
            
            if count > 0:
                print_info(f"Table contains {count} documents")
            else:
                print_info("Table is empty (upload data to populate it)")
        
        conn.close()
        return True
        
    except Exception as e:
        print_error(f"Error checking table: {e}")
        return False

def test_vector_index():
    """Test if vector index exists for fast similarity search."""
    print_header("Test 5: Vector Index")
    
    db_url = os.getenv("DATABASE_URL")
    
    try:
        conn = psycopg.connect(db_url)
        
        with conn.cursor() as cur:
            # Check for index on embedding column
            cur.execute("""
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = 'documents' 
                AND indexdef LIKE '%embedding%';
            """)
            indexes = cur.fetchall()
            
            if indexes:
                print_success(f"Vector index exists!")
                for idx_name, idx_def in indexes:
                    print_info(f"Index: {idx_name}")
            else:
                print_error("No vector index found!")
                print_info("\nTo create index:")
                print_info("1. Go to Supabase SQL Editor")
                print_info("2. Run the CREATE INDEX script from SUPABASE_SETUP.md")
                print_info("\n⚠️  Without index, queries will be SLOW!")
                conn.close()
                return False
        
        conn.close()
        return True
        
    except Exception as e:
        print_error(f"Error checking index: {e}")
        return False

def test_sample_query():
    """Test if we can perform a sample vector similarity query."""
    print_header("Test 6: Sample Vector Query")
    
    db_url = os.getenv("DATABASE_URL")
    
    try:
        conn = psycopg.connect(db_url)
        
        with conn.cursor() as cur:
            # Check if table has data
            cur.execute("SELECT COUNT(*) FROM documents;")
            count = cur.fetchone()[0]
            
            if count == 0:
                print_info("Skipping query test (no data in table)")
                print_info("Upload data first using: python upload_to_db.py")
                conn.close()
                return True  # Not a failure, just no data yet
            
            # Try a sample similarity query
            # Create a dummy embedding (384 dimensions of zeros)
            dummy_embedding = "[" + ",".join(["0"] * 384) + "]"
            
            cur.execute(f"""
                SELECT doc_id, content, embedding <=> '{dummy_embedding}'::vector AS distance
                FROM documents
                ORDER BY embedding <=> '{dummy_embedding}'::vector
                LIMIT 1;
            """)
            result = cur.fetchone()
            
            if result:
                print_success("Vector similarity query works!")
                print_info(f"Sample result: {result[0][:50]}...")
            else:
                print_error("Query returned no results")
                conn.close()
                return False
        
        conn.close()
        return True
        
    except Exception as e:
        print_error(f"Error running sample query: {e}")
        print_info("\nThis might indicate:")
        print_info("1. Vector index is missing")
        print_info("2. Table schema is incorrect")
        print_info("3. pgvector extension issue")
        return False

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  🧪 SUPABASE CONNECTION TEST SUITE")
    print("="*60)
    print("\nThis will verify your Supabase setup is correct.\n")
    
    # Track results
    tests = [
        ("Environment Configuration", test_env_file),
        ("Database Connection", test_database_connection),
        ("pgvector Extension", test_pgvector_extension),
        ("Documents Table", test_documents_table),
        ("Vector Index", test_vector_index),
        ("Sample Vector Query", test_sample_query),
    ]
    
    results = []
    
    # Run each test
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            
            # Stop if a critical test fails
            if not result and test_name in ["Environment Configuration", "Database Connection"]:
                print_error("\nCritical test failed. Fix this before continuing.")
                break
                
        except Exception as e:
            print_error(f"Test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:12} {test_name}")
    
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} tests passed")
    print(f"{'='*60}\n")
    
    if passed == total:
        print_success("All tests passed! Your Supabase setup is complete! 🎉")
        print_info("\nNext steps:")
        print_info("1. Upload your research papers")
        print_info("2. Run: python qa_system.py")
        print_info("3. Start asking questions!")
        return 0
    else:
        print_error("Some tests failed. See SUPABASE_SETUP.md for troubleshooting.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
