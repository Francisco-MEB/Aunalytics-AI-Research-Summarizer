"""
Script to test Supabase connection and verify document storage
Run this to check if your documents are being stored correctly
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")

if not supabase_url or not supabase_key:
    print("ERROR: SUPABASE_URL or SUPABASE_ANON_KEY not found in .env file")
    exit(1)

print(f"Connecting to Supabase: {supabase_url}")
supabase = create_client(supabase_url, supabase_key)

# Test 1: Check if documents table exists and has data
print("\n" + "="*70)
print("TEST 1: Checking documents table")
print("="*70)

try:
    response = supabase.table("documents").select("*").limit(5).execute()
    
    if response.data:
        print(f"✅ Found {len(response.data)} documents (showing first 5)")
        for i, doc in enumerate(response.data, 1):
            print(f"\nDocument {i}:")
            print(f"  Doc ID: {doc.get('doc_id')}")
            print(f"  Chunk ID: {doc.get('chunk_id')}")
            print(f"  User ID: {doc.get('user_id')}")
            print(f"  Content preview: {doc.get('content', '')[:100]}...")
            print(f"  Has embedding: {doc.get('embedding') is not None}")
            if doc.get('embedding'):
                print(f"  Embedding dimensions: {len(doc.get('embedding'))}")
    else:
        print("❌ No documents found in the database")
        print("   Please upload a document first using the /ingest/ endpoint")
        
except Exception as e:
    print(f"❌ Error accessing documents table: {e}")
    print("   Make sure the 'documents' table exists in Supabase")

# Test 2: Check for specific user's documents
print("\n" + "="*70)
print("TEST 2: Checking documents for user 'test-user-123'")
print("="*70)

try:
    response = supabase.table("documents").select("*").eq("user_id", "test-user-123").execute()
    
    if response.data:
        print(f"✅ Found {len(response.data)} documents for test-user-123")
        
        # Group by source file
        files = {}
        for doc in response.data:
            source = doc.get('metadata', {}).get('source_file', 'unknown')
            if source not in files:
                files[source] = []
            files[source].append(doc)
        
        print(f"\nFiles uploaded:")
        for filename, chunks in files.items():
            print(f"  - {filename}: {len(chunks)} chunks")
            
    else:
        print("❌ No documents found for test-user-123")
        print("   The upload might have failed or used a different user_id")
        
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Test RPC function
print("\n" + "="*70)
print("TEST 3: Testing match_chunks RPC function")
print("="*70)

try:
    # Create a simple test embedding (all zeros)
    test_embedding = [0.0] * 384
    
    response = supabase.rpc(
        'match_chunks',
        {
            'query_embedding': test_embedding,
            'match_count': 3,
            'filter_user_id': 'test-user-123'
        }
    ).execute()
    
    if response.data:
        print(f"✅ RPC function works! Returned {len(response.data)} results")
        for i, result in enumerate(response.data, 1):
            print(f"\nResult {i}:")
            print(f"  Doc ID: {result.get('doc_id')}")
            print(f"  Content preview: {result.get('content', '')[:80]}...")
            print(f"  Similarity: {result.get('similarity')}")
    else:
        print("⚠️  RPC function returned no results")
        print("   This might be okay if there are no documents yet")
        
except Exception as e:
    print(f"❌ RPC function failed: {e}")
    print("\n   SOLUTION: Run this SQL in your Supabase SQL Editor:")
    print("""
    CREATE OR REPLACE FUNCTION match_chunks(
        query_embedding vector(384),
        match_count int,
        filter_user_id text
    )
    RETURNS TABLE (
        doc_id text,
        content text,
        metadata jsonb,
        similarity float
    )
    LANGUAGE sql STABLE AS $$
        SELECT
            doc_id,
            content,
            metadata,
            1 - (embedding <=> query_embedding) as similarity
        FROM documents
        WHERE user_id = filter_user_id
        ORDER BY embedding <=> query_embedding
        LIMIT match_count;
    $$;
    """)

# Test 4: Check vector extension
print("\n" + "="*70)
print("TEST 4: Checking pgvector extension")
print("="*70)

try:
    # Try to query with vector operations
    response = supabase.table("documents").select("embedding").limit(1).execute()
    
    if response.data and response.data[0].get('embedding'):
        embedding = response.data[0]['embedding']
        print(f"✅ Vector column exists with {len(embedding)} dimensions")
        
        if len(embedding) != 384:
            print(f"⚠️  WARNING: Expected 384 dimensions but got {len(embedding)}")
            print("   The embedding model uses 384 dimensions (all-MiniLM-L6-v2)")
    else:
        print("⚠️  Could not verify vector column")
        
except Exception as e:
    print(f"❌ Error checking vector extension: {e}")
    print("   Make sure pgvector extension is installed:")
    print("   CREATE EXTENSION IF NOT EXISTS vector;")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("""
Next steps:
1. If RPC function failed: Copy the SQL from above and run it in Supabase SQL Editor
2. If no documents found: Upload a test document using the /ingest/ endpoint
3. If documents exist but RPC failed: Update the RPC function in Supabase
4. After fixing: Run this script again to verify everything works

To upload a test document:
curl -X POST "http://127.0.0.1:8000/ingest/" \\
  -F "file=@test_document.txt" \\
  -F "user_id=test-user-123" \\
  -F "chunk_size=500" \\
  -F "chunk_overlap=50"

To query:
curl -X POST "http://127.0.0.1:8000/query/" \\
  -F "message=What is this document about?" \\
  -F "user_id=test-user-123" \\
  -F "top_k=4"
""")
