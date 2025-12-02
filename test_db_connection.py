"""
Test script to verify Supabase connection and data
"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

print("="*70)
print("TESTING SUPABASE CONNECTION")
print("="*70)

# Get credentials
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_ANON_KEY")

print(f"\nSupabase URL: {url}")
print(f"API Key: {key[:20]}...{key[-10:]}")

# Create client
try:
    supabase = create_client(url, key)
    print("\n✓ Supabase client created successfully")
except Exception as e:
    print(f"\n✗ Failed to create client: {e}")
    exit(1)

# Test query - get all documents
try:
    response = supabase.table("documents").select("*").limit(5).execute()
    print(f"\n✓ Successfully queried documents table")
    print(f"  Found {len(response.data)} documents (showing first 5)")
    
    if response.data:
        for i, doc in enumerate(response.data, 1):
            print(f"\n  Document {i}:")
            print(f"    doc_id: {doc.get('doc_id', 'N/A')[:20]}...")
            print(f"    user_id: {doc.get('user_id', 'N/A')}")
            print(f"    content length: {len(doc.get('content', ''))} chars")
            print(f"    embedding length: {len(doc.get('embedding', [])) if doc.get('embedding') else 0}")
            print(f"    metadata: {doc.get('metadata', {})}")
            print(f"    source: {doc.get('source', 'N/A')}")
    else:
        print("\n  ⚠ No documents found in database!")
        
except Exception as e:
    print(f"\n✗ Failed to query documents: {e}")
    exit(1)

# Count documents by user_id
try:
    response = supabase.table("documents").select("user_id", count="exact").execute()
    print(f"\n✓ Total documents in database: {response.count}")
except Exception as e:
    print(f"\n  Could not get count: {e}")

# Test with a specific user_id (use the one from localStorage)
print("\n" + "="*70)
print("Enter a user_id to test (or press Enter to skip):")
test_user_id = input("> ").strip()

if test_user_id:
    try:
        response = supabase.table("documents").select("*").eq("user_id", test_user_id).execute()
        print(f"\n✓ Found {len(response.data)} documents for user {test_user_id[:8]}...")
        
        if response.data:
            for doc in response.data[:3]:
                print(f"\n  Content preview: {doc.get('content', '')[:100]}...")
                print(f"  Metadata: {doc.get('metadata', {})}")
        else:
            print(f"\n  ⚠ No documents found for this user_id")
            print("  This is why the chatbot says it can't find information!")
            
    except Exception as e:
        print(f"\n✗ Query failed: {e}")

print("\n" + "="*70)
print("Database connection test complete!")
print("="*70)
