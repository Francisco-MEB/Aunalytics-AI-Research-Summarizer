"""
Debug script to check user_id consistency and document storage
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))

print("="*70)
print("CHECKING ALL DOCUMENTS IN DATABASE")
print("="*70)

# Get all documents
response = supabase.table("documents").select("user_id, doc_id, content, source, metadata").execute()

if response.data:
    print(f"\nTotal documents: {len(response.data)}")
    
    # Group by user_id
    users = {}
    for doc in response.data:
        uid = doc['user_id']
        if uid not in users:
            users[uid] = []
        users[uid].append(doc)
    
    print(f"\nUnique users: {len(users)}")
    
    for user_id, docs in users.items():
        print(f"\n{'='*70}")
        print(f"User ID: {user_id}")
        print(f"Documents: {len(docs)}")
        
        # Show sources
        sources = {}
        for doc in docs:
            source = doc.get('source') or doc.get('metadata', {}).get('source_file', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        print(f"Sources:")
        for source, count in sources.items():
            print(f"  - {source}: {count} chunks")
        
        # Show first chunk preview
        if docs:
            print(f"\nFirst chunk preview:")
            print(f"  Content: {docs[0].get('content', '')[:100]}...")
            print(f"  Metadata: {docs[0].get('metadata', {})}")
else:
    print("\n❌ No documents found in database!")

print(f"\n{'='*70}")
print("CHECKING FRONTEND USER_ID")
print("="*70)
print("Check your browser's console or localStorage for 'user_id'")
print("Or check the Network tab in DevTools when making a request")
