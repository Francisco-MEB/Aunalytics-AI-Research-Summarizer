"""
Check the documents table schema and embedding format
"""
import os
from dotenv import load_dotenv
from supabase import create_client
import json

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))

print("="*70)
print("CHECKING EMBEDDING FORMAT")
print("="*70)

# Get one document
response = supabase.table("documents").select("*").limit(1).execute()

if response.data:
    doc = response.data[0]
    embedding = doc.get('embedding')
    
    print(f"\nEmbedding type: {type(embedding)}")
    print(f"Embedding length/size: {len(embedding) if embedding else 0}")
    
    if isinstance(embedding, str):
        print("\n⚠ WARNING: Embedding is stored as STRING, not vector!")
        print("  This means vector similarity search won't work!")
        print("  First 100 chars:", embedding[:100])
        
        # Try to parse it
        try:
            parsed = json.loads(embedding)
            print(f"\n  Can be parsed as JSON array with {len(parsed)} elements")
            if isinstance(parsed, list) and len(parsed) > 0:
                print(f"  First few values: {parsed[:5]}")
        except:
            print("  Cannot parse as JSON")
            
    elif isinstance(embedding, list):
        print(f"\n✓ Embedding is a list with {len(embedding)} dimensions")
        print(f"  First few values: {embedding[:5]}")
        
        if len(embedding) == 384:
            print("  ✓ Correct dimension (384) for all-MiniLM-L6-v2 model")
        else:
            print(f"  ⚠ Unexpected dimension count!")
    else:
        print(f"\n⚠ Unknown embedding format: {type(embedding)}")
    
    print(f"\nDocument structure:")
    for key, value in doc.items():
        if key == 'embedding':
            continue
        if key == 'content':
            print(f"  {key}: {len(value)} chars")
        else:
            print(f"  {key}: {value}")

print("\n" + "="*70)
