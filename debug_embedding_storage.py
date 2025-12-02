"""Debug script to test the exact embedding being created and stored"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client
import uuid

load_dotenv()

# Initialize
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))

# Test text
test_text = "This is a simple test to check embedding dimensions."

# Create embedding
print("Creating embedding...")
embedding = model.encode(test_text, convert_to_numpy=True)
embedding_list = embedding.tolist()

print(f"Original embedding dimensions: {len(embedding)}")
print(f"List embedding dimensions: {len(embedding_list)}")
print(f"First 10 values: {embedding_list[:10]}")

# Try to insert into Supabase
print("\nAttempting to insert into Supabase...")
test_record = {
    "doc_id": str(uuid.uuid4()),
    "chunk_id": "debug_test_chunk",
    "user_id": "debug-test",
    "content": test_text,
    "embedding": embedding_list,
    "metadata": {"test": "debug"}
}

try:
    response = supabase.table("documents").insert(test_record).execute()
    print("✅ Insert successful!")
    print(f"Response: {response}")
    
    # Now retrieve it
    print("\nRetrieving the record...")
    check = supabase.table("documents").select("*").eq("user_id", "debug-test").execute()
    if check.data:
        retrieved = check.data[0]
        retrieved_embedding = retrieved.get("embedding")
        print(f"Retrieved embedding dimensions: {len(retrieved_embedding)}")
        print(f"First 10 values: {retrieved_embedding[:10]}")
        
        if len(retrieved_embedding) == 384:
            print("✅ Embedding stored correctly!")
        else:
            print(f"❌ ERROR: Stored {len(retrieved_embedding)} dimensions instead of 384")
            
        # Clean up
        supabase.table("documents").delete().eq("user_id", "debug-test").execute()
        print("\nDebug record deleted")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
