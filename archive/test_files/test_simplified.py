"""Test the simplified QA system"""
import sys
sys.path.insert(0, '.')

from raptor_qa_simplified import SimplifiedQASystem

user_id = "5b11bef4-7ea1-4bf9-aac1-7f22f7c73705"
qa = SimplifiedQASystem(user_id)

print("=== Testing Simplified QA System ===\n")

# Test 1: List documents
print("1. Listing documents...")
docs = qa.list_documents()
print(f"   Found {len(docs)} documents:")
for doc in docs:
    import os
    fname = os.path.basename(doc['source_file'])
    print(f"   • {fname} ({doc['chunk_count']} chunks)")

# Test 2: Diagnostics
print("\n2. Running diagnostics...")
diag = qa.get_diagnostics()
print(f"   Levels: {diag['levels']}")
print(f"   Distinct docs: {diag['distinct_docs']}")
print(f"   Missing embeddings: {diag['missing_embeddings']}")

# Test 3: Summarize all
print("\n3. Testing summarize_all...")
summaries = qa.summarize_all_documents()
print(f"\n   Generated {len(summaries)} summaries")

for source_file, summary in summaries.items():
    import os
    fname = os.path.basename(source_file)
    print(f"\n   === {fname} ===")
    print(f"   {summary[:200]}...")

print("\n All tests completed!")
