"""Test what dimensions the embedding model is actually producing"""

from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print(f"Model: {model}")
print(f"Max sequence length: {model.max_seq_length}")

# Test embedding
test_text = "This is a test sentence."
embedding = model.encode(test_text, convert_to_numpy=True)

print(f"\nTest text: {test_text}")
print(f"Embedding shape: {embedding.shape}")
print(f"Embedding dimensions: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")

# Expected: 384 dimensions for all-MiniLM-L6-v2
if len(embedding) == 384:
    print("\n✅ Correct! Model produces 384 dimensions")
else:
    print(f"\n❌ ERROR: Expected 384 dimensions but got {len(embedding)}")
    print("This model is NOT all-MiniLM-L6-v2 or something is wrong")
