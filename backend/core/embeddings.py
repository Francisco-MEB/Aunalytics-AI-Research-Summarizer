from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_chunks(texts):
    return model.encode(texts, convert_to_numpy=True)

def embed_query(q):
    return model.encode([q], convert_to_numpy=True)[0]
