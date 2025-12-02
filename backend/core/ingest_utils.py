# backend/core/ingest_utils.py

import os
from sentence_transformers import SentenceTransformer
from uuid import uuid4
from pypdf import PdfReader

USER_UUID = "00000000-0000-0000-0000-000000000001"  # consistent user for your demo

def read_pdf(path):
    text = ""
    reader = PdfReader(path)
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def read_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return open(path).read()
    elif ext == ".pdf":
        return read_pdf(path)
    else:
        return ""

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def ingest_file_to_supabase(path, supabase, embedding_model):
    text = read_document(path)
    chunks = chunk_text(text)
    vectors = embedding_model.encode(chunks).tolist()

    rows = []
    for i, (chunk, emb) in enumerate(zip(chunks, vectors)):
        rows.append({
            "id": str(uuid4()),
            "chunk_id": str(uuid4()),
            "text": chunk,
            "metadata": {"source": os.path.basename(path), "chunk_index": i},
            "embedding": emb,
            "user_id": USER_UUID
        })

    supabase.table("documents").insert(rows).execute()
    return len(rows)
