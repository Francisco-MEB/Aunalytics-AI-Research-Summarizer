from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

SUPA_URL = os.getenv("SUPABASE_URL")
SUPA_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPA_URL, SUPA_KEY)

def supabase_insert_document(doc_id, user_id, chunks, embeddings):
    records = []
    for chunk, emb in zip(chunks, embeddings):
        records.append({
            "doc_id": doc_id,
            "user_id": user_id,
            "chunk_id": chunk["id"],
            "text": chunk["text"],
            "embedding": emb.tolist(),
            "metadata": chunk["metadata"]
        })

    supabase.table("documents").insert(records).execute()

# simple wrapper for LLM if needed
def llm(prompt):
    # You can redirect this to OpenAI/Gemini
    raise NotImplementedError("Connect to your LLM here")
