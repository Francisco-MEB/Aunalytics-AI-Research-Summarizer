# backend/core/retrieval.py

import os
import numpy as np
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPA_URL = os.getenv("SUPABASE_URL")
SUPA_KEY = os.getenv("SUPABASE_ANON_KEY")

supabase = create_client(SUPA_URL, SUPA_KEY)


def retrieve_chunks_rpc(query_vector, user_id="default", k=8):
    """Use pgvector RPC match_chunks."""
    res = supabase.rpc("match_chunks", {
        "query_embedding": query_vector,
        "match_count": k,
        "user_id": user_id
    }).execute()

    return res.data or []


# --- Python fallback if RPC unavailable (optional) ---

import ast

def retrieve_chunks_python(query_vector, user_id="default", k=8):
    """Manual cosine similarity fallback."""
    res = supabase.from_("documents").select("*").eq("user_id", user_id).execute()
    rows = res.data or []

    def cosine(a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return -1.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    scored = []
    for r in rows:
        emb = r.get("embedding")
        if emb is None:
            continue

        if isinstance(emb, str):
            emb = ast.literal_eval(emb)

        r["similarity"] = cosine(query_vector, emb)
        r["content"] = r.get("content") or r.get("text") or ""
        scored.append(r)

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:k]
