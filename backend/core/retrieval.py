# backend/core/retrieval.py

from supabase import create_client
import os
import numpy as np
from dotenv import load_dotenv
import ast   # ← needed to parse string embeddings

load_dotenv()

SUPA_URL = os.getenv("SUPABASE_URL")
SUPA_KEY = os.getenv("SUPABASE_ANON_KEY")

supabase = create_client(SUPA_URL, SUPA_KEY)


def retrieve_similar_chunks(user_id, query_vector, limit=8):
    """
    Pure Python fallback vector search using the `documents` table.
    Safely handles embeddings stored as strings, lists, or vectors.
    """

    # 1. Load rows
    res = supabase.from_("documents").select("*").execute()
    rows = res.data or []

    if not rows:
        return []

    # 2. Cosine similarity helper
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

        # ---- FIX: convert string "[0.1, 0.2]" → list ----
        if isinstance(emb, str):
            try:
                emb = ast.literal_eval(emb)
            except Exception:
                # corrupted embedding
                continue

        # emb SHOULD now be list or NumPy-compatible
        try:
            sim = cosine(query_vector, emb)
        except Exception:
            # skip invalid vectors
            continue

        r["similarity"] = sim

        # prefer 'content' column but fallback to 'text'
        r["text"] = r.get("content") or r.get("text") or ""

        scored.append(r)

    # 3. Sort by similarity
    scored.sort(key=lambda x: x["similarity"], reverse=True)

    # 4. Return top N
    return scored[:limit]
