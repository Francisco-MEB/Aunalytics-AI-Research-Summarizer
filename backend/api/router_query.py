# backend/api/router_query.py

import os
import json
from fastapi import APIRouter, UploadFile, File, Form
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from backend.core.retrieval import retrieve_chunks_rpc

load_dotenv()

router = APIRouter()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-2.5-flash")


@router.post("/")
async def chat(
    message: str = Form(...),
    user_id: str = Form("default"),
    file: UploadFile = File(None),
    history: str = Form("[]")
):
    # =============================
    # 1. Optional file ingestion
    # =============================
    if file:
        raw = await file.read()
        text = raw.decode("utf-8", errors="ignore")
        vec = embedder.encode([text])[0].tolist()

        supabase.table("documents").insert({
            "doc_id": os.urandom(12).hex(),
            "chunk_id": os.urandom(12).hex(),
            "user_id": user_id,
            "content": text,          # unified field
            "embedding": vec,
            "metadata": {"filename": file.filename},
            "source": "upload"
        }).execute()

    # =============================
    # 2. Vector search
    # =============================
    query_vector = embedder.encode([message])[0].tolist()
    matches = retrieve_chunks_rpc(query_vector, user_id=user_id, k=5)

    context = "\n\n".join(
        (m.get("content") or "")
        for m in matches
    )

    # =============================
    # 3. LLM prompt
    # =============================
    prompt = f"""
Use ONLY the context provided.

CONTEXT:
{context}

QUESTION:
{message}

Write a concise academic answer.
    """

    result = llm.generate_content(prompt)
    return {"response": result.text.strip()}
