import os
import json
from fastapi import APIRouter, UploadFile, File, Form
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()

router = APIRouter()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-2.5-flash")


def vector_search(query: str, user_id="default", k=4):
    embedding = model.encode([query])[0].tolist()

    rpc = supabase.rpc("match_chunks", {
        "query_embedding": embedding,
        "match_count": k,
        "user_id": user_id
    }).execute()

    return rpc.data


@router.post("/")
async def chat(message: str = Form(...), user_id: str = Form("default"), file: UploadFile = File(None)):
    # If file is attached → treat as ingestion + chat
    if file:
        content = (await file.read()).decode("utf-8", errors="ignore")
        file_vec = model.encode([content])[0].tolist()

        supabase.table("documents").insert({
            "chunk_id": os.urandom(12).hex(),
            "doc_id": os.urandom(12).hex(),
            "user_id": user_id,
            "text": content,
            "metadata": {"filename": file.filename},
            "embedding": file_vec
        }).execute()

    # RAG similarity search
    matches = vector_search(message, user_id=user_id, k=5)
    context = "\n\n".join([m["text"] for m in matches])

    prompt = f"""
You are a research assistant. Use ONLY the context.

CONTEXT:
{context}

QUESTION:
{message}

Answer concisely.
"""

    response = llm.generate_content(prompt)
    final = response.text.strip()

    return {"response": final}
