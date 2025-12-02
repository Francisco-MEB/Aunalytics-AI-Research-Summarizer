import os
import uuid
import json
from fastapi import APIRouter, UploadFile, File, Form

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx2txt

from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))


# ---------------------- Helpers -----------------------------

def read_pdf(file_bytes):
    reader = PdfReader(file_bytes)
    text = ""
    for p in reader.pages:
        t = p.extract_text()
        if t:
            text += t
    return text


def read_docx(file_bytes):
    path = "/tmp/temp.docx"
    with open(path, "wb") as f:
        f.write(file_bytes)
    return docx2txt.process(path)


def read_txt(file_bytes):
    return file_bytes.decode("utf-8", errors="ignore")


# ---------------------- Route -------------------------------

@router.post("/")
async def ingest_file(file: UploadFile = File(...), user_id: str = Form("default")):
    ext = file.filename.split(".")[-1].lower()
    raw = await file.read()

    if ext == "pdf":
        text = read_pdf(raw)
    elif ext == "docx":
        text = read_docx(raw)
    else:
        text = read_txt(raw)

    if not text.strip():
        return {"error": "Empty document"}

    emb = model.encode([text])[0].tolist()
    doc_id = str(uuid.uuid4())

    supabase.table("documents").insert({
        "doc_id": doc_id,
        "user_id": user_id,
        "chunk_id": str(uuid.uuid4()),
        "text": text,
        "metadata": {"filename": file.filename},
        "embedding": emb
    }).execute()

    return {"status": "ok", "doc_id": doc_id}
