import os
import uuid
import json
import io
from typing import List, Dict
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx2txt
from langchain_text_splitters import RecursiveCharacterTextSplitter

from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(tags=["Ingest"])
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))


# ==================== DOCUMENT READING ====================

def read_pdf(file_bytes: bytes) -> str:
    """Read text content from a PDF file using pypdf."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    # Remove NUL characters that can cause database errors
    return text.replace('\x00', '')


def read_docx(file_bytes: bytes) -> str:
    """Extract text from a .docx file."""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    text = docx2txt.process(tmp_path) or ""
    os.unlink(tmp_path)
    return text.replace('\x00', '')


def read_txt(file_bytes: bytes) -> str:
    """Read text file."""
    text = file_bytes.decode("utf-8", errors="ignore")
    return text.replace('\x00', '')


def read_document(file_bytes: bytes, filename: str) -> str:
    """Read TXT, PDF, or DOCX file into a single text string."""
    ext = filename.split(".")[-1].lower()
    
    if ext == "txt":
        return read_txt(file_bytes)
    elif ext == "pdf":
        return read_pdf(file_bytes)
    elif ext == "docx":
        return read_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: .txt, .pdf, .docx")


# ==================== TEXT CHUNKING ====================

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Dict]:
    """Split text into overlapping chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.create_documents([text], metadatas=[{}])
    return [{"text": d.page_content, "metadata": d.metadata} for d in docs]


# ==================== EMBEDDING GENERATION ====================

def embed_chunks(chunks: List[Dict], batch_size: int = 64) -> List[List[float]]:
    """Generate embeddings for all chunks using SentenceTransformer."""
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


# ==================== SUPABASE STORAGE ====================

def store_to_supabase(chunks: List[Dict], vectors: List[List[float]], source_file: str, user_id: str):
    """Store chunks and embeddings directly in Supabase."""
    
    data_to_insert = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        doc_id = str(uuid.uuid4())
        metadata = {
            "source_file": source_file,
            "chunk_index": i
        }
        data_to_insert.append({
            "doc_id": doc_id,
            "content": chunk["text"],
            "embedding": vector,
            "metadata": metadata,
            "user_id": user_id
        })
    
    # Batch insert
    supabase.table("documents").insert(data_to_insert).execute()
    
    return len(data_to_insert)


# ==================== ROUTE ====================

@router.post("/")
async def ingest_file(
    file: UploadFile = File(...), 
    user_id: str = Form(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(100)
):
    """
    Ingest a file (PDF/DOCX/TXT) -> chunk -> embed -> store in Supabase
    
    Parameters:
    - file: The document file to upload
    - user_id: User UUID for Row Level Security (required)
    - chunk_size: Characters per chunk (default 1000)
    - chunk_overlap: Overlap between chunks (default 100)
    """
    
    try:
        # Read file
        file_bytes = await file.read()
        
        # Extract text
        text = read_document(file_bytes, file.filename)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Document is empty or contains only whitespace")
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks were created from the input text")
        
        # Generate embeddings
        vectors = embed_chunks(chunks, batch_size=64)
        
        # Store in Supabase
        num_chunks = store_to_supabase(chunks, vectors, file.filename, user_id=user_id)
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Successfully processed {num_chunks} chunks from '{file.filename}'",
            "num_chunks": num_chunks,
            "filename": file.filename,
            "user_id": user_id
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")