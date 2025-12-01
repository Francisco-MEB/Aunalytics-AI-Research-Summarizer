"""
FastAPI backend for Aunalytics RAG Research Summarizer
Integrates with qa_system.py and store_to_supabase for document processing
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import sys
import uuid
import tempfile
import shutil
import importlib.util
from pathlib import Path
from dotenv import load_dotenv

# Import our QA system and scraper
from qa_system import QASystem
from scraper import ProfessorScraper

# Import ingestion utilities - add embeddings to path first
sys.path.insert(0, str(Path(__file__).parent / "embeddings"))
try:
    # Import from the store_to_supabase module (no .py extension)
    spec = importlib.util.spec_from_file_location(
        "store_module",
        Path(__file__).parent / "embeddings" / "store_to_supabase"
    )
    store_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(store_module)
    
    read_document = store_module.read_document
    chunk_text = store_module.chunk_text
    embed_chunks = store_module.embed_chunks
    store_chunks_to_db = store_module.store_to_supabase
except Exception as e:
    print(f"Warning: Could not import store_to_supabase module: {e}")
    read_document = None
    chunk_text = None
    embed_chunks = None
    store_chunks_to_db = None

load_dotenv()

app = FastAPI(
    title="Aunalytics RAG Research Summarizer API",
    description="Backend API for document ingestion and question answering",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory
UPLOAD_DIR = Path("upload")
UPLOAD_DIR.mkdir(exist_ok=True)


# ============= REQUEST/RESPONSE MODELS =============

class QuestionRequest(BaseModel):
    question: str
    user_id: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    num_sources: int
    sources: list = []

class UploadResponse(BaseModel):
    message: str
    file_id: str
    chunks_created: int
    user_id: str

class ScrapeRequest(BaseModel):
    professor_url: str
    user_id: str
    max_papers: int = 10

class PaperInfo(BaseModel):
    title: str
    authors: str
    year: str
    abstract: str
    url: Optional[str] = None
    source: str

class ScrapeResponse(BaseModel):
    message: str
    professor_url: str
    scholar_url: Optional[str] = None
    source: str
    papers_found: int
    papers: List[PaperInfo]
    bio_info: Optional[dict] = None
    chunks_created: int
    error: Optional[str] = None



# ============= ENDPOINTS =============

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Aunalytics RAG Research Summarizer",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check with database connectivity"""
    try:
        # Test database connection
        from db_connection import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        
        return {
            "status": "healthy",
            "database": "connected",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm": "gemini-2.5-flash"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Upload and process a document (PDF, DOCX, or TXT)
    Creates embeddings and stores in database with RLS
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not supported. Use PDF, DOCX, or TXT"
        )
    
    # Validate user_id format (basic UUID check)
    if len(user_id) != 36 or user_id.count('-') != 4:
        raise HTTPException(
            status_code=400,
            detail="Invalid user_id format. Must be a valid UUID"
        )
    
    # Save uploaded file temporarily
    temp_path = UPLOAD_DIR / file.filename
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document and store in database
        print(f"[PROCESSING] Ingesting {file.filename} for user {user_id[:8]}...")
        
        # Read document
        text = read_document(str(temp_path))
        if not text.strip():
            raise HTTPException(status_code=400, detail="Document is empty or could not be read")
        
        # Chunk text
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from document")
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk["id"] = str(uuid.uuid4())
            chunk["metadata"]["source"] = file.filename
            chunk["metadata"]["chunk_index"] = i
        
        # Generate embeddings
        vectors = embed_chunks(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Store in database
        store_chunks_to_db(chunks, vectors, file.filename, user_id=user_id)
        
        return UploadResponse(
            message="Document processed successfully",
            file_id=file.filename,
            chunks_created=len(chunks),
            user_id=user_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()


@app.post("/api/scrape", response_model=ScrapeResponse)
async def scrape_professor_website(request: ScrapeRequest):
    """
    Scrape a professor's website for research papers
    Finds Google Scholar, Academia.edu, or ResearchGate profiles
    Extracts paper titles and abstracts, then embeds and stores them
    """
    try:
        print(f"[API] Scraping request for: {request.professor_url}")
        
        # Initialize scraper
        scraper = ProfessorScraper()
        
        # Scrape the website
        scrape_result = scraper.scrape_professor_website(
            request.professor_url,
            max_papers=request.max_papers
        )
        
        if scrape_result["error"]:
            raise HTTPException(status_code=500, detail=scrape_result["error"])
        
        if not scrape_result["papers"]:
            raise HTTPException(
                status_code=404,
                detail="No papers found. Could not locate Google Scholar, Academia.edu, or ResearchGate profile."
            )
        
        # Process and embed the abstracts
        print(f"[API] Processing {len(scrape_result['papers'])} papers for user {request.user_id[:8]}...")
        
        total_chunks = 0
        
        for paper in scrape_result["papers"]:
            # Combine title and abstract for better context
            paper_text = f"Title: {paper['title']}\n\nAuthors: {paper['authors']}\n\nAbstract: {paper['abstract']}"
            
            # Chunk the paper (abstracts might be long)
            chunks = chunk_text(paper_text, chunk_size=500, chunk_overlap=50)
            
            if not chunks:
                continue
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                chunk["id"] = str(uuid.uuid4())
                chunk["metadata"]["source"] = f"{paper['title']} ({paper['year']})"
                chunk["metadata"]["paper_url"] = paper.get('url', '')
                chunk["metadata"]["authors"] = paper['authors']
                chunk["metadata"]["year"] = paper['year']
                chunk["metadata"]["chunk_index"] = i
            
            # Generate embeddings
            vectors = embed_chunks(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Store in database
            store_chunks_to_db(
                chunks,
                vectors,
                source_file=f"{paper['title']} ({paper['year']})",
                user_id=request.user_id
            )
            
            total_chunks += len(chunks)
            print(f"[API] Processed paper: {paper['title'][:50]}... ({len(chunks)} chunks)")
        
        # Convert papers to Pydantic models for response
        paper_models = [
            PaperInfo(
                title=p["title"],
                authors=p["authors"],
                year=p["year"],
                abstract=p["abstract"][:500] + "..." if len(p["abstract"]) > 500 else p["abstract"],
                url=p.get("url"),
                source=p["source"]
            )
            for p in scrape_result["papers"]
        ]
        
        return ScrapeResponse(
            message=f"Successfully scraped and embedded {len(scrape_result['papers'])} papers",
            professor_url=request.professor_url,
            scholar_url=scrape_result.get("scholar_url"),
            source=scrape_result["source"],
            papers_found=len(scrape_result["papers"]),
            papers=paper_models,
            bio_info=scrape_result.get("bio_info"),
            chunks_created=total_chunks
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error scraping professor website: {str(e)}"
        )


@app.post("/api/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the uploaded documents
    Uses RAG to retrieve context and generate answer
    """
    try:
        # Initialize QA system with user's ID for RLS
        qa_system = QASystem(user_id=request.user_id)
        
        print(f"[QUERY] Processing question for user {request.user_id[:8]}...")
        
        # Get answer from QA system
        result = qa_system.ask_question(request.question)
        
        # Format sources for response
        sources = []
        if result['context'] and result['num_sources'] <= 10:
            for i, doc in enumerate(result['context'], 1):
                sources.append({
                    "index": i,
                    "relevance": doc.metadata.get('score', 0),
                    "preview": doc.page_content[:150]
                })
        
        return QuestionResponse(
            question=result['question'],
            answer=result['answer'],
            num_sources=result['num_sources'],
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.delete("/api/documents/{user_id}")
async def delete_user_documents(user_id: str):
    """
    Delete all documents for a specific user
    Useful for testing or user data cleanup
    """
    try:
        from db_connection import get_db_connection
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE user_id = %s RETURNING doc_id",
                    (user_id,)
                )
                deleted = cur.fetchall()
                conn.commit()
        
        return {
            "message": f"Deleted {len(deleted)} documents for user {user_id[:8]}...",
            "deleted_count": len(deleted)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting documents: {str(e)}"
        )


@app.get("/api/documents/{user_id}/count")
async def get_document_count(user_id: str):
    """
    Get the number of document chunks stored for a user
    """
    try:
        from db_connection import get_db_connection
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM documents WHERE user_id = %s",
                    (user_id,)
                )
                count = cur.fetchone()[0]
        
        return {
            "user_id": user_id,
            "document_count": count
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error counting documents: {str(e)}"
        )


# ============= STARTUP/SHUTDOWN =============

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    print("\n" + "="*70)
    print(" "*15 + "AUNALYTICS RAG API STARTING")
    print("="*70)
    print("[SYSTEM] Loading sentence transformer model...")
    print("[SYSTEM] Connecting to database...")
    print("[SYSTEM] API ready to accept requests")
    print("="*70 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n[SYSTEM] Shutting down gracefully...")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
