"""
Enhanced ingestion script: reads PDF/DOCX/TXT -> chunks -> embeddings -> Supabase
Stores directly in database instead of JSONL files
"""
import argparse
import os
import sys
import uuid
from typing import List, Dict
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import docx2txt
from pypdf import PdfReader
import psycopg2
from psycopg2.extras import execute_values

# Add parent directory to path for db_connection import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_connection import get_db_connection

# Load environment variables from .env file
load_dotenv()


# ==================== DOCUMENT READING ====================

def read_pdf(path: str) -> str:
    """Read text content from a PDF file using pypdf."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def read_docx(path: str) -> str:
    """Extract text from a .docx file and return one big string."""
    text = docx2txt.process(path) or ""
    return " ".join(text.split())


def read_document(path: str) -> str:
    """Read TXT, PDF, or DOCX file into a single text string."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            # Remove NUL characters that can cause database errors
            text = text.replace('\x00', '')
            return text
    elif ext == ".pdf":
        print(f"Detected PDF format. Extracting text from '{path}'...")
        text = read_pdf(path)
        text = text.replace('\x00', '')
        return text
    elif ext == ".docx":
        print(f"Detected DOCX format. Extracting text from '{path}'...")
        text = read_docx(path)
        text = text.replace('\x00', '')
        return text
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: .txt, .pdf, .docx")


# ==================== TEXT CHUNKING ====================

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """Split text into overlapping chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.create_documents([text], metadatas=[{}])
    return [{"text": d.page_content, "metadata": d.metadata} for d in docs]


# ==================== EMBEDDING GENERATION ====================

def embed_chunks(chunks: List[Dict], model_name: str, batch_size: int = 64) -> List[List[float]]:
    """Generate embeddings for all chunks using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    return [v.tolist() for v in vectors]


# ==================== SUPABASE DATABASE OPERATIONS ====================

def store_to_supabase(chunks: List[Dict], vectors: List[List[float]], source_file: str, user_id: str = None):
    """Store chunks and embeddings directly in Supabase.
    
    Args:
        chunks: List of text chunks with metadata
        vectors: List of embedding vectors
        source_file: Path to source document
        user_id: Optional UUID for RLS. If None, data is accessible to all authenticated users.
    """
    
    conn = get_db_connection()
    
    try:
        # Prepare data for batch insert
        import json
        data_to_insert = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            doc_id = chunk.get("id", str(uuid.uuid4()))
            # Store source_file and chunk_index in metadata JSON
            metadata = {
                "source_file": source_file,
                "chunk_index": i
            }
            data_to_insert.append((
                doc_id,
                chunk["text"],
                vector,
                json.dumps(metadata),  # Convert metadata to JSON string
                user_id  # Add user_id for RLS
            ))
        
        # Batch insert using execute_values (much faster)
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO documents (doc_id, content, embedding, metadata, user_id)
                VALUES %s
                ON CONFLICT (doc_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    user_id = EXCLUDED.user_id;
                """,
                data_to_insert
            )
            conn.commit()
        
        print(f"SUCCESS: Stored {len(data_to_insert)} chunks in Supabase")
        
    except Exception as e:
        print(f"ERROR: Database error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


# ==================== MAIN FUNCTION ====================

def main() -> int:
    p = argparse.ArgumentParser(
        description="Ingest documents (PDF/DOCX/TXT) -> chunks -> embeddings -> Supabase"
    )
    
    # Required arguments
    p.add_argument("--in", dest="inp", required=True, 
                   help="Input file path (.txt, .pdf, or .docx)")
    
    # Optional arguments, for the command line
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Sentence transformer model name")
    p.add_argument("--chunk", dest="chunk", type=int, default=1000,
                   help="Approximate characters per chunk")
    p.add_argument("--overlap", dest="overlap", type=int, default=100,
                   help="Overlap between chunks (characters)")
    p.add_argument("--batch", dest="batch", type=int, default=64,
                   help="Embedding batch size")
    p.add_argument("--user-id", dest="user_id", type=str, required=True,
                   help="User UUID for Row Level Security (required)")
    p.add_argument("--clean", dest="clean", action="store_true",
                   help="Delete all existing documents for this user before uploading")
    
    args = p.parse_args()
    
    # Validate input file exists
    if not os.path.isfile(args.inp):
        print(f"Error: Input file {args.inp} does not exist.", file=sys.stderr)
        return 2
    
    # Read document
    print(f"\nReading document: {args.inp}")
    try:
        text = read_document(args.inp)
    except Exception as e:
        print(f"ERROR: Error reading document: {e}", file=sys.stderr)
        return 3
    
    if not text.strip():
        print(f"ERROR: Document is empty or contains only whitespace.", file=sys.stderr)
        return 4
    
    print(f"SUCCESS: Read {len(text)} characters from document")
    
    # Chunk the text
    print(f"\nChunking text (chunk_size={args.chunk}, overlap={args.overlap})...")
    chunks = chunk_text(text, chunk_size=args.chunk, chunk_overlap=args.overlap)
    print(f"SUCCESS: Created {len(chunks)} chunks")
    
    if not chunks:
        print("ERROR: No chunks were created from the input text.", file=sys.stderr)
        return 5
    
    # Add metadata
    for i, c in enumerate(chunks):
        c["id"] = str(uuid.uuid4())
        c["metadata"].update({
            "source": os.path.normpath(args.inp),
            "chunk_index": i
        })
    
    # Generate embeddings
    print(f"\nGenerating embeddings using model '{args.model}'...")
    try:
        vectors = embed_chunks(chunks, model_name=args.model, batch_size=args.batch)
    except Exception as e:
        print(f"ERROR: Error generating embeddings: {e}", file=sys.stderr)
        return 6
    
    print(f"SUCCESS: Generated {len(vectors)} embeddings (384 dimensions each)")
    
    # Clean old data if requested
    if args.clean:
        print(f"\nCleaning old documents...")
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                if args.user_id:
                    cur.execute("DELETE FROM documents WHERE user_id = %s;", (args.user_id,))
                    deleted = cur.rowcount
                    print(f"SUCCESS: Deleted {deleted} old documents for user {args.user_id}")
                else:
                    cur.execute("DELETE FROM documents WHERE user_id IS NULL;")
                    deleted = cur.rowcount
                    print(f"SUCCESS: Deleted {deleted} old documents (no user_id)")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"WARNING: Could not clean old documents: {e}")
    
    # Store in Supabase
    print(f"\nStoring data in Supabase...")
    if args.user_id:
        print(f"Using user_id: {args.user_id} (RLS-protected)")
    try:
        store_to_supabase(chunks, vectors, os.path.normpath(args.inp), user_id=args.user_id)
    except Exception as e:
        print(f"ERROR: Error storing to Supabase: {e}", file=sys.stderr)
        return 7
    
    print(f"\nSUCCESS! Processed {len(chunks)} chunks from '{args.inp}' and stored in Supabase")
    return 0


if __name__ == "__main__":
    sys.exit(main())