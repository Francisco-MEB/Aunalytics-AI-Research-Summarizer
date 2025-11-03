#!/usr/bin/env python3
"""
Document Ingestion Script for AI Research Summarizer
====================================================
Processes documents (PDF, DOCX, TXT) and creates embeddings for vector search.

Workflow:
1. Read document and extract text
2. Split text into chunks (with overlap for context)
3. Generate embeddings using sentence-transformers
4. Save to JSONL format with metadata

Output: JSONL file where each line contains:
{
    "id": "unique-uuid",
    "text": "chunk text content",
    "metadata": {"source": "file_path", "chunk_index": 0},
    "embedding": [0.123, 0.456, ...]  # 384-dim vector
}
"""
import argparse
import json
import os
import sys
import uuid
from typing import List, Dict

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import docx2txt
from pypdf import PdfReader
def read_pdf(path: str) -> str:
    """
    Read text content from a PDF file.
    
    Args:
        path: Path to PDF file
        
    Returns:
        Extracted text from all pages
    """
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def read_docx(path: str) -> str:
    """
    Extract text from a DOCX file.
    
    Args:
        path: Path to DOCX file
        
    Returns:
        Extracted text with normalized whitespace
    """
    text = docx2txt.process(path) or ""
    # Normalize whitespace
    return " ".join(text.split())


def read_document(path: str) -> str:
    """
    Read document and extract text (supports TXT, PDF, DOCX).
    
    Args:
        path: Path to document file
        
    Returns:
        Extracted text content
        
    Raises:
        ValueError: If file type is not supported
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".txt":
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif ext == ".pdf":
        print(f" Detected PDF format. Extracting text from '{path}'...")
        return read_pdf(path)
    elif ext == ".docx":
        print(f" Detected DOCX format. Extracting text from '{path}'...")
        return read_docx(path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported formats: .txt, .pdf, .docx"
        )

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """
    Split text into overlapping chunks for processing.
    
    Chunks are created with overlap to preserve context across boundaries.
    Uses RecursiveCharacterTextSplitter which tries to split on:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Spaces
    4. Characters (as last resort)
    
    Args:
        text: Input text to chunk
        chunk_size: Target size in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of dicts with 'text' and 'metadata' keys
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    
    docs = splitter.create_documents([text], metadatas=[{}])
    return [{"text": d.page_content, "metadata": d.metadata} for d in docs]


def embed_chunks(chunks: List[Dict], model_name: str, batch_size: int = 64) -> List[List[float]]:
    """
    Generate embeddings for text chunks using sentence-transformers.
    
    Args:
        chunks: List of chunk dicts (each with 'text' key)
        model_name: HuggingFace model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        batch_size: Number of texts to process at once
        
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    print(f" Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    texts = [c["text"] for c in chunks]
    
    print(f"🔄 Generating embeddings for {len(texts)} chunks...")
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True  # L2 normalization for cosine similarity
    )
    
    # Convert numpy arrays to Python lists for JSON serialization
    return [v.tolist() for v in vectors]


def write_jsonl(out_path: str, records: List[Dict]) -> None:
    """
    Write records to JSONL file (one JSON object per line).
    
    Args:
        out_path: Output file path
        records: List of dicts to write
    """
    # Create directory if it doesn't exist
    dir_name = os.path.dirname(out_path) or "."
    os.makedirs(dir_name, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main() -> int:
    """
    Main entry point for document ingestion.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Ingest documents -> chunks -> embeddings -> JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a PDF with default settings
  python ingest.py --in research.pdf --out embeddings.jsonl
  
  # Custom chunk size and model
  python ingest.py --in paper.pdf --out data.jsonl --chunk 500 --overlap 50
  
  # Use different embedding model
  python ingest.py --in doc.txt --out out.jsonl --model sentence-transformers/paraphrase-MiniLM-L6-v2
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--in",
        dest="inp",
        required=True,
        help="Input file path (.txt, .pdf, or .docx)"
    )
    parser.add_argument(
        "--out",
        dest="out",
        required=True,
        help="Output JSONL file path"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model name (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--chunk",
        dest="chunk",
        type=int,
        default=1000,
        help="Approximate characters per chunk (default: 1000)"
    )
    parser.add_argument(
        "--overlap",
        dest="overlap",
        type=int,
        default=100,
        help="Overlap between chunks in characters (default: 100)"
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=64,
        help="Embedding batch size (default: 64)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.isfile(args.inp):
        print(f" Error: Input file '{args.inp}' does not exist.", file=sys.stderr)
        return 2
    
    # Read document
    print(f"\n Reading document: {args.inp}")
    try:
        text = read_document(args.inp)
    except Exception as e:
        print(f" Error reading document: {e}", file=sys.stderr)
        return 3
    
    # Validate content
    if not text.strip():
        print(f" Error: Document is empty or contains only whitespace.", file=sys.stderr)
        return 4
    
    print(f"✅ Read {len(text):,} characters")
    
    # Chunk the text
    print(f"\n Chunking text (size={args.chunk}, overlap={args.overlap})...")
    chunks = chunk_text(text, chunk_size=args.chunk, chunk_overlap=args.overlap)
    print(f"✅ Created {len(chunks)} chunks")
    
    if not chunks:
        print("Error: No chunks were created from the input text.", file=sys.stderr)
        return 5
    
    # Add metadata to each chunk
    for i, chunk in enumerate(chunks):
        chunk["id"] = str(uuid.uuid4())
        chunk["metadata"].update({
            "source": os.path.normpath(args.inp),
            "chunk_index": i,
            "total_chunks": len(chunks)
        })
    
    # Generate embeddings
    print(f"\n Generating embeddings...")
    try:
        vectors = embed_chunks(chunks, model_name=args.model, batch_size=args.batch)
    except Exception as e:
        print(f"Error generating embeddings: {e}", file=sys.stderr)
        return 6
    
    # Combine chunks and embeddings into records
    print(f"\n Writing to {args.out}...")
    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": vector
        })
    
    # Write to JSONL
    try:
        write_jsonl(args.out, records)
        print(f" Successfully wrote {len(records)} records to '{args.out}'")
        print(f"\n Summary:")
        print(f"   - Input file: {args.inp}")
        print(f"   - Output file: {args.out}")
        print(f"   - Chunks created: {len(records)}")
        print(f"   - Embedding dimension: {len(vectors[0]) if vectors else 0}")
        print(f"   - Model used: {args.model}")
        return 0
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        return 7

if __name__ == "__main__":
    sys.exit(main())
