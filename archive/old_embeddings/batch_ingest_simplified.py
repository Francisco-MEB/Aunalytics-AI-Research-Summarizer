"""
Simplified Batch Ingestion - Clean, maintainable version
Key improvements:
- Removed RAPTOR hierarchy complexity (optional, can be added later)
- Simpler incremental logic
- Better error handling
- Clearer code structure
"""
import argparse
import os
import sys
import uuid
import hashlib
from typing import List, Dict, Set
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import docx2txt
from pypdf import PdfReader
import psycopg2
from psycopg2.extras import execute_values
import json

load_dotenv()


def read_pdf(path: str) -> str:
    """Read text from PDF file"""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def read_docx(path: str) -> str:
    """Extract text from DOCX file"""
    text = docx2txt.process(path) or ""
    return " ".join(text.split())


def read_document(path: str) -> str:
    """Read TXT, PDF, or DOCX file"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            text = text.replace('\x00', '')
            return text
    elif ext == ".pdf":
        return read_pdf(path)
    elif ext == ".docx":
        return read_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def compute_file_hash(path: str) -> str:
    """Generate SHA256 hash of file content for unique identification"""
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_files_from_path(path: str) -> List[str]:
    """Get list of supported files from path (file or directory)"""
    supported_extensions = {'.txt', '.pdf', '.docx'}
    
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        files = []
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                if Path(filename).suffix.lower() in supported_extensions:
                    files.append(os.path.join(root, filename))
        return sorted(files)
    else:
        raise ValueError(f"Path does not exist: {path}")


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Dict]:
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    splits = splitter.split_text(text)
    return [{"text": chunk, "metadata": {}} for chunk in splits]


def get_db_config():
    """Get database configuration"""
    return {
        'user': os.getenv('user'),
        'password': os.getenv('password'),
        'host': os.getenv('host'),
        'port': int(os.getenv('port', '5432')),
        'dbname': os.getenv('dbname')
    }


def get_existing_file_hashes(user_id: str) -> Set[str]:
    """Get set of file hashes already in database"""
    db_config = get_db_config()
    conn = psycopg2.connect(**db_config)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT DISTINCT metadata->>'file_hash' as file_hash
                   FROM documents
                   WHERE user_id = %s AND hierarchy_level = 0
                   AND metadata->>'file_hash' IS NOT NULL""",
                (user_id,)
            )
            return {row[0] for row in cur.fetchall() if row[0]}
    finally:
        conn.close()


def display_existing_documents(user_id: str):
    """Display information about documents already in database"""
    db_config = get_db_config()
    conn = psycopg2.connect(**db_config)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT metadata->>'source_file' as source_file,
                          metadata->>'file_hash' as file_hash,
                          COUNT(*) as chunk_count
                   FROM documents
                   WHERE user_id = %s AND hierarchy_level = 0
                   GROUP BY metadata->>'source_file', metadata->>'file_hash'
                   ORDER BY metadata->>'source_file'""",
                (user_id,)
            )
            rows = cur.fetchall()
            
            if rows:
                print(f"\n Existing documents in database ({len(rows)} documents):")
                for source_file, file_hash, chunk_count in rows:
                    file_name = os.path.basename(source_file) if source_file else "Unknown"
                    hash_display = file_hash[:8] if file_hash else "no-hash"
                    print(f"  • {file_name} ({chunk_count} chunks) [hash: {hash_display}...]")
            else:
                print("\n No existing documents in database")
    finally:
        conn.close()


def delete_all_documents(user_id: str) -> int:
    """Delete all documents for a user (--clean mode)"""
    db_config = get_db_config()
    conn = psycopg2.connect(**db_config)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE user_id = %s;", (user_id,))
            deleted = cur.rowcount
            conn.commit()
            return deleted
    finally:
        conn.close()


def process_and_upload_files(file_paths: List[str], user_id: str, 
                              chunk_size: int, chunk_overlap: int, 
                              model_name: str, skip_hashes: Set[str] = None):
    """
    Process files and upload to database
    Simplified version: just chunks -> embeddings -> upload
    No complex hierarchy building (can be added as separate step if needed)
    """
    if skip_hashes is None:
        skip_hashes = set()
    
    # Filter out duplicates
    files_to_process = []
    skipped_files = []
    
    print(f"\n Checking {len(file_paths)} files for duplicates...")
    for file_path in file_paths:
        file_hash = compute_file_hash(file_path)
        if file_hash in skip_hashes:
            file_name = os.path.basename(file_path)
            skipped_files.append(file_name)
            print(f"  ⏭️  Skipping {file_name} (already uploaded)")
        else:
            files_to_process.append((file_path, file_hash))
    
    if skipped_files:
        print(f"\n Skipped {len(skipped_files)} duplicate files")
    
    if not files_to_process:
        print(f"\n All files already in database! No new documents to process.")
        return
    
    # Process new files
    print(f"\n Processing {len(files_to_process)} NEW files...")
    
    model = SentenceTransformer(model_name)
    db_config = get_db_config()
    
    total_chunks_added = 0
    
    for file_path, file_hash in files_to_process:
        file_name = os.path.basename(file_path)
        print(f"\n   Reading: {file_name}")
        
        try:
            # Read document
            text = read_document(file_path)
            print(f"     Read {len(text):,} characters")
            
            # Chunk text
            chunks = chunk_text(text, chunk_size, chunk_overlap)
            print(f"     Created {len(chunks)} chunks")
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                chunk["id"] = str(uuid.uuid4())
                chunk["metadata"] = {
                    "source_file": os.path.normpath(file_path),
                    "file_hash": file_hash,
                    "chunk_index": i
                }
            
            # Generate embeddings
            print(f"     Generating embeddings...")
            texts = [c["text"] for c in chunks]
            embeddings = model.encode(texts, batch_size=64, convert_to_numpy=True, 
                                     show_progress_bar=False, normalize_embeddings=True)
            
            # Upload to database
            print(f"     Uploading to database...")
            conn = psycopg2.connect(**db_config)
            try:
                data_to_insert = []
                for chunk, embedding in zip(chunks, embeddings):
                    data_to_insert.append((
                        chunk["id"],
                        chunk["text"],
                        embedding.tolist(),
                        json.dumps(chunk["metadata"]),
                        user_id,
                        0  # hierarchy_level = 0 (raw chunks)
                    ))
                
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        """INSERT INTO documents (doc_id, content, embedding, metadata, user_id, hierarchy_level)
                           VALUES %s""",
                        data_to_insert
                    )
                    conn.commit()
                
                print(f"      Stored {len(data_to_insert)} chunks")
                total_chunks_added += len(data_to_insert)
            finally:
                conn.close()
            
        except Exception as e:
            print(f"      Error processing {file_name}: {e}")
            continue
    
    print(f"\n SUCCESS! Added {total_chunks_added} total chunks from {len(files_to_process)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Simplified Batch Ingestion - Upload documents to vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload new documents (incremental - skips duplicates)
  python batch_ingest_simplified.py --in data/papers --user-id <uuid>
  
  # Clean and re-upload everything
  python batch_ingest_simplified.py --in data/papers --user-id <uuid> --clean
        """
    )
    
    parser.add_argument("--in", dest="inp", required=True,
                       help="Input file or directory containing documents")
    parser.add_argument("--user-id", dest="user_id", required=True,
                       help="User UUID for Row-Level Security")
    parser.add_argument("--model", dest="model", 
                       default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Sentence transformer model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--chunk", dest="chunk", type=int, default=1000,
                       help="Chunk size in characters (default: 1000)")
    parser.add_argument("--overlap", dest="overlap", type=int, default=100,
                       help="Chunk overlap in characters (default: 100)")
    parser.add_argument("--clean", dest="clean", action="store_true",
                       help="Delete ALL existing documents before upload (DESTRUCTIVE!)")
    
    args = parser.parse_args()
    
    # Validate database config
    db_config = get_db_config()
    if not all(db_config.values()):
        print(" Error: Missing database connection parameters in .env file")
        return 1
    
    # Display existing documents
    print("=" * 60)
    print("BATCH DOCUMENT INGESTION")
    print("=" * 60)
    display_existing_documents(args.user_id)
    
    # Clean mode (full delete)
    skip_hashes = set()
    if args.clean:
        print("\n️  FULL CLEAN MODE")
        print("️  This will DELETE ALL your documents!")
        confirmation = input("Type 'DELETE' to confirm: ")
        if confirmation != 'DELETE':
            print(" Aborted. No documents were deleted.")
            return 0
        
        deleted = delete_all_documents(args.user_id)
        print(f" Deleted {deleted} existing documents")
    else:
        # Incremental mode - get existing hashes
        print("\n INCREMENTAL MODE (skips duplicates)")
        skip_hashes = get_existing_file_hashes(args.user_id)
        if skip_hashes:
            print(f"   Found {len(skip_hashes)} existing file hashes")
    
    # Get files to process
    try:
        files = get_files_from_path(args.inp)
        if not files:
            print(f" No supported files found in {args.inp}")
            return 1
        print(f"\n Found {len(files)} total files in input path")
    except Exception as e:
        print(f" Error: {e}")
        return 1
    
    # Process and upload
    process_and_upload_files(
        files, 
        args.user_id,
        args.chunk, 
        args.overlap, 
        args.model,
        skip_hashes
    )
    
    # Show final stats
    print("\n" + "=" * 60)
    print("FINAL DATABASE STATE:")
    print("=" * 60)
    display_existing_documents(args.user_id)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
