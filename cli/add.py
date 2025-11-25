"""
Quick script to add a document using the incremental RAPTOR builder
Supports semantic chunking and configurable strategies

Includes graceful interrupt handling - if you Ctrl+C during ingestion,
the partial data will be cleaned up automatically.
"""
import sys
import os
import uuid
import signal
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embeddings.incremental_raptor import IncrementalRaptorBuilder
from utils.chunking import chunk_text, ChunkingStrategy  # New improved chunking
from auth import AuthManager

load_dotenv()

# Global flag for interrupt handling
_interrupted = False
_current_doc_id = None
_current_user_id = None


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global _interrupted
    _interrupted = True
    print("\n\n️  Interrupt received! Cleaning up...")


def cleanup_partial_ingestion(doc_id: str, user_id: str):
    """Clean up a partially ingested document"""
    try:
        from cleanup_db import DatabaseCleaner
        cleaner = DatabaseCleaner()
        print(f"\n Cleaning up partial document {doc_id[:8]}...")
        cleaner.cleanup_document(user_id, doc_id=doc_id, dry_run=False)
        print(" Cleanup complete. Database is clean.")
    except Exception as e:
        print(f"️  Cleanup failed: {e}")
        print(f"   Run 'python cleanup_db.py' to manually clean up.")


def read_document(path: str) -> str:
    """Read TXT, PDF, or DOCX file"""
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".txt":
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            text = text.replace('\x00', '')
            return text
    
    elif ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    
    elif ext == ".docx":
        import docx2txt
        text = docx2txt.process(path) or ""
        return " ".join(text.split())
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def add_document(file_path: str, user_id: str, chunking_strategy: str = "semantic"):
    """
    Add a document to the RAPTOR system using incremental builder
    
    Args:
        file_path: Path to the document (txt, pdf, docx)
        user_id: User UUID
        chunking_strategy: One of "fixed", "semantic", "sliding", "hierarchical"
    """
    global _interrupted, _current_doc_id, _current_user_id
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    # Set up interrupt handler
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    _interrupted = False
    
    print(f"\nAdding document: {file_path}")
    print(f"User ID: {user_id}")
    print(f"Chunking strategy: {chunking_strategy}")
    print("(Press Ctrl+C to cancel - partial data will be cleaned up)\n")
    
    # Read document
    print(f"Reading document...")
    text = read_document(file_path)
    print(f"Document length: {len(text):,} characters")
    
    if _interrupted:
        signal.signal(signal.SIGINT, old_handler)
        print("Cancelled before processing.")
        return None
    
    # Chunk text with improved semantic chunking
    print(f"\n Chunking text ({chunking_strategy} strategy)...")
    raw_chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200, strategy=chunking_strategy)
    print(f"   Created {len(raw_chunks)} chunks")
    
    if _interrupted:
        signal.signal(signal.SIGINT, old_handler)
        print("Cancelled before ingestion.")
        return None
    
    # Prepare chunks with IDs
    doc_id = str(uuid.uuid4())
    _current_doc_id = doc_id
    _current_user_id = user_id
    
    chunks = []
    for i, chunk in enumerate(raw_chunks):
        chunks.append({
            'id': str(uuid.uuid4()),
            'text': chunk['text']
        })
    
    # Add document incrementally
    print(f"\n Adding document incrementally...")
    builder = IncrementalRaptorBuilder()
    
    import time
    start_time = time.time()
    
    try:
        # Pass the actual filename (not doc_id) for metadata
        filename = os.path.basename(file_path)
        builder.add_document_incremental(chunks, doc_id, user_id, source_file=filename)
        
        if _interrupted:
            raise KeyboardInterrupt("User cancelled")
        
        elapsed = time.time() - start_time
        
        print(f"\n Document added in {elapsed:.2f} seconds!")
        print(f"   Document ID: {doc_id}")
        
        # Get updated stats
        print(f"\n Getting tree statistics...")
        stats = builder.get_tree_stats(user_id)
        print(f"   Total chunks: {stats.get('total_chunks', 0)}")
        print(f"   Active chunks: {stats.get('active_chunks', 0)}")
        print(f"   Tree levels: {stats.get('tree_levels', 0)}")
        print(f"   Nodes per level: {stats.get('nodes_per_level', {})}")
        
        return doc_id
        
    except KeyboardInterrupt:
        print("\n\n️  Document ingestion was interrupted!")
        cleanup_partial_ingestion(doc_id, user_id)
        return None
        
    except Exception as e:
        print(f"\n\n Error during ingestion: {e}")
        print("Cleaning up partial data...")
        cleanup_partial_ingestion(doc_id, user_id)
        raise
        
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, old_handler)
        _current_doc_id = None
        _current_user_id = None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add a document to RAPTOR")
    parser.add_argument("file_path", help="Path to the document")
    parser.add_argument("--strategy", default="semantic",
                        choices=["fixed", "semantic", "sliding", "hierarchical"],
                        help="Chunking strategy (default: semantic)")
    
    args = parser.parse_args()
    
    # Load user from session
    auth = AuthManager()
    user_id = auth.load_session()
    
    if not user_id:
        print("Not logged in! Please login first:")
        print("  python auth.py login <username> <password>")
        print("\nOr register a new account:")
        print("  python auth.py register <username> <password>")
        sys.exit(1)
    
    add_document(args.file_path, user_id, args.strategy)
