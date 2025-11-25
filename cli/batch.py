"""
Batch Document Processing for RAPTOR System
Efficiently process multiple documents with parallel processing and progress tracking
"""
import os
import sys
import uuid
import time
import hashlib
from typing import List, Dict, Optional, Tuple, Generator
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import json

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chunking import chunk_text, ChunkConfig, ChunkingStrategy
from utils.caching import EmbeddingCache, CacheConfig, CachedEmbedder

load_dotenv()


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: str = "semantic"  # "fixed", "semantic", "sliding", "hierarchical"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_workers: int = 4                  # Parallel file reading
    embedding_batch_size: int = 64        # Batch size for embedding model
    use_cache: bool = True
    skip_duplicates: bool = True          # Skip files already in database
    dry_run: bool = False                 # Preview without actually inserting
    verbose: bool = True


@dataclass
class ProcessingResult:
    """Result from processing a single file"""
    file_path: str
    success: bool
    doc_id: Optional[str] = None
    num_chunks: int = 0
    error: Optional[str] = None
    processing_time: float = 0.0
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class BatchResult:
    """Result from batch processing"""
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_chunks: int = 0
    total_time: float = 0.0
    results: List[ProcessingResult] = field(default_factory=list)


class BatchProcessor:
    """
    Efficiently process multiple documents for RAPTOR ingestion.
    Features:
    - Parallel file reading
    - Batched embedding computation
    - Caching to avoid re-computing embeddings
    - Progress tracking
    - Duplicate detection
    """
    
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx'}
    
    def __init__(self, config: Optional[BatchConfig] = None, user_id: str = None):
        self.config = config or BatchConfig()
        self.user_id = user_id
        
        # Database config
        self.db_config = {
            'user': os.getenv('user'),
            'password': os.getenv('password'),
            'host': os.getenv('host'),
            'port': int(os.getenv('port', '5432')),
            'dbname': os.getenv('dbname')
        }
        
        # Initialize caching
        if self.config.use_cache:
            cache_config = CacheConfig()
            self.embedder = CachedEmbedder(self.config.embedding_model)
        else:
            self.embedder = None
            self._model = None
        
        # Track existing file hashes
        self._existing_hashes = set()
    
    @property
    def model(self):
        """Lazy load embedding model"""
        if self.embedder:
            return self.embedder.model
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model
    
    def _get_conn(self):
        """Get database connection"""
        import psycopg2
        return psycopg2.connect(**self.db_config)
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file content"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        return sha256.hexdigest()
    
    def _load_existing_hashes(self):
        """Load hashes of files already in database"""
        if not self.config.skip_duplicates:
            return
        
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT metadata->>'file_hash' as file_hash
                    FROM chunks
                    WHERE user_id = %s AND deleted = FALSE
                      AND metadata->>'file_hash' IS NOT NULL
                """, (self.user_id,))
                self._existing_hashes = {row[0] for row in cur.fetchall() if row[0]}
            conn.close()
            
            if self.config.verbose:
                print(f"   Found {len(self._existing_hashes)} existing files in database")
        except Exception as e:
            if self.config.verbose:
                print(f"  ️ Could not load existing hashes: {e}")
    
    def _read_document(self, file_path: str) -> Tuple[str, str]:
        """Read document and return (text, file_hash)"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().replace('\x00', '')
        
        elif ext == '.pdf':
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        elif ext == '.docx':
            import docx2txt
            text = docx2txt.process(file_path) or ""
            text = " ".join(text.split())
        
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        file_hash = self._compute_file_hash(file_path)
        return text, file_hash
    
    def _get_files_from_path(self, path: str) -> List[str]:
        """Get list of supported files from path (file or directory)"""
        path = Path(path)
        
        if path.is_file():
            if path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                return [str(path)]
            return []
        
        elif path.is_dir():
            files = []
            for ext in self.SUPPORTED_EXTENSIONS:
                files.extend(path.rglob(f"*{ext}"))
            return sorted([str(f) for f in files])
        
        return []
    
    def _process_single_file(self, file_path: str) -> ProcessingResult:
        """Process a single file (reading + chunking)"""
        start_time = time.time()
        
        try:
            # Read document
            text, file_hash = self._read_document(file_path)
            
            # Check for duplicates
            if self.config.skip_duplicates and file_hash in self._existing_hashes:
                return ProcessingResult(
                    file_path=file_path,
                    success=True,
                    skipped=True,
                    skip_reason="Already in database",
                    processing_time=time.time() - start_time
                )
            
            # Chunk text
            chunks = chunk_text(
                text,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                strategy=self.config.chunking_strategy
            )
            
            # Add metadata
            doc_id = str(uuid.uuid4())
            filename = os.path.basename(file_path)
            
            for i, chunk in enumerate(chunks):
                chunk['id'] = str(uuid.uuid4())
                chunk['metadata']['source_file'] = filename
                chunk['metadata']['file_hash'] = file_hash
                chunk['metadata']['doc_id'] = doc_id
            
            return ProcessingResult(
                file_path=file_path,
                success=True,
                doc_id=doc_id,
                num_chunks=len(chunks),
                processing_time=time.time() - start_time
            ), chunks
        
        except Exception as e:
            return ProcessingResult(
                file_path=file_path,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            ), []
    
    def process_files(self, paths: List[str], progress_callback=None) -> Generator[Tuple[ProcessingResult, List[Dict]], None, None]:
        """
        Process multiple files in parallel (reading phase).
        Yields (result, chunks) for each file.
        """
        # Collect all files
        all_files = []
        for path in paths:
            all_files.extend(self._get_files_from_path(path))
        
        if not all_files:
            return
        
        if self.config.verbose:
            print(f" Found {len(all_files)} files to process")
        
        # Load existing hashes for duplicate detection
        self._load_existing_hashes()
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, f): f 
                for f in all_files
            }
            
            completed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                completed += 1
                
                try:
                    result = future.result()
                    if isinstance(result, tuple):
                        proc_result, chunks = result
                    else:
                        proc_result, chunks = result, []
                    
                    if progress_callback:
                        progress_callback(completed, len(all_files), proc_result)
                    
                    yield proc_result, chunks
                    
                except Exception as e:
                    yield ProcessingResult(
                        file_path=file_path,
                        success=False,
                        error=str(e)
                    ), []
    
    def batch_embed(self, all_chunks: List[Dict]) -> List[List[float]]:
        """
        Compute embeddings for all chunks in batches.
        Uses caching if enabled.
        """
        texts = [c['text'] for c in all_chunks]
        
        if self.config.verbose:
            print(f" Computing embeddings for {len(texts)} chunks...")
        
        if self.embedder:
            # Use cached embedder
            embeddings = self.embedder.encode(texts)
            return embeddings
        else:
            # Direct embedding
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.embedding_batch_size,
                convert_to_numpy=True,
                show_progress_bar=self.config.verbose
            )
            return embeddings.tolist()
    
    def store_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> int:
        """Store chunks with embeddings in database"""
        if self.config.dry_run:
            if self.config.verbose:
                print(f"   DRY RUN: Would store {len(chunks)} chunks")
            return len(chunks)
        
        from psycopg2.extras import execute_values
        
        conn = self._get_conn()
        try:
            data = []
            for chunk, emb in zip(chunks, embeddings):
                doc_id = chunk['metadata'].get('doc_id', str(uuid.uuid4()))
                data.append((
                    chunk['id'],
                    doc_id,
                    self.user_id,
                    chunk['text'],
                    emb,
                    False,  # deleted
                    json.dumps(chunk['metadata'])
                ))
            
            with conn.cursor() as cur:
                execute_values(cur, """
                    INSERT INTO chunks (chunk_id, doc_id, user_id, text, embedding, deleted, metadata)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO NOTHING
                """, data)
                conn.commit()
                
            return len(data)
        finally:
            conn.close()
    
    def run(self, paths: List[str]) -> BatchResult:
        """
        Run full batch processing pipeline.
        
        Args:
            paths: List of file paths or directories
        
        Returns:
            BatchResult with statistics
        """
        start_time = time.time()
        result = BatchResult()
        
        # Collect all chunks from all files
        all_chunks = []
        
        def progress(completed, total, proc_result):
            status = "" if proc_result.success else ""
            if proc_result.skipped:
                status = "⏭"
            if self.config.verbose:
                filename = os.path.basename(proc_result.file_path)
                print(f"  [{completed}/{total}] {status} {filename}")
        
        print("\n Phase 1: Reading and chunking files...")
        for proc_result, chunks in self.process_files(paths, progress):
            result.results.append(proc_result)
            result.total_files += 1
            
            if proc_result.skipped:
                result.skipped += 1
            elif proc_result.success:
                result.successful += 1
                all_chunks.extend(chunks)
            else:
                result.failed += 1
        
        if not all_chunks:
            print("\n️ No new chunks to process")
            result.total_time = time.time() - start_time
            return result
        
        # Embed all chunks
        print(f"\n Phase 2: Computing embeddings for {len(all_chunks)} chunks...")
        embeddings = self.batch_embed(all_chunks)
        
        # Store in database
        print(f"\n Phase 3: Storing in database...")
        stored = self.store_chunks(all_chunks, embeddings)
        result.total_chunks = stored
        
        result.total_time = time.time() - start_time
        
        # Summary
        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 60)
        print(f"  Files processed: {result.total_files}")
        print(f"  Successful: {result.successful}")
        print(f"  Skipped: {result.skipped}")
        print(f"  Failed: {result.failed}")
        print(f"  Total chunks: {result.total_chunks}")
        print(f"  Time: {result.total_time:.2f}s")
        
        if self.embedder and self.config.use_cache:
            print(f"\n  Cache stats: {self.embedder.stats()}")
        
        return result


def batch_add_documents(paths: List[str], user_id: str, 
                        chunk_size: int = 1000,
                        chunk_overlap: int = 200,
                        strategy: str = "semantic",
                        skip_duplicates: bool = True,
                        dry_run: bool = False) -> BatchResult:
    """
    Convenience function for batch document ingestion.
    
    Args:
        paths: List of file paths or directories
        user_id: User UUID
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        strategy: Chunking strategy ("fixed", "semantic", "sliding", "hierarchical")
        skip_duplicates: Skip files already in database
        dry_run: Preview without storing
    
    Returns:
        BatchResult with statistics
    """
    config = BatchConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunking_strategy=strategy,
        skip_duplicates=skip_duplicates,
        dry_run=dry_run
    )
    
    processor = BatchProcessor(config, user_id)
    return processor.run(paths)


# CLI
if __name__ == "__main__":
    import argparse
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from auth import AuthManager
    
    parser = argparse.ArgumentParser(description="Batch document processing for RAPTOR")
    parser.add_argument("paths", nargs="+", help="Files or directories to process")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument("--strategy", default="semantic", 
                        choices=["fixed", "semantic", "sliding", "hierarchical"],
                        help="Chunking strategy")
    parser.add_argument("--no-cache", action="store_true", help="Disable embedding cache")
    parser.add_argument("--force", action="store_true", help="Process duplicates too")
    parser.add_argument("--dry-run", action="store_true", help="Preview without storing")
    
    args = parser.parse_args()
    
    # Check auth
    auth = AuthManager()
    user_id = auth.load_session()
    
    if not user_id:
        print(" Not logged in! Please run:")
        print("   python auth.py login <username> <password>")
        sys.exit(1)
    
    print(f" Logged in as: {auth.current_user['username']}")
    
    config = BatchConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=args.strategy,
        use_cache=not args.no_cache,
        skip_duplicates=not args.force,
        dry_run=args.dry_run
    )
    
    processor = BatchProcessor(config, user_id)
    result = processor.run(args.paths)
    
    if result.failed > 0:
        print("\n️ Some files failed:")
        for r in result.results:
            if not r.success and not r.skipped:
                print(f"  - {r.file_path}: {r.error}")
