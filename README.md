# RAPTOR QA System

A sophisticated question-answering system using RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) hierarchical document processing with PostgreSQL/pgvector storage.

## Features

- **RAPTOR Hierarchical Processing**: Documents are chunked, embedded, clustered, and summarized into a tree structure for better retrieval
- **Multiple Chunking Strategies**: Semantic, sliding window, recursive, and sentence-based chunking
- **Advanced Clustering**: K-means, hierarchical, DBSCAN, Gaussian Mixture, spectral, and mini-batch K-means
- **Embedding Caching**: LRU cache to avoid redundant API calls
- **Query Caching**: Cache query results for faster repeated queries
- **Streaming Answers**: Real-time streaming responses from the LLM
- **Batch Processing**: Process multiple documents or queries in bulk
- **Database Cleanup**: Comprehensive tools for soft delete, hard delete, and orphan cleanup

## Project Structure

```
project/
    main.py              # Main entry point
    requirements.txt     # Python dependencies
    .env                 # Environment variables (not committed)
    
    core/                # Core processing modules
        __init__.py
        raptor.py        # Incremental RAPTOR tree builder
        chunking.py      # Document chunking strategies
        clustering.py    # Embedding clustering methods
        caching.py       # Embedding and query caching
    
    cli/                 # Command-line tools
        qa.py            # Interactive Q&A mode
        add.py           # Add documents
        cleanup.py       # Database cleanup utility
        streaming.py     # Streaming Q&A mode
        batch.py         # Batch processing
        auth.py          # User authentication
    
    docs/                # Documentation
        architecture.md
        chunking.md
        usage.md
    
    sql/                 # Database schemas
        schema.sql
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Aunalytics-AI-Research-Summarizer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
OPENAI_API_KEY=your_openai_key
```

## Usage

### Main Entry Point

```bash
python main.py --help
```

### Add Documents

```bash
# Add a single file
python main.py add path/to/document.txt

# Add with custom chunk size
python main.py add document.txt --chunk-size 1000 --chunk-overlap 100
```

### Interactive Q&A

```bash
python main.py qa
```

### Streaming Q&A

```bash
python main.py stream
```

### Batch Processing

```bash
# Process multiple documents
python main.py batch documents/ --pattern "*.txt"

# Process queries from file
python main.py batch --queries queries.txt
```

### Database Cleanup

```bash
# View statistics
python main.py cleanup --stats

# Validate database integrity
python main.py cleanup --validate

# Soft delete cleanup (preview)
python main.py cleanup --cleanup

# Execute soft delete cleanup
python main.py cleanup --cleanup --execute

# Hard delete (purge) - permanent
python main.py cleanup --purge --execute
```

## Architecture

The system uses a hierarchical tree structure:

1. **Chunks**: Documents are split into manageable chunks
2. **Leaf Nodes**: Chunks are embedded and stored as leaf nodes
3. **Tree Nodes**: Clusters of chunks are summarized and stored as parent nodes
4. **Multi-level**: The process repeats, building a hierarchy

### Database Tables

- **chunks**: Stores document chunks with embeddings and metadata
- **tree_nodes**: Stores the RAPTOR tree hierarchy
- **chunk_to_leaf**: Maps chunks to their leaf node positions

## Configuration

Key settings in `.env`:

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase API key |
| `OPENAI_API_KEY` | OpenAI API key for embeddings and LLM |
| `CHUNK_SIZE` | Default chunk size (default: 500) |
| `CHUNK_OVERLAP` | Chunk overlap (default: 50) |
| `EMBEDDING_MODEL` | Embedding model (default: text-embedding-3-small) |
| `LLM_MODEL` | LLM model (default: gpt-4o-mini) |

## Development

### Running Tests

```bash
python test.py
```

### Cleaning Up

If document ingestion is interrupted, use the cleanup utility:

```bash
# Check for orphans
python main.py cleanup --validate

# Clean up orphaned data
python main.py cleanup --cleanup --execute
```

## License

MIT License
