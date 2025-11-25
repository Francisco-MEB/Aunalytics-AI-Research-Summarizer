# RAPTOR Hierarchical Document System

Multi-file research paper ingestion with hierarchical clustering and intelligent retrieval.

## Features

- **Multi-File Batch Processing**: Process entire directories of research papers
- **RAPTOR Hierarchical Clustering**: K-means clustering with LLM-powered summarization
- **3-Level Hierarchy**:
  - Level 0: Original document chunks
  - Level 1: Cluster summaries
  - Level 2: High-level document overview
- **Intelligent Question Routing**: Adaptive retrieval based on question type
- **Row-Level Security**: Multi-tenant data isolation

## Quick Start

### 1. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
```

Required environment variables:
```env
# Supabase Connection (Transaction Pooler)
user=postgres.jitakvphoodobwvlvsqu
password=YOUR_PASSWORD
host=aws-1-us-east-2.pooler.supabase.com
port=5432
dbname=postgres

# Google Gemini API Key
# Get from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Database Schema

Run the SQL script in Supabase SQL Editor:
```bash
# Copy contents of supabase_schema.sql
# Paste into Supabase SQL Editor and run
```

### 4. Ingest Documents

**Single File:**
```bash
python embeddings/store_to_supabase --in path/to/paper.pdf --user-id YOUR_UUID --raptor-mode clustering
```

**Multiple Files (Directory):**
```bash
python embeddings/batch_ingest.py --in data/papers/ --user-id YOUR_UUID --raptor-mode clustering
```

**Clean existing data before ingestion:**
```bash
python embeddings/batch_ingest.py --in data/test/ --user-id YOUR_UUID --raptor-mode clustering --clean
```

### 5. Query System

```bash
python app/cli.py --user-id YOUR_UUID
```

## Testing

Use the short test papers for development:

```bash
# Generate test UUID
python -c "import uuid; print(uuid.uuid4())"

# Ingest test papers (fast, 3 papers, 8 chunks)
python embeddings/batch_ingest.py --in data/test --user-id YOUR_UUID --raptor-mode clustering --clean
```

Test papers location: `data/test/`
- paper1_ml_basics.txt (1.7KB)
- paper2_deep_learning.txt (2KB)
- paper3_nlp_transformers.txt (2.5KB)

## Architecture

### RAPTOR Hierarchy

```
Level 2: [Global Overview]
         /               \
Level 1: [Cluster 1]  [Cluster 2]
        /    |    \     /    |    \
Level 0: Chunk1 Chunk2 Chunk3 Chunk4 Chunk5 Chunk6
```

### Clustering Process

1. **Level 0**: Store original document chunks
2. **K-means Clustering**: Group semantically similar chunks
3. **LLM Summarization**: Gemini 2.5 Flash generates cluster summaries
4. **Recursive**: Repeat until single summary remains

### Question Classification

- **Factual**: Specific facts, dates, definitions → Level 0 (detailed)
- **Summary**: Overview, main points → Level 1-2 (abstracts)
- **Comparison**: Multiple concepts → Hybrid retrieval
- **Analytical**: Deep analysis → Multi-level traversal

## Files

### Core Scripts

- `embeddings/batch_ingest.py` - Multi-file batch processing
- `embeddings/store_to_supabase` - Single file ingestion
- `embeddings/raptor_advanced.py` - RAPTOR clustering implementation
- `app/qa_system.py` - Question answering with intelligent routing
- `app/cli.py` - Command-line interface

### Configuration

- `.env` - Environment variables (connection, API keys)
- `supabase_schema.sql` - Database schema setup
- `requirements.txt` - Python dependencies

### Test Data

- `data/test/` - Short papers for testing (3 files, ~6KB total)
- `data/sample.txt` - Large sample (4.3MB)

## Database Schema

```sql
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    doc_id UUID UNIQUE,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB,
    user_id UUID,
    hierarchy_level INTEGER DEFAULT 0,  -- RAPTOR level
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Indexes
CREATE INDEX idx_documents_user_hierarchy ON documents(user_id, hierarchy_level);
CREATE INDEX idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);
```

## Arguments

### batch_ingest.py

```
--in PATH              Input file or directory
--user-id UUID         User UUID for RLS
--model NAME           Sentence transformer model (default: all-MiniLM-L6-v2)
--chunk SIZE           Chunk size (default: 1000)
--overlap SIZE         Chunk overlap (default: 100)
--raptor-mode MODE     clustering | batching | none
--clean                Delete existing user documents
```

## Performance

**Test Dataset (data/test):**
- Files: 3 papers
- Size: ~6KB total
- Chunks: 8
- Levels: 3 (0, 1, 2)
- Time: ~5 seconds

**Large Dataset (data/sample.txt):**
- Size: 4.3MB
- Chunks: 4915
- Levels: Multiple
- Time: ~3 minutes

## Troubleshooting

### Connection Issues

If you see DNS errors:
```
Failed to connect: could not translate host name
```

Solution: Use Transaction Pooler connection:
```env
host=aws-1-us-east-2.pooler.supabase.com
user=postgres.jitakvphoodobwvlvsqu
```

### API Key Error

```
400 API key not valid
```

1. Get valid key from: https://aistudio.google.com/app/apikey
2. Update GEMINI_API_KEY in .env
3. Verify: `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GEMINI_API_KEY'))"`

### UUID Format Error

```
invalid input syntax for type uuid
```

Generate valid UUID:
```bash
python -c "import uuid; print(uuid.uuid4())"
```

## Development

Run tests with short papers to iterate quickly:

```bash
# Clean and ingest
python embeddings/batch_ingest.py --in data/test --user-id YOUR_UUID --raptor-mode clustering --clean

# Query
python app/cli.py --user-id YOUR_UUID
```

## License

MIT
