# ️ System Architecture with Supabase

## Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                   AI Research Summarizer System                   │
│                                                                   │
│  ┌────────────┐    ┌─────────────┐    ┌────────────────────┐   │
│  │   PDF/     │───▶│  ingest.py  │───▶│   embeddings.jsonl │   │
│  │   DOCX     │    │  (LangChain │    │   (384-dim vectors)│   │
│  │   TXT      │    │  + Sentence │    └────────────────────┘   │
│  │   Files    │    │  Transformers)   │                          │
│  └────────────┘    └─────────────┘    │                          │
│                                        ▼                          │
│                            ┌───────────────────────┐             │
│                            │   upload_to_db.py     │             │
│                            │   (psycopg connector) │             │
│                            └───────────────────────┘             │
│                                        │                          │
│                                        ▼                          │
│                    ┌───────────────────────────────┐             │
│                    │      SUPABASE DATABASE        │             │
│                    │  (PostgreSQL + pgvector)      │             │
│                    │                               │             │
│                    │  ┌─────────────────────────┐ │             │
│                    │  │  documents table        │ │             │
│                    │  │  ├─ doc_id (PK)        │ │             │
│                    │  │  ├─ content (TEXT)     │ │             │
│                    │  │  ├─ embedding (vector) │ │             │
│                    │  │  ├─ metadata (JSONB)   │ │             │
│                    │  │  └─ created_at         │ │             │
│                    │  └─────────────────────────┘ │             │
│                    │                               │             │
│                    │  Index: IVFFlat (cosine)     │             │
│                    └───────────────────────────────┘             │
│                                   ▲                               │
│                                   │                               │
│                          ┌────────┴────────┐                     │
│                          │   qa_system.py  │                     │
│                          │   (RAG Pipeline)│                     │
│                          └────────┬────────┘                     │
│                                   │                               │
│                          ┌────────▼────────┐                     │
│                          │  Google Gemini  │                     │
│                          │  (Answer Gen)   │                     │
│                          └─────────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Document Ingestion

```
1️⃣ USER UPLOADS PDF
   └─▶ research_paper.pdf

2️⃣ INGEST.PY PROCESSES
   ├─▶ Extract text (pypdf/docx2txt)
   ├─▶ Split into chunks (LangChain RecursiveTextSplitter)
   │   └─▶ 1000 chars per chunk, 100 char overlap
   ├─▶ Generate embeddings (sentence-transformers)
   │   └─▶ all-MiniLM-L6-v2 model → 384-dim vectors
   └─▶ Save to JSONL
       └─▶ embeddings.jsonl

3️⃣ UPLOAD_TO_DB.PY STORES IN SUPABASE
   ├─▶ Read JSONL file
   ├─▶ Connect to Supabase (psycopg)
   ├─▶ Create table if not exists
   ├─▶ Insert/upsert records with vectors
   └─▶ Build IVFFlat index for fast search
```

---

## Query Flow: Answering Questions

```
1️⃣ USER ASKS QUESTION
   └─▶ "What are the main findings?"

2️⃣ QA_SYSTEM.PY RETRIEVES CONTEXT
   ├─▶ Convert question to embedding (384-dim)
   ├─▶ Query Supabase with vector similarity
   │   └─▶ SELECT * ORDER BY embedding <=> query_vector
   │   └─▶ Retrieve top 5 most similar chunks
   └─▶ Get relevant text passages

3️⃣ GENERATE ANSWER WITH GEMINI
   ├─▶ Build prompt with context + question
   ├─▶ Send to Google Gemini 2.5 Flash
   └─▶ Return grounded answer

4️⃣ USER RECEIVES ANSWER
   └─▶ "The main findings are..."
```

---

## Supabase Components

### 1. PostgreSQL Database
- **Type:** Relational database with vector support
- **Version:** PostgreSQL 15+
- **Extension:** pgvector 0.5.0+
- **Free Tier:** 500 MB storage (enough for ~1,000 papers)

### 2. Connection Pooler
- **Port:** 5432 (use this!)
- **Type:** Transaction pooling
- **Purpose:** Handles multiple concurrent connections efficiently
- **Why:** Prevents "too many connections" errors

### 3. pgvector Extension
- **Purpose:** Adds vector data type and similarity search
- **Operations:**
  - Cosine distance: `<=>` operator
  - L2 distance: `<->` operator
  - Inner product: `<#>` operator
- **Indexes:**
  - IVFFlat: Good for most use cases (100-10K docs)
  - HNSW: Better for large datasets (100K+ docs)

### 4. Vector Index (IVFFlat)
- **Algorithm:** Inverted File with Flat compression
- **Lists:** 100 (default) - increase for larger datasets
- **Distance Metric:** Cosine similarity
- **Speed:** ~10ms for 10K documents
- **Accuracy:** ~98% recall @ top-5

---

## Key Configuration Files

### `.env` - Environment Variables
```bash
# Supabase connection string (Connection Pooler, port 5432)
DATABASE_URL=postgresql://postgres.xxx:password@host.pooler.supabase.com:5432/postgres

# Google Gemini API key
GEMINI_API_KEY=your_key_here
```

### `setup_supabase.sql` - Database Schema
```sql
-- Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create index
CREATE INDEX documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

---

## Performance Characteristics

### Vector Search Speed
| Dataset Size | Search Time | Index Type | Lists |
|--------------|-------------|------------|-------|
| 1K documents | ~5ms | IVFFlat | 100 |
| 10K documents | ~15ms | IVFFlat | 100 |
| 100K documents | ~50ms | IVFFlat | 200 |
| 1M documents | ~100ms | HNSW | N/A |

### Storage Requirements
| Item | Size per Document | Example (20-page paper) |
|------|-------------------|-------------------------|
| Text content | ~50 KB | 1 MB |
| Embeddings (384-dim) | ~1.5 KB per chunk | ~75 KB (50 chunks) |
| Metadata | ~0.5 KB per chunk | ~25 KB |
| **Total** | - | **~1.1 MB per paper** |

**Free Tier Capacity:** ~450 research papers

---

## Supabase vs. Alternatives

| Feature | Supabase | Pinecone | Weaviate | ChromaDB |
|---------|----------|----------|----------|----------|
| **Database** | PostgreSQL | Purpose-built | Purpose-built | SQLite |
| **Hosting** | Cloud | Cloud | Cloud/Self | Self-hosted |
| **Free Tier** | 500 MB | 1 index | 512 MB RAM | Unlimited |
| **Setup Time** | 5 min | 10 min | 15 min | 2 min |
| **SQL Support** |  Yes |  No |  No | Limited |
| **Filtering** | Full SQL | Metadata | GraphQL | Basic |
| **Best For** | This project! | Production APIs | Enterprise | Prototyping |

**Why Supabase for this project:**
-  Free tier is generous
-  Full PostgreSQL features (SQL, JSONB, etc.)
-  Easy to set up (5 minutes)
-  Built-in dashboard for data inspection
-  Connection pooling handles concurrency
-  Can scale later if needed

---

## Connection Types Explained

### Connection Pooler (Port 5432)  USE THIS
```
postgresql://postgres.xxx:password@host.pooler.supabase.com:5432/postgres
```
- **Purpose:** Handle many short-lived connections
- **Mode:** Transaction pooling
- **Max connections:** Shared pool (handles 1000s)
- **Use for:** Application connections (this project)

### Direct Connection (Port 5433) ️ DON'T USE
```
postgresql://postgres.xxx:password@host.supabase.co:5433/postgres
```
- **Purpose:** Long-running connections
- **Mode:** Direct to PostgreSQL
- **Max connections:** Limited (10-20)
- **Use for:** Database migrations, pg_dump, etc.

**Important:** Always use the **Connection Pooler (5432)** for this project!

---

## Security Considerations

### Row Level Security (RLS)
- **Default:** Disabled for simplicity
- **For production:** Enable RLS to restrict access

```sql
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read access"
ON documents FOR SELECT
USING (true);
```

### API Keys
- Store in `.env` (already in `.gitignore`)
- Never commit to Git
- Rotate if accidentally exposed

### Database Password
- Use strong password (Supabase generates one)
- Don't share connection strings
- Can reset in Supabase dashboard

---

## Monitoring & Debugging

### Supabase Dashboard
1. **Table Editor:** View/edit data directly
2. **SQL Editor:** Run custom queries
3. **Database → Usage:** Monitor storage/bandwidth
4. **Logs:** View connection logs and errors

### Test Script
```bash
python test_supabase_connection.py
```

Checks:
-  Environment variables
-  Database connection
-  pgvector extension
-  Table schema
-  Vector index
-  Sample queries

---

## Troubleshooting Quick Reference

| Error | Cause | Solution |
|-------|-------|----------|
| Connection refused | Wrong port | Use port 5432 (pooler) |
| Type "vector" does not exist | Extension not enabled | Run `CREATE EXTENSION vector;` |
| Too many connections | Direct connection | Switch to pooler (5432) |
| Slow queries | Missing index | Run index creation SQL |
| Permission denied | Wrong credentials | Check DATABASE_URL |

---

## Useful SQL Queries

### Check Table Size
```sql
SELECT 
    pg_size_pretty(pg_total_relation_size('documents')) as size,
    count(*) as rows
FROM documents;
```

### View Recent Documents
```sql
SELECT doc_id, LEFT(content, 100), created_at
FROM documents
ORDER BY created_at DESC
LIMIT 10;
```

### Test Vector Search
```sql
SELECT doc_id, content
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;
```

### Count by File
```sql
SELECT 
    metadata->>'filename' as file,
    count(*) as chunks
FROM documents
GROUP BY metadata->>'filename';
```

---

## Next Steps

1. **Complete Supabase Setup**
   - Follow [`SUPABASE_SETUP.md`](SUPABASE_SETUP.md)
   - Run `setup_supabase.sql` in SQL Editor
   - Copy your DATABASE_URL

2. **Test Connection**
   - Add DATABASE_URL to `.env`
   - Run `python test_supabase_connection.py`
   - Verify all tests pass

3. **Upload Data**
   - Process documents: `python embeddings/ingest.py`
   - Upload to Supabase: `python upload_to_db.py`

4. **Start Querying**
   - Run: `python qa_system.py`
   - Ask questions about your documents

---

**Questions?** Check `SUPABASE_SETUP.md` for detailed troubleshooting!
