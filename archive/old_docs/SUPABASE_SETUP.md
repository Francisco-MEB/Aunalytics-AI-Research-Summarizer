# ️ Supabase Setup Guide for AI Research Summarizer

##  Overview

This guide walks you through setting up Supabase (PostgreSQL + pgvector) for the QA system.

---

##  Quick Setup (5 Minutes)

### Step 1: Create Supabase Project

1. Go to [https://supabase.com](https://supabase.com)
2. Click "Start your project" (or sign in)
3. Click "New Project"
4. Fill in:
   - **Project name:** `ai-research-summarizer` (or your choice)
   - **Database password:** Choose a strong password (save it!)
   - **Region:** Choose closest to you
   - **Pricing plan:** Free tier works perfectly
5. Click "Create new project"
6. Wait 2-3 minutes for provisioning

---

### Step 2: Enable pgvector Extension

1. In your Supabase dashboard, go to **SQL Editor** (left sidebar)
2. Click **"+ New query"**
3. Paste this SQL:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify it's installed
SELECT * FROM pg_available_extensions WHERE name = 'vector';
```

4. Click **"Run"** (or press Ctrl+Enter)
5. You should see a success message

---

### Step 3: Create the Documents Table

Still in the SQL Editor, run this:

```sql
-- Create documents table with pgvector
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),  -- 384 dimensions for all-MiniLM-L6-v2
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create index for fast vector similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Verify table was created
SELECT 
    table_name, 
    column_name, 
    data_type 
FROM information_schema.columns 
WHERE table_name = 'documents'
ORDER BY ordinal_position;
```

**Expected output:** You should see 5 columns listed (doc_id, content, embedding, metadata, created_at)

---

### Step 4: Get Your Connection String

1. In Supabase dashboard, go to **Project Settings** (gear icon in sidebar)
2. Click **Database** (left menu)
3. Scroll down to **Connection string**
4. Select **Connection pooling** tab (important!)
5. Mode: **Transaction**
6. Copy the connection string (looks like):
   ```
   postgresql://postgres.xxxxx:[YOUR-PASSWORD]@aws-0-us-west-1.pooler.supabase.com:5432/postgres
   ```
7. Replace `[YOUR-PASSWORD]` with the password you set in Step 1

---

### Step 5: Configure Your .env File

1. Open `.env` file in your project root
2. Paste your connection string:

```bash
# Supabase Database Connection
DATABASE_URL=postgresql://postgres.xxxxx:YOUR_ACTUAL_PASSWORD@aws-0-us-west-1.pooler.supabase.com:5432/postgres

# Google Gemini API Key (get from https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=your_gemini_api_key_here
```

️ **Important:** Use the **Connection Pooler** URL (port 5432), not the direct connection (port 5433)

---

### Step 6: Test the Connection

Run this test script:

```bash
python test_supabase_connection.py
```

**Expected output:**
```
 Connected to Supabase successfully!
 pgvector extension is enabled
 Documents table exists
 All checks passed!
```

---

##  Verify Everything Works

### Test 1: Check Database from Supabase Dashboard

1. Go to **Table Editor** in Supabase dashboard
2. You should see the `documents` table
3. It will be empty (0 rows) - that's normal!

---

### Test 2: Upload Sample Data

```bash
# Process sample document
python embeddings/ingest.py --in data/science.1203877.docx --out data/test.jsonl

# Upload to Supabase
python upload_to_db.py --input data/test.jsonl
```

**Check in Supabase:**
1. Go to **Table Editor** → `documents`
2. You should now see rows of data
3. Click on a row to see the content and metadata

---

### Test 3: Run QA System

```bash
python qa_system.py
```

Try asking:
- "What is this research about?"
- "What are the main findings?"

---

##  Supabase-Specific Configuration

### Connection Pooler vs Direct Connection

| Connection Type | Port | When to Use |
|----------------|------|-------------|
| **Connection Pooler** | 5432 |  Use this (handles many connections) |
| Direct Connection | 5433 | Only for single long-running process |

**Always use the Pooler URL (port 5432) for this project.**

---

### Supabase Free Tier Limits

| Resource | Free Tier Limit | Enough for This Project? |
|----------|----------------|-------------------------|
| Database size | 500 MB |  Yes (can store 10,000+ documents) |
| Bandwidth | 5 GB |  Yes |
| API requests | Unlimited |  Yes |
| Rows | Unlimited |  Yes |

**Estimate:** 1 research paper (~20 pages) = ~50 chunks = ~0.5 MB
- You can store ~1,000 papers in free tier

---

##  Monitor Your Database

### Check Storage Usage

In Supabase dashboard:
1. Go to **Project Settings** → **Database**
2. See **Disk usage** graph

### View Your Data

1. Go to **Table Editor**
2. Select `documents` table
3. Browse your uploaded chunks
4. Search by content or filter by metadata

---

## ️ Common Issues & Solutions

### Issue 1: Connection Failed
```
psycopg2.OperationalError: could not connect to server
```

**Solutions:**
-  Use **Connection Pooler** URL (port 5432), not direct (5433)
-  Verify password is correct (no special characters issues)
-  Check your internet connection
-  Ensure Supabase project is active (not paused)

---

### Issue 2: pgvector Extension Error
```
ERROR: type "vector" does not exist
```

**Solution:**
```sql
-- Run in Supabase SQL Editor
CREATE EXTENSION IF NOT EXISTS vector;
```

---

### Issue 3: Permission Denied
```
ERROR: permission denied for table documents
```

**Solution:** You're using the wrong connection string. Use the one from **Database Settings**, not the API URL.

---

### Issue 4: Slow Queries
```
Queries taking > 5 seconds
```

**Solution:** Recreate the index with more lists:
```sql
-- Drop old index
DROP INDEX IF EXISTS documents_embedding_idx;

-- Create new index with more lists (for larger datasets)
CREATE INDEX documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 200);  -- Increase from 100 to 200
```

---

##  Security Best Practices

### 1. Never Commit .env File
```bash
# .env is already in .gitignore
# Double-check:
cat .gitignore | grep .env
```

### 2. Rotate Database Password
If you accidentally expose your password:
1. Go to **Project Settings** → **Database**
2. Click **"Reset database password"**
3. Update your `.env` file

### 3. Use Row Level Security (Optional for Production)

```sql
-- Enable RLS on documents table
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Create policy (example: allow all for now)
CREATE POLICY "Allow all access to documents" 
ON documents FOR ALL 
USING (true);
```

---

##  Scaling Considerations

### When to Upgrade from Free Tier

| Scenario | Recommendation |
|----------|---------------|
| < 500 MB data |  Free tier |
| 500 MB - 8 GB | Upgrade to Pro ($25/month) |
| > 8 GB | Consider dedicated hosting |
| High traffic | Pro tier for better connection pooling |

### Performance Optimization

1. **For < 10,000 chunks:**
   - Default settings work fine
   - IVFFlat index with lists=100

2. **For 10,000 - 100,000 chunks:**
   - Increase index lists to 200-500
   - Consider Pro tier for better performance

3. **For > 100,000 chunks:**
   - Use HNSW index instead of IVFFlat
   - Definitely need Pro tier

---

##  Maintenance

### Clean Up Old Data

```sql
-- View storage usage
SELECT 
    pg_size_pretty(pg_total_relation_size('documents')) as total_size,
    count(*) as row_count 
FROM documents;

-- Delete old documents (if needed)
DELETE FROM documents 
WHERE created_at < NOW() - INTERVAL '30 days';

-- Vacuum to reclaim space
VACUUM FULL documents;
```

### Backup Your Data

Supabase automatically backs up your database, but you can export manually:

1. Go to **Database** → **Backups**
2. Click **"Download backup"**
3. Or use `pg_dump`:

```bash
# Export documents table
pg_dump "your_connection_string" -t documents > backup.sql
```

---

##  Getting Help

### Supabase Resources
-  [Docs](https://supabase.com/docs)
-  [Discord Community](https://discord.supabase.com)
-  [GitHub Issues](https://github.com/supabase/supabase)

### pgvector Resources
-  [pgvector Docs](https://github.com/pgvector/pgvector)
-  [Performance Guide](https://github.com/pgvector/pgvector#performance)

---

##  Setup Checklist

Use this to verify your setup:

- [ ] Supabase project created
- [ ] pgvector extension enabled
- [ ] Documents table created
- [ ] Vector index created
- [ ] Connection string copied (with Pooler, port 5432)
- [ ] .env file configured
- [ ] Connection test passed
- [ ] Sample data uploaded successfully
- [ ] QA system can query database
- [ ] Questions return relevant results

---

##  You're Ready!

Once all checks pass, your Supabase database is fully configured and ready to use with the AI Research Summarizer!

**Next Steps:**
1. Upload your research papers
2. Start asking questions
3. Monitor performance in Supabase dashboard

---

**Need Help?** Run `python test_supabase_connection.py` to diagnose issues.
