# Supabase Setup for Document Upload & Retrieval

## Overview
This guide will help you set up your Supabase database to work with the document upload and query system.

## What Was Fixed

The following issues were resolved:

1. **RPC Function Field Mismatch**: Updated `match_chunks` function to return `content` instead of `text` and `doc_id` instead of `chunk_id`
2. **Query Parameter Alignment**: Fixed query endpoint to pass correct parameters to RPC function
3. **Chunk ID Tracking**: Added `chunk_id` field to uploaded documents for better tracking

## Database Schema

Your Supabase database needs a `documents` table with the following structure:

```sql
-- Create the documents table with pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chunk_id TEXT,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),  -- 384 dimensions for all-MiniLM-L6-v2
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);
```

## RPC Function for Cosine Similarity Search

Run this SQL in your Supabase SQL Editor to create/update the RPC function:

```sql
-- VECTOR search RPC function
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(384),
    match_count int,
    filter_user_id text
)
RETURNS TABLE (
    doc_id text,
    content text,
    metadata jsonb,
    similarity float
)
LANGUAGE sql STABLE AS $$
    SELECT
        doc_id,
        content,
        metadata,
        1 - (embedding <=> query_embedding) as similarity
    FROM documents
    WHERE user_id = filter_user_id
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
```

## Row Level Security (RLS)

To enable user-specific document isolation, set up RLS policies:

```sql
-- Enable Row Level Security
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own documents
CREATE POLICY "Users can view their own documents"
    ON documents FOR SELECT
    USING (auth.uid()::text = user_id);

-- Policy: Users can insert their own documents
CREATE POLICY "Users can insert their own documents"
    ON documents FOR INSERT
    WITH CHECK (auth.uid()::text = user_id);

-- Policy: Users can delete their own documents
CREATE POLICY "Users can delete their own documents"
    ON documents FOR DELETE
    USING (auth.uid()::text = user_id);
```

**Note**: If you're testing without authentication, you can disable RLS temporarily:
```sql
ALTER TABLE documents DISABLE ROW LEVEL SECURITY;
```

## Testing the System

### 1. Test Document Upload

Using curl or Postman, test the upload endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/ingest/" \
  -F "file=@your_document.pdf" \
  -F "user_id=test-user-123" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=100"
```

Expected response:
```json
{
  "status": "success",
  "message": "Successfully processed 15 chunks from 'your_document.pdf'",
  "num_chunks": 15,
  "filename": "your_document.pdf",
  "user_id": "test-user-123"
}
```

### 2. Verify Data in Supabase

Check your Supabase dashboard:
1. Go to Table Editor → documents
2. You should see rows with:
   - `doc_id`: UUID for each chunk
   - `chunk_id`: Filename + chunk number
   - `user_id`: Your test user ID
   - `content`: Text content of the chunk
   - `embedding`: Array of 384 floats
   - `metadata`: JSON with source_file, chunk_index, chunk_id

### 3. Test Query/Retrieval

Query the uploaded documents:

```bash
curl -X POST "http://127.0.0.1:8000/query/" \
  -F "message=What is this document about?" \
  -F "user_id=test-user-123" \
  -F "top_k=4"
```

Expected response:
```json
{
  "response": "Based on the research documents, ...",
  "num_sources": 4,
  "user_id": "test-user-123"
}
```

### 4. Check Server Logs

Monitor the terminal where uvicorn is running. You should see:
- `Retrieved X chunks via RPC` (if RPC works)
- OR `Using fallback retrieval...` (if RPC fails, fallback to Python)
- Similarity scores for retrieved chunks

## Troubleshooting

### "No documents found"

**Possible causes:**
1. No documents uploaded yet → Upload a document first
2. Wrong `user_id` → Use the same user_id for upload and query
3. RLS enabled but not authenticated → Disable RLS for testing or use proper auth

### "RPC retrieval failed"

**Possible causes:**
1. RPC function not created → Run the SQL from above
2. Wrong function signature → Drop and recreate the function
3. Vector extension not enabled → Run `CREATE EXTENSION vector;`

**Solution**: The system has a fallback that computes similarity in Python, so queries will still work (but slower).

### "Embedding dimension mismatch"

**Possible causes:**
1. Table created with wrong vector size → Should be `vector(384)` for all-MiniLM-L6-v2

**Solution**: 
```sql
-- Check current dimension
SELECT vector_dims(embedding) FROM documents LIMIT 1;

-- If wrong, recreate the column
ALTER TABLE documents DROP COLUMN embedding;
ALTER TABLE documents ADD COLUMN embedding vector(384);
```

## Environment Variables

Make sure your `.env` file has:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_KEY=your-service-role-key  # Optional, for admin operations
GEMINI_API_KEY=your-gemini-api-key
```

## API Documentation

With the server running, visit:
- Interactive API docs: http://127.0.0.1:8000/docs
- Alternative docs: http://127.0.0.1:8000/redoc

## Next Steps

1. Run the SQL commands above in Supabase
2. Upload a test document via the API
3. Query the document to verify retrieval works
4. Check the server logs for any errors
5. Integrate with your frontend

## Need Help?

If you're still getting "can't find documents" errors:
1. Check Supabase Table Editor to verify data was inserted
2. Look at server logs for detailed error messages
3. Try the fallback retrieval (it should work even if RPC fails)
4. Verify your `user_id` matches between upload and query
