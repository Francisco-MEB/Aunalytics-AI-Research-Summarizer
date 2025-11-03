-- ============================================
-- Supabase Setup Script for AI Research Summarizer
-- Run this in Supabase SQL Editor
-- ============================================

-- Step 1: Enable pgvector extension
-- This adds vector similarity search capabilities to PostgreSQL
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify extension is installed
SELECT 
    extname AS "Extension Name",
    extversion AS "Version"
FROM pg_extension 
WHERE extname = 'vector';

-- Expected output: One row showing "vector" and version number


-- ============================================
-- Step 2: Create documents table
-- ============================================

-- Drop table if it exists (CAREFUL: This deletes all data!)
-- Uncomment the next line only if you want to start fresh
-- DROP TABLE IF EXISTS documents;

CREATE TABLE IF NOT EXISTS documents (
    -- Primary key: unique identifier for each text chunk
    doc_id TEXT PRIMARY KEY,
    
    -- The actual text content of the chunk
    content TEXT NOT NULL,
    
    -- Vector embedding (384 dimensions for all-MiniLM-L6-v2 model)
    embedding vector(384),
    
    -- Additional metadata (filename, chunk number, etc.) stored as JSON
    metadata JSONB,
    
    -- Timestamp when record was created
    created_at TIMESTAMP DEFAULT NOW()
);

-- Verify table was created
SELECT 
    table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns 
WHERE table_name = 'documents'
ORDER BY ordinal_position;

-- Expected output: 5 rows (doc_id, content, embedding, metadata, created_at)


-- ============================================
-- Step 3: Create vector similarity index
-- ============================================

-- This index makes similarity searches FAST
-- Uses IVFFlat algorithm with cosine distance
-- lists=100 is good for up to ~10,000 documents
-- Increase to 200-500 for larger datasets

CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Verify index was created
SELECT 
    indexname AS "Index Name",
    indexdef AS "Index Definition"
FROM pg_indexes 
WHERE tablename = 'documents';

-- Expected output: At least one index on the embedding column


-- ============================================
-- Step 4: Check database statistics
-- ============================================

-- View current table size and row count
SELECT 
    pg_size_pretty(pg_total_relation_size('documents')) AS "Total Size",
    pg_size_pretty(pg_relation_size('documents')) AS "Table Size",
    pg_size_pretty(pg_total_relation_size('documents') - pg_relation_size('documents')) AS "Index Size",
    (SELECT COUNT(*) FROM documents) AS "Row Count";

-- Expected output: All sizes should be small initially (0 rows)


-- ============================================
-- Step 5: Test vector operations
-- ============================================

-- This tests that pgvector is working correctly
-- Creates a test vector and performs similarity search

-- Insert a test record
INSERT INTO documents (doc_id, content, embedding, metadata)
VALUES (
    'test-doc-1',
    'This is a test document for verifying pgvector functionality.',
    -- Random 384-dimensional vector
    array_fill(0.5, ARRAY[384])::vector,
    '{"source": "test", "test": true}'::jsonb
)
ON CONFLICT (doc_id) DO NOTHING;

-- Test similarity query
-- Creates a query vector and finds the most similar document
SELECT 
    doc_id,
    content,
    embedding <=> array_fill(0.5, ARRAY[384])::vector AS cosine_distance
FROM documents
ORDER BY embedding <=> array_fill(0.5, ARRAY[384])::vector
LIMIT 5;

-- Expected output: Should show test document with small distance value

-- Clean up test data (optional)
-- DELETE FROM documents WHERE doc_id = 'test-doc-1';


-- ============================================
-- Step 6: Set up helpful views (optional)
-- ============================================

-- View to see documents with readable metadata
CREATE OR REPLACE VIEW documents_readable AS
SELECT 
    doc_id,
    LEFT(content, 100) || '...' AS content_preview,
    metadata->>'filename' AS filename,
    metadata->>'chunk_number' AS chunk_number,
    created_at
FROM documents
ORDER BY created_at DESC;

-- Test the view
SELECT * FROM documents_readable LIMIT 5;


-- ============================================
-- Step 7: Grant permissions (if needed)
-- ============================================

-- Usually not needed for Supabase, but included for completeness

-- Grant access to the postgres role (default Supabase user)
GRANT ALL PRIVILEGES ON TABLE documents TO postgres;

-- If using Row Level Security (RLS), you might need:
-- ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Allow all access" ON documents FOR ALL USING (true);


-- ============================================
-- Useful maintenance queries
-- ============================================

-- Count documents by filename
SELECT 
    metadata->>'filename' AS filename,
    COUNT(*) AS chunk_count
FROM documents
GROUP BY metadata->>'filename'
ORDER BY chunk_count DESC;

-- Find documents created today
SELECT 
    doc_id,
    LEFT(content, 100) AS preview,
    created_at
FROM documents
WHERE created_at >= CURRENT_DATE
ORDER BY created_at DESC;

-- Check for duplicate doc_ids
SELECT 
    doc_id, 
    COUNT(*) AS duplicates
FROM documents
GROUP BY doc_id
HAVING COUNT(*) > 1;

-- Calculate average embedding dimensions (should be 384)
SELECT 
    AVG(vector_dims(embedding)) AS avg_dimensions,
    MIN(vector_dims(embedding)) AS min_dimensions,
    MAX(vector_dims(embedding)) AS max_dimensions
FROM documents;


-- ============================================
-- Performance tuning queries
-- ============================================

-- For datasets > 10,000 documents, recreate index with more lists
-- DROP INDEX IF EXISTS documents_embedding_idx;
-- CREATE INDEX documents_embedding_idx 
-- ON documents 
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 200);

-- For very large datasets (> 100,000 documents), consider HNSW index
-- Requires pgvector 0.5.0+
-- CREATE INDEX documents_embedding_hnsw_idx 
-- ON documents 
-- USING hnsw (embedding vector_cosine_ops)
-- WITH (m = 16, ef_construction = 64);


-- ============================================
-- Cleanup queries (use with caution!)
-- ============================================

-- Delete all documents (keeps table structure)
-- TRUNCATE documents;

-- Delete documents older than 30 days
-- DELETE FROM documents WHERE created_at < NOW() - INTERVAL '30 days';

-- Vacuum to reclaim space after deletions
-- VACUUM FULL documents;


-- ============================================
-- ✅ Setup Complete!
-- ============================================
-- 
-- Next steps:
-- 1. Copy your DATABASE_URL from Supabase Settings → Database
-- 2. Add it to your .env file
-- 3. Run: python test_supabase_connection.py
-- 4. Upload data: python upload_to_db.py
-- 5. Ask questions: python qa_system.py
-- ============================================
