-- FORCEFUL FIX - This will definitely work

-- Step 1: Delete all data
TRUNCATE TABLE documents;

-- Step 2: Drop ALL constraints and indexes
DROP INDEX IF EXISTS idx_documents_embedding;
DROP INDEX IF EXISTS idx_documents_user_id;

-- Step 3: Forcefully drop and recreate the embedding column
ALTER TABLE documents DROP COLUMN IF EXISTS embedding CASCADE;
ALTER TABLE documents ADD COLUMN embedding vector(384);

-- Step 4: Ensure pgvector extension is enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 5: Create index for vector similarity search
CREATE INDEX idx_documents_embedding ON documents 
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Step 6: Create other useful indexes
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id);

-- Step 7: Drop ALL existing match_chunks functions
DROP FUNCTION IF EXISTS match_chunks CASCADE;

-- Step 8: Create the correct RPC function
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

-- Step 9: Grant all necessary permissions
GRANT ALL ON TABLE documents TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION match_chunks TO anon, authenticated, service_role;

-- Step 10: Verify the schema
SELECT 
    column_name, 
    data_type, 
    udt_name,
    CASE 
        WHEN data_type = 'USER-DEFINED' AND udt_name = 'vector' THEN '✅ CORRECT'
        ELSE '❌ WRONG'
    END as status
FROM information_schema.columns
WHERE table_name = 'documents' 
  AND column_name = 'embedding';
