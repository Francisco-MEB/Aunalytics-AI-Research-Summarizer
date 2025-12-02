-- COMPLETE CLEANUP - Run this to fix everything

-- Step 1: Delete ALL old documents with wrong dimensions
DELETE FROM documents WHERE true;

-- Step 2: Drop the old RPC function (all versions)
DROP FUNCTION IF EXISTS match_chunks(vector, int, text, bool);
DROP FUNCTION IF EXISTS match_chunks(vector, int, text);
DROP FUNCTION IF EXISTS match_chunks;

-- Step 3: Recreate embedding column with correct dimensions
ALTER TABLE documents DROP COLUMN IF EXISTS embedding;
ALTER TABLE documents ADD COLUMN embedding vector(384);

-- Step 4: Recreate index
DROP INDEX IF EXISTS idx_documents_embedding;
CREATE INDEX idx_documents_embedding ON documents 
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Step 5: Create the new RPC function
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

-- Step 6: Grant permissions
GRANT EXECUTE ON FUNCTION match_chunks TO anon, authenticated, service_role;

-- Verify it worked
SELECT 'Setup complete! Table is empty and ready for uploads.' as status;
