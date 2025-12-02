/*
SQL commands to fix Supabase setup

IMPORTANT: The database currently has embeddings with 4703 dimensions from a different model.
You have two options:

OPTION 1: Update the vector column to use 384 dimensions (RECOMMENDED)
This will delete existing data but ensure compatibility with all-MiniLM-L6-v2 model.

OPTION 2: Change the embedding model to match 4703 dimensions
You'll need to use a different embedding model in the code.

==============================================================================
OPTION 1: Fix vector dimensions (384 for all-MiniLM-L6-v2)
==============================================================================
*/

-- Step 1: Drop existing RPC function
DROP FUNCTION IF EXISTS match_chunks;

-- Step 2: Backup and clear old data (optional - skip if you want to keep data)
-- DELETE FROM documents WHERE true;

-- Step 3: Drop and recreate embedding column with correct dimensions
ALTER TABLE documents DROP COLUMN IF EXISTS embedding;
ALTER TABLE documents ADD COLUMN embedding vector(384);

-- Step 4: Recreate index for vector search
DROP INDEX IF EXISTS idx_documents_embedding;
CREATE INDEX idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);

-- Step 5: Create the correct RPC function
-- Note: The function parameters must be in THIS EXACT ORDER for PostgREST
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
GRANT EXECUTE ON FUNCTION match_chunks TO anon, authenticated;

/*
==============================================================================
After running the SQL above:
==============================================================================

1. Re-upload your test document using curl:
   curl -X POST "http://127.0.0.1:8000/ingest/" -F "file=@test_document.txt" -F "user_id=test-user-123"

2. Verify with the test script:
   python test_supabase.py

3. Test querying:
   curl -X POST "http://127.0.0.1:8000/query/" -F "message=What are the main applications of AI?" -F "user_id=test-user-123"
*/
