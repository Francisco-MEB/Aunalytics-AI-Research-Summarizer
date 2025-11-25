-- ============================================================
-- FIX DOC_IDS: Group chunks by source_file (OPTIMIZED)
-- ============================================================
-- This script fixes the issue where each chunk has a different doc_id
-- Instead, all chunks from the same source file should share the same doc_id
-- ============================================================

-- Step 1: Add index on metadata for faster lookups (if not exists)
CREATE INDEX IF NOT EXISTS idx_chunks_metadata_source 
ON chunks ((metadata->>'source_file'));

-- Step 2: Create mapping table with consistent doc_ids per source_file
-- Use first chunk's ID as the doc_id (using DISTINCT ON to pick one)
CREATE TEMP TABLE doc_id_mapping AS
SELECT DISTINCT ON (metadata->>'source_file', user_id)
    metadata->>'source_file' AS source_file,
    user_id,
    chunk_id AS new_doc_id  -- Use first chunk's ID as doc_id
FROM chunks
WHERE metadata->>'source_file' IS NOT NULL
  AND deleted = FALSE
  AND user_id IS NOT NULL
ORDER BY metadata->>'source_file', user_id, chunk_id;

-- Step 3: Create index on temp table for faster joins
CREATE INDEX ON doc_id_mapping(source_file, user_id);

-- Step 4: Update chunks in smaller batches (by user_id to reduce lock contention)
-- Get your user_id from the chunks
DO $$
DECLARE
    target_user_id UUID;
BEGIN
    -- Update for each user separately to avoid timeouts
    FOR target_user_id IN 
        SELECT DISTINCT user_id FROM chunks WHERE user_id IS NOT NULL
    LOOP
        UPDATE chunks c
        SET doc_id = m.new_doc_id
        FROM doc_id_mapping m
        WHERE c.metadata->>'source_file' = m.source_file
          AND c.user_id = m.user_id
          AND c.user_id = target_user_id
          AND c.deleted = FALSE
          AND c.doc_id != m.new_doc_id;  -- Only update if different
        
        RAISE NOTICE 'Updated chunks for user: %', target_user_id;
    END LOOP;
END $$;

-- Step 5: Verify the fix
SELECT 
    metadata->>'source_file' AS source_file,
    doc_id,
    user_id,
    COUNT(*) AS chunk_count
FROM chunks
WHERE deleted = FALSE
  AND metadata->>'source_file' IS NOT NULL
GROUP BY metadata->>'source_file', doc_id, user_id
ORDER BY source_file, chunk_count DESC
LIMIT 20;

-- Expected result: Each source_file should have exactly ONE doc_id per user
