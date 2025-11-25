-- Drop the old 'documents' table (no longer used after migration to chunks/tree_nodes/chunk_to_leaf)
-- Run this in Supabase SQL Editor ONLY after confirming:
-- 1. Your new schema (chunks, tree_nodes, chunk_to_leaf) is working
-- 2. You have a backup of any important data
-- 3. No other code references the documents table

-- Check if anything references documents table (should be empty)
SELECT
    tc.table_name, 
    kcu.column_name,
    ccu.table_name AS foreign_table_name
FROM information_schema.table_constraints AS tc 
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY' 
    AND ccu.table_name = 'documents';

-- If above returns no rows, safe to drop:
-- DROP TABLE IF EXISTS documents CASCADE;

-- UNCOMMENT the line above after verifying no foreign key references
