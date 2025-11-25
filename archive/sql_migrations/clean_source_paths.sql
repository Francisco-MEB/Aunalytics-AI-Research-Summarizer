-- Clean up source_file metadata to show only filename (not full path)
-- Run this in Supabase SQL Editor

-- Extract just the filename from paths like "data\research_papers\w27392.pdf"
UPDATE chunks
SET metadata = jsonb_set(
    metadata, 
    '{source_file}', 
    to_jsonb(
        CASE 
            WHEN metadata->>'source_file' LIKE '%\%' THEN 
                -- Windows path: extract after last backslash
                substring(metadata->>'source_file' from '[^\\]+$')
            WHEN metadata->>'source_file' LIKE '%/%' THEN 
                -- Unix path: extract after last forward slash
                substring(metadata->>'source_file' from '[^/]+$')
            ELSE 
                -- Already just filename
                metadata->>'source_file'
        END
    )
)
WHERE user_id = '5b11bef4-7ea1-4bf9-aac1-7f22f7c73705'::uuid
  AND (metadata->>'source_file' LIKE '%\%' OR metadata->>'source_file' LIKE '%/%');

-- Verify the fix
SELECT 
    metadata->>'source_file' AS source_file,
    COUNT(*) as chunk_count
FROM chunks
WHERE user_id = '5b11bef4-7ea1-4bf9-aac1-7f22f7c73705'::uuid
  AND deleted = FALSE
GROUP BY metadata->>'source_file'
ORDER BY chunk_count DESC;
