-- Migrate existing documents to new authenticated user
-- Run this in Supabase SQL Editor

-- Your old user_id: 5b11bef4-7ea1-4bf9-aac1-7f22f7c73705
-- Your new user_id: 71b3dc50-fd36-49bc-856a-24cb64387b43

-- Update chunks
UPDATE chunks 
SET user_id = '71b3dc50-fd36-49bc-856a-24cb64387b43'::uuid
WHERE user_id = '5b11bef4-7ea1-4bf9-aac1-7f22f7c73705'::uuid;

-- Update tree_nodes
UPDATE tree_nodes 
SET user_id = '71b3dc50-fd36-49bc-856a-24cb64387b43'::uuid
WHERE user_id = '5b11bef4-7ea1-4bf9-aac1-7f22f7c73705'::uuid;

-- Update chunk_to_leaf mappings
UPDATE chunk_to_leaf 
SET user_id = '71b3dc50-fd36-49bc-856a-24cb64387b43'::uuid
WHERE user_id = '5b11bef4-7ea1-4bf9-aac1-7f22f7c73705'::uuid;

-- Verify the migration
SELECT 'chunks' as table_name, COUNT(*) as count 
FROM chunks 
WHERE user_id = '71b3dc50-fd36-49bc-856a-24cb64387b43'::uuid
UNION ALL
SELECT 'tree_nodes', COUNT(*) 
FROM tree_nodes 
WHERE user_id = '71b3dc50-fd36-49bc-856a-24cb64387b43'::uuid
UNION ALL
SELECT 'chunk_to_leaf', COUNT(*) 
FROM chunk_to_leaf 
WHERE user_id = '71b3dc50-fd36-49bc-856a-24cb64387b43'::uuid;
