-- ============================================================
-- RAPTOR INCREMENTAL ARCHITECTURE MIGRATION
-- Run this in Supabase SQL Editor
-- ============================================================
-- 
-- This migration creates a 2-layer architecture:
--   Layer 1: chunks (base truth, raw embeddings)
--   Layer 2: tree_nodes (hierarchical overlay)
--
-- BACKUP FIRST: Export your documents table before running!
-- ============================================================

-- Step 1: Ensure pgvector extension exists
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- LAYER 1: BASE CHUNKS TABLE (Ground Truth)
-- ============================================================

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID NOT NULL,
    user_id UUID NOT NULL,
    text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for chunks
CREATE INDEX IF NOT EXISTS idx_chunks_user_doc ON chunks(user_id, doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_deleted ON chunks(user_id, deleted) WHERE deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

COMMENT ON TABLE chunks IS 'Base layer: Raw chunk storage with embeddings. Always correct, easy to update.';

-- ============================================================
-- LAYER 2: TREE NODES TABLE (Hierarchical Overlay)
-- ============================================================

CREATE TABLE IF NOT EXISTS tree_nodes (
    node_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    level INTEGER NOT NULL,  -- 0=leaf, 1+=higher levels
    summary_text TEXT,  -- NULL for leaf nodes
    summary_embedding vector(384),  -- NULL for leaf nodes
    children_ids JSONB DEFAULT '[]'::jsonb,  -- Array of UUIDs (chunk_ids or node_ids)
    parent_id UUID,  -- NULL for root nodes
    member_count INTEGER DEFAULT 0,  -- Total chunks under this subtree
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_dirty BOOLEAN DEFAULT FALSE  -- Needs re-summarization
);

-- Indexes for tree_nodes
CREATE INDEX IF NOT EXISTS idx_tree_user_level ON tree_nodes(user_id, level);
CREATE INDEX IF NOT EXISTS idx_tree_parent ON tree_nodes(user_id, parent_id);
CREATE INDEX IF NOT EXISTS idx_tree_dirty ON tree_nodes(user_id, is_dirty) WHERE is_dirty = TRUE;
CREATE INDEX IF NOT EXISTS idx_tree_embedding ON tree_nodes USING ivfflat (summary_embedding vector_cosine_ops) WITH (lists = 50);

COMMENT ON TABLE tree_nodes IS 'Tree layer: Hierarchical structure with summaries. Incrementally updated.';

-- ============================================================
-- MAPPING TABLE: Chunk to Leaf Node
-- ============================================================

CREATE TABLE IF NOT EXISTS chunk_to_leaf (
    chunk_id UUID NOT NULL,
    leaf_node_id UUID NOT NULL,
    user_id UUID NOT NULL,
    PRIMARY KEY (chunk_id, leaf_node_id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (leaf_node_id) REFERENCES tree_nodes(node_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunk_to_leaf_leaf ON chunk_to_leaf(user_id, leaf_node_id);

COMMENT ON TABLE chunk_to_leaf IS 'Mapping: which chunks belong to which leaf nodes';

-- ============================================================
-- MIGRATION: Copy existing data from documents table
-- ============================================================

-- Migrate Level 0 chunks → chunks table
INSERT INTO chunks (chunk_id, doc_id, user_id, text, embedding, deleted, created_at, metadata)
SELECT 
    doc_id::uuid,  -- Use doc_id as chunk_id
    doc_id::uuid,  -- Use doc_id as doc_id (same value for now)
    COALESCE(user_id, '00000000-0000-0000-0000-000000000000'::uuid),  -- Handle NULL user_id
    content,
    embedding,
    FALSE,
    created_at,
    metadata
FROM documents
WHERE hierarchy_level = 0
  AND user_id IS NOT NULL  -- Skip rows with NULL user_id
ON CONFLICT (chunk_id) DO NOTHING;  -- Skip if already migrated

-- Migrate Level 1+ nodes → tree_nodes table
INSERT INTO tree_nodes (node_id, user_id, level, summary_text, summary_embedding, children_ids, parent_id, member_count, last_updated)
SELECT 
    doc_id::uuid,  -- Cast to UUID
    COALESCE(user_id, '00000000-0000-0000-0000-000000000000'::uuid),  -- Handle NULL user_id
    hierarchy_level,
    content,
    embedding,
    COALESCE(metadata->'children', '[]'::jsonb),  -- Extract children from metadata
    (metadata->>'parent_node_id')::uuid,  -- Extract parent from metadata
    COALESCE(
        jsonb_array_length(metadata->'children'),
        0
    ),
    created_at  -- Use created_at instead of updated_at
FROM documents
WHERE hierarchy_level > 0
  AND user_id IS NOT NULL  -- Skip rows with NULL user_id
ON CONFLICT (node_id) DO NOTHING;  -- Skip if already migrated

-- Build chunk_to_leaf mapping from documents.metadata
-- (Assumes documents have parent_node_id in metadata)
INSERT INTO chunk_to_leaf (chunk_id, leaf_node_id, user_id)
SELECT 
    d.doc_id::uuid,  -- Cast to UUID
    (d.metadata->>'parent_node_id')::uuid,
    COALESCE(d.user_id, '00000000-0000-0000-0000-000000000000'::uuid)  -- Handle NULL user_id
FROM documents d
WHERE d.hierarchy_level = 0 
  AND d.metadata->>'parent_node_id' IS NOT NULL
  AND d.user_id IS NOT NULL  -- Skip rows with NULL user_id
ON CONFLICT (chunk_id, leaf_node_id) DO NOTHING;

-- ============================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================

-- Enable RLS on all tables
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE tree_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunk_to_leaf ENABLE ROW LEVEL SECURITY;

-- Chunks policies
DROP POLICY IF EXISTS "Users can view own chunks" ON chunks;
CREATE POLICY "Users can view own chunks"
    ON chunks FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own chunks" ON chunks;
CREATE POLICY "Users can insert own chunks"
    ON chunks FOR INSERT
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own chunks" ON chunks;
CREATE POLICY "Users can update own chunks"
    ON chunks FOR UPDATE
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own chunks" ON chunks;
CREATE POLICY "Users can delete own chunks"
    ON chunks FOR DELETE
    USING (auth.uid() = user_id);

-- Tree nodes policies
DROP POLICY IF EXISTS "Users can view own tree nodes" ON tree_nodes;
CREATE POLICY "Users can view own tree nodes"
    ON tree_nodes FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own tree nodes" ON tree_nodes;
CREATE POLICY "Users can insert own tree nodes"
    ON tree_nodes FOR INSERT
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own tree nodes" ON tree_nodes;
CREATE POLICY "Users can update own tree nodes"
    ON tree_nodes FOR UPDATE
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own tree nodes" ON tree_nodes;
CREATE POLICY "Users can delete own tree nodes"
    ON tree_nodes FOR DELETE
    USING (auth.uid() = user_id);

-- Chunk to leaf policies
DROP POLICY IF EXISTS "Users can view own mappings" ON chunk_to_leaf;
CREATE POLICY "Users can view own mappings"
    ON chunk_to_leaf FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own mappings" ON chunk_to_leaf;
CREATE POLICY "Users can insert own mappings"
    ON chunk_to_leaf FOR INSERT
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own mappings" ON chunk_to_leaf;
CREATE POLICY "Users can delete own mappings"
    ON chunk_to_leaf FOR DELETE
    USING (auth.uid() = user_id);

-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================

-- Function: Search chunks (replaces match_documents for Level 0)
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(384),
    match_user_id UUID,
    match_count INT DEFAULT 10,
    include_deleted BOOLEAN DEFAULT FALSE
)
RETURNS TABLE (
    chunk_id UUID,
    text TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        chunks.chunk_id,
        chunks.text,
        chunks.metadata,
        1 - (chunks.embedding <=> query_embedding) AS similarity
    FROM chunks
    WHERE chunks.user_id = match_user_id
        AND (include_deleted OR chunks.deleted = FALSE)
    ORDER BY chunks.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function: Search tree nodes (for summaries)
CREATE OR REPLACE FUNCTION match_tree_nodes(
    query_embedding vector(384),
    match_user_id UUID,
    match_level INT,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    node_id UUID,
    summary_text TEXT,
    level INTEGER,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        tree_nodes.node_id,
        tree_nodes.summary_text,
        tree_nodes.level,
        1 - (tree_nodes.summary_embedding <=> query_embedding) AS similarity
    FROM tree_nodes
    WHERE tree_nodes.user_id = match_user_id
        AND tree_nodes.level = match_level
        AND tree_nodes.summary_embedding IS NOT NULL
    ORDER BY tree_nodes.summary_embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function: Get tree statistics
CREATE OR REPLACE FUNCTION get_tree_stats(input_user_id UUID)
RETURNS TABLE (
    total_chunks BIGINT,
    active_chunks BIGINT,
    deleted_chunks BIGINT,
    tree_levels BIGINT,
    nodes_per_level JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SELECT COUNT(*) FROM chunks WHERE user_id = input_user_id),
        (SELECT COUNT(*) FROM chunks WHERE user_id = input_user_id AND deleted = FALSE),
        (SELECT COUNT(*) FROM chunks WHERE user_id = input_user_id AND deleted = TRUE),
        (SELECT COUNT(DISTINCT level) FROM tree_nodes WHERE user_id = input_user_id),
        (SELECT jsonb_object_agg(level::text, node_count) 
         FROM (
             SELECT level, COUNT(*) as node_count 
             FROM tree_nodes 
             WHERE user_id = input_user_id 
             GROUP BY level
         ) subq);
END;
$$;

-- Function: Get document info (for backward compatibility)
CREATE OR REPLACE FUNCTION get_document_chunks(
    input_doc_id UUID,
    input_user_id UUID
)
RETURNS TABLE (
    chunk_id UUID,
    text TEXT,
    deleted BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        chunks.chunk_id,
        chunks.text,
        chunks.deleted,
        chunks.created_at
    FROM chunks
    WHERE chunks.doc_id = input_doc_id
        AND chunks.user_id = input_user_id
    ORDER BY chunks.created_at;
END;
$$;

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Check migration success
DO $$
DECLARE
    old_level0_count INTEGER;
    new_chunks_count INTEGER;
    old_higher_count INTEGER;
    new_nodes_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO old_level0_count FROM documents WHERE hierarchy_level = 0;
    SELECT COUNT(*) INTO new_chunks_count FROM chunks;
    SELECT COUNT(*) INTO old_higher_count FROM documents WHERE hierarchy_level > 0;
    SELECT COUNT(*) INTO new_nodes_count FROM tree_nodes;
    
    RAISE NOTICE 'Migration Summary:';
    RAISE NOTICE '  Old Level 0 chunks: %, New chunks table: %', old_level0_count, new_chunks_count;
    RAISE NOTICE '  Old Level 1+ nodes: %, New tree_nodes table: %', old_higher_count, new_nodes_count;
    
    IF new_chunks_count >= old_level0_count AND new_nodes_count >= old_higher_count THEN
        RAISE NOTICE '✅ Migration successful!';
    ELSE
        RAISE WARNING '⚠️ Migration may be incomplete. Check data manually.';
    END IF;
END $$;

-- ============================================================
-- GRANT PERMISSIONS
-- ============================================================

GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON chunks TO anon, authenticated;
GRANT ALL ON tree_nodes TO anon, authenticated;
GRANT ALL ON chunk_to_leaf TO anon, authenticated;

-- ============================================================
-- DONE!
-- ============================================================
-- Next steps:
-- 1. Run: SELECT * FROM get_tree_stats('YOUR_USER_ID'::uuid);
-- 2. Test queries with match_chunks() and match_tree_nodes()
-- 3. Update Python code to use new tables
-- 4. (Optional) Keep 'documents' table as backup for now
-- ============================================================
