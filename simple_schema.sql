-- Simple Pre-RAPTOR Database Schema
-- Just documents table with vector similarity search

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop old RAPTOR tables if they exist
DROP TABLE IF EXISTS chunk_to_leaf CASCADE;
DROP TABLE IF EXISTS tree_nodes CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;

-- Add missing columns to documents table if they don't exist
DO $$ 
BEGIN
    -- Add user_id column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='documents' AND column_name='user_id') THEN
        ALTER TABLE documents ADD COLUMN user_id UUID;
    END IF;
    
    -- Add source column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='documents' AND column_name='source') THEN
        ALTER TABLE documents ADD COLUMN source TEXT;
    END IF;
END $$;

-- Create index for fast vector similarity search
DROP INDEX IF EXISTS documents_embedding_idx;
CREATE INDEX documents_embedding_idx
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for user_id lookups
CREATE INDEX IF NOT EXISTS documents_user_id_idx ON documents(user_id);

-- Enable Row Level Security
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can only see their own documents
DROP POLICY IF EXISTS "Users can view own documents" ON documents;
CREATE POLICY "Users can view own documents"
ON documents FOR SELECT
USING (user_id = current_setting('app.user_id', true)::UUID OR user_id IS NULL);

-- RLS Policy: Users can insert their own documents
DROP POLICY IF EXISTS "Users can insert own documents" ON documents;
CREATE POLICY "Users can insert own documents"
ON documents FOR INSERT
WITH CHECK (user_id = current_setting('app.user_id', true)::UUID OR user_id IS NULL);

-- RLS Policy: Users can delete their own documents
DROP POLICY IF EXISTS "Users can delete own documents" ON documents;
CREATE POLICY "Users can delete own documents"
ON documents FOR DELETE
USING (user_id = current_setting('app.user_id', true)::UUID OR user_id IS NULL);

-- Verify setup
SELECT 
    'Table created' as status,
    COUNT(*) as row_count 
FROM documents;

SELECT 
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename = 'documents';
