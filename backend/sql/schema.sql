-- VECTOR search RPC function
create or replace function match_chunks(
    query_embedding vector(384),
    match_count int,
    user_id text
)
returns table (
    chunk_id text,
    text text,
    metadata jsonb,
    similarity float
)
language sql stable as $$
    select
        chunk_id,
        text,
        metadata,
        1 - (embedding <=> query_embedding) as similarity
    from documents
    where user_id = match_chunks.user_id
    order by embedding <=> query_embedding
    limit match_count;
$$;
