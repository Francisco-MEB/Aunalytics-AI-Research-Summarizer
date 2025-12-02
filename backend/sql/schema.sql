-- VECTOR search RPC function
create or replace function match_chunks(
    query_embedding vector(384),
    match_count int,
    filter_user_id text
)
returns table (
    doc_id text,
    content text,
    metadata jsonb,
    similarity float
)
language sql stable as $$
    select
        doc_id,
        content,
        metadata,
        1 - (embedding <=> query_embedding) as similarity
    from documents
    where user_id = filter_user_id
    order by embedding <=> query_embedding
    limit match_count;
$$;
