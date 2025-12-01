# 🎯 Quick Reference: RLS Commands

## Setup (Run Once)
```powershell
# 1. Enable RLS on the database
python embeddings/store_to_supabase --in data/sample.txt --enable-rls
```

## Store Documents
```powershell
# With RLS (user-specific)
python embeddings/store_to_supabase --in document.pdf --user-id "your-uuid-here"

# Without RLS (public/testing)
python embeddings/store_to_supabase --in document.pdf
```

## Query Documents
```powershell
# Query user-specific documents
python qa_system.py --user-id "your-uuid-here"

# Query all accessible documents
python qa_system.py
```

## Common User IDs for Testing
```
User A: aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa
User B: bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb
User C: cccccccc-cccc-cccc-cccc-cccccccccccc
```

## Check RLS Status (SQL)
```sql
-- View RLS status
SELECT tablename, rowsecurity FROM pg_tables WHERE tablename = 'documents';

-- View policies
SELECT * FROM pg_policies WHERE tablename = 'documents';

-- Count documents per user
SELECT user_id, COUNT(*) FROM documents GROUP BY user_id;
```

## Environment Variables Required
```
DATABASE_URL=postgresql://user:pass@db.supabase.co:5432/postgres
GEMINI_API_KEY=your_gemini_api_key_here
```
