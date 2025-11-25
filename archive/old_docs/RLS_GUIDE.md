# Row Level Security (RLS) Guide

##  What is Row Level Security?

Row Level Security (RLS) is a PostgreSQL feature that allows you to restrict which rows users can access in a table. With RLS enabled, users can only see/modify data that belongs to them.

##  Why Use RLS?

- **Multi-tenant security**: Prevent users from accessing each other's documents
- **Automatic enforcement**: Database-level protection (can't be bypassed)
- **Fine-grained control**: Different policies for SELECT, INSERT, UPDATE, DELETE

##  RLS Implementation in This Project

### 1. Database Schema
The `documents` table includes a `user_id` column:
```sql
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source_file TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(384) NOT NULL,
    user_id UUID,  -- Links document to a specific user
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 2. RLS Policies
Four policies control access:

- **SELECT**: Users can only view their own documents
- **INSERT**: Users can only insert documents with their user_id
- **UPDATE**: Users can only update their own documents
- **DELETE**: Users can only delete their own documents

### 3. Supabase Authentication
RLS uses `auth.uid()` to get the current user's ID from Supabase Auth.

##  Usage Examples

### Enable RLS (One-time setup)
```powershell
python embeddings/store_to_supabase --in data/sample.txt --enable-rls
```

### Store Documents with User ID
```powershell
# Store document for a specific user
python embeddings/store_to_supabase --in data/research.pdf --user-id "123e4567-e89b-12d3-a456-426614174000"
```

### Store Documents without RLS (Testing/Development)
```powershell
# No user_id = accessible to all authenticated users
python embeddings/store_to_supabase --in data/sample.txt
```

##  Getting User IDs

In a production Supabase app:
1. Users sign up/login via Supabase Auth
2. `auth.uid()` automatically returns their UUID
3. Frontend passes this to your backend/scripts

For testing:
```sql
-- Generate a test UUID
SELECT gen_random_uuid();
-- Example: 123e4567-e89b-12d3-a456-426614174000
```

## ️ RLS Configuration

### Check if RLS is Enabled
```sql
SELECT tablename, rowsecurity 
FROM pg_tables 
WHERE tablename = 'documents';
```

### View Current Policies
```sql
SELECT * FROM pg_policies WHERE tablename = 'documents';
```

### Disable RLS (Not Recommended)
```sql
ALTER TABLE documents DISABLE ROW LEVEL SECURITY;
```

##  Testing RLS

### Test 1: Insert with User ID
```powershell
python embeddings/store_to_supabase --in test.txt --user-id "user-123"
```

### Test 2: Query as Different User
```sql
-- Set session to user-123
SET LOCAL "request.jwt.claims" = '{"sub": "user-123"}';
SELECT COUNT(*) FROM documents;  -- Should see their docs

-- Set session to user-456
SET LOCAL "request.jwt.claims" = '{"sub": "user-456"}';
SELECT COUNT(*) FROM documents;  -- Should see 0 (or their docs only)
```

## ️ Security Best Practices

1. **Always Enable RLS in Production**: Prevents data leaks
2. **Use Service Role Key Carefully**: Bypasses RLS (admin access only)
3. **Test Policies Thoroughly**: Verify users can't access others' data
4. **Log Access Attempts**: Monitor for unauthorized access patterns

##  Common Issues

### Issue: "new row violates row-level security policy"
**Cause**: Trying to insert with wrong user_id  
**Solution**: Ensure `--user-id` matches authenticated user

### Issue: "permission denied for table documents"
**Cause**: RLS enabled but no matching policy  
**Solution**: Run with `--enable-rls` to create policies

### Issue: Can't see any documents after enabling RLS
**Cause**: No user_id set during insertion  
**Solution**: Re-insert with `--user-id` parameter

##  Learn More

- [Supabase RLS Documentation](https://supabase.com/docs/guides/auth/row-level-security)
- [PostgreSQL RLS Official Docs](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
