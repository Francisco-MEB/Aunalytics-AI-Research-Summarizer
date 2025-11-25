# Files to Keep vs Delete

##  KEEP - Core System Files (Active)

### Main Application
- `auth.py` - User authentication system
- `raptor_qa_incremental.py` - Main QA system (CURRENT)
- `add_document.py` - Add documents with auth
- `delete_document.py` - Delete documents with auth

### Core Logic
- `embeddings/incremental_raptor.py` - Incremental RAPTOR builder (CURRENT)
- `embeddings/batch_ingest.py` - Document reading and chunking utilities

### Utilities
- `audit_system.py` - Database diagnostic tool

### Documentation
- `AUTH_README.md` - Auth system documentation
- `README.md` - Main documentation

## ️ KEEP - SQL Files (One-Time Use, Keep for Reference)

### Already Run (can archive)
- `migrate_to_incremental.sql` - Migration from old schema (DONE)
- `fix_doc_ids.sql` - Fixed doc grouping (DONE)
- `clean_source_paths.sql` - Cleaned filenames (DONE)

### Need to Run Once
- `migrate_user_id.sql` - Migrate your existing data to new user_id
- `drop_old_documents_table.sql` - Clean up old table (run after backup)

### Reference (keep)
- `supabase_schema.sql` - Schema reference
- `scripts/setup_supabase.sql` - Initial setup

##  DELETE - Old/Obsolete Files

### Old QA Systems (superseded)
- `raptor_qa.py` - Old version
- `raptor_qa_simplified.py` - Old version
- `qa_system.py` - Old version
- `demo_qa.py` - Old demo

### Old Batch Ingest (not needed)
- `embeddings/batch_ingest_simplified.py` - Obsolete
- `embeddings/raptor_advanced.py` - Obsolete
- `embeddings/reembed_missing.py` - One-time fix script
- `embeddings/ingest.py` - Old version

### Test Files (keep if actively testing, else delete)
- `test_simplified.py`
- `test_simple_connection.py`
- `test_qa.py`
- `test_incremental.py`
- `test_direct_connection.py`
- `test_connection.py`
- `test_comparison.py`
- `tests/test_supabase_connection.py`
- `tests/test_rls_user.py`
- `tests/test_rls.py`
- `tests/test_qa_system.py`
- `tests/test_pdf_workflow.py`

### Debug/Diagnostic (one-time use)
- `debug_env.py` - Debug script
- `diagnose_db.py` - Old diagnostic
- `check_documents_table.py` - One-time check
- `tools/backfill_file_hashes.py` - One-time migration

### Temporary Files
- `test.py` - Generic test file
- `init.txt` - Unknown

##  RECOMMENDED ACTION

Create an `archive/` folder and move obsolete files there (don't delete yet, just in case):

```powershell
mkdir archive
mkdir archive\old_qa_systems
mkdir archive\old_embeddings
mkdir archive\test_files
mkdir archive\one_time_scripts

# Move old systems
move raptor_qa.py, raptor_qa_simplified.py, qa_system.py, demo_qa.py archive\old_qa_systems\

# Move old embeddings
move embeddings\batch_ingest_simplified.py, embeddings\raptor_advanced.py, embeddings\reembed_missing.py, embeddings\ingest.py archive\old_embeddings\

# Move test files
move test_*.py archive\test_files\
move tests\* archive\test_files\

# Move one-time scripts
move debug_env.py, diagnose_db.py, check_documents_table.py, tools\backfill_file_hashes.py archive\one_time_scripts\
```

After a week of using the new system successfully, you can delete the archive folder.
