#  Clean Workspace Structure

##  Active Files (Your Working System)

###  Root Directory
```
auth.py                      # User login/registration
raptor_qa_incremental.py     # Main QA system
add_document.py              # Add documents with auth
delete_document.py           # Delete documents with auth
README.md                    # Main documentation
AUTH_README.md               # Authentication guide
QUICK_START.md               # Quick start guide
requirements.txt             # Python dependencies
```

###  embeddings/
```
incremental_raptor.py        # Core incremental RAPTOR builder
batch_ingest.py              # Document reading & chunking utilities
```

###  scripts/
```
setup_supabase.sql           # Database setup reference
database_connection          # Connection utility
```

###  data/
```
sample.txt                   # Sample document
science.1203877.docx         # Research paper
w27392.pdf                   # Research paper
*.jsonl                      # Embedding outputs
```

###  Config Files (Hidden)
```
.env                         # API keys & database config
.env.example                 # Template for .env
.gitignore                   # Git ignore rules
.auth_users.json             # User database (ignored by git)
.session                     # Current login session (ignored by git)
```

###  SQL Files (One-Time Use)
```
supabase_schema.sql          # Schema reference
migrate_user_id.sql          # Migrate existing data to new user
drop_old_documents_table.sql # Clean up old table
```

##  archive/ (Old Files - Can Delete After Testing)
```
archive/old_qa_systems/      # Old QA system versions
archive/old_embeddings/      # Old embedding scripts
archive/test_files/          # Test scripts
archive/one_time_scripts/    # Diagnostic & migration scripts
archive/sql_migrations/      # Completed SQL migrations
archive/old_docs/            # Outdated documentation
```

##  What's Next

1. **Run migrate_user_id.sql** in Supabase to link your existing documents
2. **Test the system**: `python raptor_qa_incremental.py`
3. **Add a document**: `python add_document.py data/sample.txt`
4. **After 1 week**: Delete the `archive/` folder if everything works

##  Before vs After

**Before**: 60+ files across multiple directories
**After**: ~15 active files + organized archive

Your system is now clean and ready to use! 
