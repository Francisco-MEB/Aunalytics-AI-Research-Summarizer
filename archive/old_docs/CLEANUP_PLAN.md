# Workspace Cleanup Plan

## Files to KEEP (Production Code)

### Core Application
- `raptor_qa.py` - Main QA system with hierarchy-aware retrieval 
- `qa_system.py` - Legacy/alternative QA system (review if still needed)

### Embeddings & Ingestion
- `embeddings/batch_ingest.py` - Production ingestion with RAPTOR hierarchy 
- `embeddings/raptor_advanced.py` - RAPTOR clustering implementation 
- `embeddings/ingest.py` - Single-file ingestion utility

### Diagnostics & Utilities
- `diagnose_db.py` - Database diagnostics tool 
- `tools/backfill_file_hashes.py` - Utility for adding file hashes to existing docs

### Tests
- `tests/test_qa_system.py` - Core QA tests
- `tests/test_rls.py` - Row-level security tests
- `tests/test_pdf_workflow.py` - PDF processing tests

### Scripts
- `scripts/upload_to_db.py` - Upload utility
- `scripts/check_models.py` - Model checking utility

## Files to ARCHIVE (Old/Simplified Versions)

### Simplified Implementations (Created during misunderstanding)
- `raptor_qa_simplified.py` → Archive to `archive/`
- `embeddings/batch_ingest_simplified.py` → Archive to `archive/`

### Test Files (Temporary/Debug)
- `test_simplified.py` → Archive to `archive/`
- `test_comparison.py` → Archive to `archive/`
- `test_qa.py` → Archive to `archive/` (redundant with tests/test_qa_system.py)
- `test_connection.py` → Archive to `archive/`
- `test_direct_connection.py` → Archive to `archive/`
- `test_simple_connection.py` → Archive to `archive/`
- `tests/test_supabase_connection.py` → Archive to `archive/`

### Debug Files
- `demo_qa.py` → Archive to `archive/` (if not used)
- `debug_env.py` → Archive to `archive/`
- `check_hierarchy.py` → Archive to `archive/` (functionality now in diagnose_db.py)

### Legacy/Unused
- `embeddings/reembed_missing.py` → Archive to `archive/` (one-off utility)
- `tests/test_rls_user.py` → Review and archive if redundant with test_rls.py

## Files to REVIEW

### Unclear Status
- `qa_system.py` - Is this still used? If not, archive
- `demo_qa.py` - Is this a demo script or production code?
- `embeddings/ingest.py` - Is this still needed vs. batch_ingest.py?

## Action Plan

1. Create `archive/` directory
2. Move simplified versions to archive
3. Move test files to archive
4. Update README.md with current file structure
5. Add `.gitignore` entries for common temp files

## Commands to Execute

```bash
# Create archive directory
mkdir archive
mkdir archive/simplified
mkdir archive/tests
mkdir archive/debug

# Archive simplified versions
mv raptor_qa_simplified.py archive/simplified/
mv embeddings/batch_ingest_simplified.py archive/simplified/

# Archive test files
mv test_simplified.py archive/tests/
mv test_comparison.py archive/tests/
mv test_qa.py archive/tests/
mv test_connection.py archive/tests/
mv test_direct_connection.py archive/tests/
mv test_simple_connection.py archive/tests/
mv tests/test_supabase_connection.py archive/tests/

# Archive debug files
mv debug_env.py archive/debug/
mv check_hierarchy.py archive/debug/
mv demo_qa.py archive/debug/

# Archive one-off utilities
mv embeddings/reembed_missing.py archive/debug/

# Add to .gitignore
echo "\n# Archived files\narchive/\n*.pyc\n__pycache__/\n.env\n.DS_Store" >> .gitignore
```

## Updated File Structure (After Cleanup)

```
Aunalytics-AI-Research-Summarizer/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── raptor_qa.py                    # Main QA system 
├── qa_system.py                    # [REVIEW] Legacy QA?
├── diagnose_db.py                  # DB diagnostics 
│
├── embeddings/
│   ├── batch_ingest.py             # Production ingestion 
│   ├── raptor_advanced.py          # RAPTOR implementation 
│   └── ingest.py                   # [REVIEW] Single-file utility?
│
├── tools/
│   └── backfill_file_hashes.py     # Utility
│
├── scripts/
│   ├── upload_to_db.py
│   └── check_models.py
│
├── tests/
│   ├── test_qa_system.py           # Core tests
│   ├── test_rls.py                 # Security tests
│   └── test_pdf_workflow.py        # PDF tests
│
├── data/
│   └── research_papers/            # User documents
│
├── docs/                           # Documentation
│   ├── RAPTOR_CLUSTERING_EXPLAINED.md 
│   ├── HIERARCHY_AWARE_RETRIEVAL.md 
│   ├── INCREMENTAL_UPDATES_STRATEGY.md 
│   ├── SIMPLIFICATION_GUIDE.md     # [ARCHIVE?]
│   └── QUICK_START.md              # [UPDATE?]
│
└── archive/                        # Old/deprecated code
    ├── simplified/
    ├── tests/
    └── debug/
```

## Status
- [ ] Create archive directories
- [ ] Move simplified files
- [ ] Move test files
- [ ] Move debug files
- [ ] Review qa_system.py
- [ ] Review ingest.py
- [ ] Update README.md
- [ ] Update .gitignore
- [ ] Test that production code still works
