#  AI Research Summarizer - System Documentation

##  Overview

The AI Research Summarizer is a **Retrieval-Augmented Generation (RAG)** system that helps users understand complex academic papers by:
1. Processing research documents (PDF, DOCX, TXT)
2. Creating searchable embeddings
3. Answering questions using relevant context + AI

**Think of it as:** A smart research assistant that reads papers for you and answers your questions based on what it learned.

---

## ️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERACTION                          │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: DOCUMENT INGESTION (embeddings/ingest.py)         │
│  ───────────────────────────────────────────────────────────│
│  1. Read PDF/DOCX/TXT                                       │
│  2. Split into chunks (1000 chars, 100 overlap)             │
│  3. Generate embeddings (all-MiniLM-L6-v2, 384-dim)        │
│  4. Save to JSONL file                                      │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: DATABASE UPLOAD (upload_to_db.py)                 │
│  ───────────────────────────────────────────────────────────│
│  1. Read JSONL file                                         │
│  2. Connect to PostgreSQL + pgvector                        │
│  3. Create table (if needed)                                │
│  4. Insert embeddings + text                                │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: QUESTION ANSWERING (qa_system.py)                 │
│  ───────────────────────────────────────────────────────────│
│  1. User asks question                                      │
│  2. Question → embedding                                    │
│  3. Vector similarity search (cosine)                       │
│  4. Retrieve top 4 relevant chunks                          │
│  5. Send chunks + question to Gemini                        │
│  6. Gemini generates grounded answer                        │
│  7. Return answer to user                                   │
└─────────────────────────────────────────────────────────────┘
```

---

##  File Structure & What Each Does

### **Core System Files**

#### `embeddings/ingest.py` - Document Processor
**Purpose:** Converts documents into searchable embeddings

**What it does:**
1. Reads PDF, DOCX, or TXT files
2. Splits text into overlapping chunks (preserves context)
3. Generates 384-dimensional embeddings using sentence-transformers
4. Saves to JSONL format (one embedding per line)

**Key Functions:**
- `read_document()` - Extracts text from various file formats
- `chunk_text()` - Splits text intelligently (paragraphs → sentences → words)
- `embed_chunks()` - Converts text to vector embeddings
- `write_jsonl()` - Saves embeddings to file

**Example Usage:**
```bash
python embeddings/ingest.py --in paper.pdf --out embeddings.jsonl
```

---

#### `upload_to_db.py` - Database Uploader
**Purpose:** Loads embeddings into PostgreSQL with pgvector

**What it does:**
1. Reads JSONL file created by ingest.py
2. Connects to PostgreSQL database
3. Creates `documents` table with pgvector extension
4. Uploads all embeddings + metadata
5. Creates vector index for fast similarity search

**Key Functions:**
- `create_table_if_not_exists()` - Sets up database schema
- `load_jsonl()` - Reads embedding file
- `upload_records()` - Inserts data into database

**Database Schema:**
```sql
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),  -- 384-dimensional vector
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Example Usage:**
```bash
python upload_to_db.py --input embeddings.jsonl
```

---

#### `qa_system.py` - Question Answering Engine
**Purpose:** Main system that answers user questions

**What it does:**
1. Takes user's question
2. Converts question to embedding (same model as documents)
3. Searches database for similar chunks (cosine similarity)
4. Retrieves top 4 most relevant chunks
5. Sends chunks + question to Google Gemini
6. Gemini generates contextual answer
7. Returns answer to user

**Key Classes & Methods:**
- `QASystem` - Main orchestrator
  - `retrieve_context()` - Vector similarity search
  - `generate_answer()` - Gemini API call
  - `ask_question()` - Complete workflow

**Example Usage:**
```bash
python qa_system.py
# Then type questions interactively
```

---

### **Testing & Utilities**

#### `test_qa_system.py` - Component Tests
**Purpose:** Tests individual components

**Tests:**
- Import checks (all packages installed?)
- Embedding model loading
- Gemini API connection
- QA system initialization
- Mock answer generation

**Example Usage:**
```bash
python test_qa_system.py
```

---

#### `test_pdf_workflow.py` - End-to-End Test
**Purpose:** Tests complete pipeline

**What it tests:**
1. Environment variables configured?
2. PDF ingestion works?
3. Database upload works?
4. QA system works?

**Example Usage:**
```bash
python test_pdf_workflow.py
```

---

### **Configuration Files**

#### `.env` - Environment Variables
**Purpose:** Stores sensitive credentials

**Required variables:**
```bash
DATABASE_URL=postgresql://user:pass@host:5432/dbname
GEMINI_API_KEY=your_api_key_here
```

#### `requirements.txt` - Python Dependencies
**Purpose:** Lists all required packages

**Install with:**
```bash
pip install -r requirements.txt
```

---

##  How the System Works (Deep Dive)

### **1. Embedding Generation**

**What are embeddings?**
- Embeddings are numerical representations of text (vectors)
- Similar texts have similar vectors
- Allows mathematical comparison of meaning

**Model Used:** `sentence-transformers/all-MiniLM-L6-v2`
- Creates 384-dimensional vectors
- Fast and accurate for semantic search
- Works well for research papers

**Example:**
```python
"machine learning" → [0.23, -0.45, 0.12, ..., 0.34]  # 384 numbers
"deep learning"    → [0.21, -0.42, 0.15, ..., 0.31]  # Similar!
"cooking recipes"  → [-0.89, 0.12, -0.43, ..., -0.12] # Different!
```

---

### **2. Chunking Strategy**

**Why chunk documents?**
- Models have token limits
- Smaller chunks = more precise retrieval
- Overlap preserves context at boundaries

**Our Strategy:**
- **Chunk size:** 1000 characters (~200-250 words)
- **Overlap:** 100 characters (~20-25 words)
- **Splitting order:** Paragraphs → Sentences → Spaces → Characters

**Example:**
```
Original: "AI is transforming healthcare. Machine learning models..."

Chunk 1: "AI is transforming healthcare. Machine learning models..."
Chunk 2: "...Machine learning models can predict diseases..."
         ↑ Overlap preserves context
```

---

### **3. Vector Similarity Search**

**How it works:**
1. Convert question to embedding
2. Compare question embedding to all document embeddings
3. Use cosine similarity to measure closeness
4. Return top-k most similar chunks

**Cosine Similarity:**
```
similarity = 1 - cosine_distance
score of 1.0 = identical
score of 0.0 = completely different
```

**SQL Query (pgvector):**
```sql
SELECT content, 1 - (embedding <=> question_vector) AS similarity
FROM documents
ORDER BY embedding <=> question_vector
LIMIT 4;
```

---

### **4. Answer Generation (RAG)**

**RAG = Retrieval-Augmented Generation**
1. **Retrieval:** Find relevant chunks from database
2. **Augmentation:** Add chunks to prompt as context
3. **Generation:** LLM generates answer based on context

**Prompt Structure:**
```
Context: [Retrieved chunks 1-4]

Question: [User's question]

Instructions: Answer based ONLY on context above

Answer: [Gemini generates this]
```

**Why RAG?**
-  Grounded answers (no hallucinations)
-  Cites specific information
-  Works with private documents
-  More accurate than pure LLM

---

##  How to Test the System

### **Quick Start (Automated)**

```bash
# 1. Configure environment
# Edit .env file with your credentials

# 2. Run end-to-end test
python test_pdf_workflow.py
```

This will automatically:
- Check configuration
- Process sample document
- Upload to database
- Test QA with sample questions

---

### **Manual Testing (Step-by-Step)**

#### **Test 1: Document Ingestion**
```bash
# Process a document
python embeddings/ingest.py --in data/sample.pdf --out data/embeddings.jsonl

# Expected output:
#  Read 12,543 characters
#  Created 15 chunks
#  Successfully wrote 15 records to 'data/embeddings.jsonl'
```

#### **Test 2: Database Upload**
```bash
# Upload embeddings
python upload_to_db.py --input data/embeddings.jsonl

# Expected output:
#  Connected successfully!
#  Table 'documents' created/verified
#  Uploaded 15 new documents
```

#### **Test 3: Question Answering**
```bash
# Start interactive QA
python qa_system.py

# Try questions like:
# - "What is this research about?"
# - "What are the main findings?"
# - "What methods were used?"
```

---

### **Verification Checks**

#### Check 1: Database Contents
```sql
-- Connect to your PostgreSQL database
SELECT COUNT(*) FROM documents;
-- Should show number of uploaded chunks

SELECT doc_id, LEFT(content, 100) AS preview 
FROM documents 
LIMIT 5;
-- Should show document previews
```

#### Check 2: Embedding Quality
```bash
python test_qa_system.py

# Look for:
#  Embedding model works! Vector dimension: 384
#  Gemini connection successful!
```

#### Check 3: Search Relevance
Ask test questions and verify:
- Are retrieved chunks relevant?
- Is similarity score > 0.5 for good matches?
- Does answer make sense given context?

---

##  Performance Metrics

### **Expected Performance**

| Task | Time | Notes |
|------|------|-------|
| Load embedding model | 2-5 sec | One-time on startup |
| Process 10-page PDF | 10-20 sec | Includes chunking + embedding |
| Upload 100 chunks | 2-5 sec | Depends on network |
| Answer 1 question | 2-4 sec | Retrieval + Gemini generation |

### **Scaling Considerations**

| Documents | Chunks | Search Time | Storage |
|-----------|--------|-------------|---------|
| 10 papers | ~500 | < 0.1 sec | ~20 MB |
| 100 papers | ~5,000 | < 0.5 sec | ~200 MB |
| 1,000 papers | ~50,000 | ~1 sec | ~2 GB |

---

##  Customization Options

### **Adjust Chunk Size**
```bash
# Smaller chunks (more precise, but more chunks)
python embeddings/ingest.py --in doc.pdf --out out.jsonl --chunk 500 --overlap 50

# Larger chunks (more context, but less precise)
python embeddings/ingest.py --in doc.pdf --out out.jsonl --chunk 2000 --overlap 200
```

### **Change Embedding Model**
```python
# In ingest.py and qa_system.py, change:
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# To:
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
# or
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
```

### **Adjust Retrieval Count**
```python
# In qa_system.py, change retrieve_context():
def retrieve_context(self, question: str, top_k: int = 4):  # Default is 4

# Try:
top_k = 6  # More context (slower, more expensive)
top_k = 2  # Less context (faster, cheaper)
```

### **Modify Gemini Prompt**
Edit the prompt in `qa_system.py` → `generate_answer()` to:
- Change tone (formal vs. casual)
- Add specific instructions
- Request structured output
- Include examples

---

##  Common Issues & Solutions

### Issue 1: Import Errors
```
ImportError: No module named 'google.generativeai'
```
**Solution:**
```bash
.\venv\Scripts\Activate.ps1  # Activate virtual environment
pip install -r requirements.txt
```

### Issue 2: Database Connection Failed
```
psycopg2.OperationalError: could not connect to server
```
**Solution:**
- Check `DATABASE_URL` in `.env`
- Verify Supabase dashboard (Database > Connection Pooler)
- Ensure port is 5432 (Pooler) not 5433 (Direct)

### Issue 3: Gemini API Errors
```
google.api_core.exceptions.PermissionDenied: 403
```
**Solution:**
- Verify `GEMINI_API_KEY` in `.env`
- Check API key is valid: https://aistudio.google.com/app/apikey
- Ensure API is enabled for your project

### Issue 4: No Results from Search
```
Found 0 relevant documents
```
**Solution:**
- Verify documents are in database: `SELECT COUNT(*) FROM documents;`
- Check pgvector extension: `CREATE EXTENSION IF NOT EXISTS vector;`
- Re-upload embeddings: `python upload_to_db.py --input data.jsonl`

---

##  Further Reading

### Understanding RAG
- [RAG Explained (LangChain)](https://python.langchain.com/docs/use_cases/question_answering/)
- [Vector Databases Guide](https://www.pinecone.io/learn/vector-database/)

### Sentence Transformers
- [Model Hub](https://www.sbert.net/docs/pretrained_models.html)
- [Training Custom Models](https://www.sbert.net/docs/training/overview.html)

### Pgvector
- [GitHub Repository](https://github.com/pgvector/pgvector)
- [Performance Tuning](https://github.com/pgvector/pgvector#performance)

---

##  Next Steps

1. **Add More Documents**
   - Process more research papers
   - Build a comprehensive knowledge base

2. **Improve Retrieval**
   - Experiment with different chunk sizes
   - Try hybrid search (keyword + vector)
   - Add metadata filtering

3. **Enhance UI**
   - Build a web interface (Streamlit/Gradio)
   - Add citation display
   - Show confidence scores

4. **Production Deployment**
   - Add caching (Redis)
   - Implement rate limiting
   - Add user authentication
   - Monitor performance

---

**Questions?** Check QUICKSTART.md for hands-on examples!
