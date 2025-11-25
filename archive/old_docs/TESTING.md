#  Testing Guide - AI Research Summarizer

Complete guide for testing your QA system from scratch to production.

---

##  Testing Checklist

Use this checklist to verify your system works correctly:

- [ ] Environment configured (.env file with credentials)
- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] Database accessible and pgvector enabled
- [ ] Documents ingested and embeddings created
- [ ] Embeddings uploaded to database
- [ ] QA system answers questions correctly
- [ ] Error handling works as expected

---

##  Test Categories

### 1. **Environment Tests** 
### 2. **Component Tests** 
### 3. **Integration Tests** 
### 4. **End-to-End Tests** 
### 5. **Edge Case Tests** 

---

## 1️⃣ Environment Tests

### Test 1.1: Verify Python Environment

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Check Python version (should be 3.8+)
python --version

# Verify key packages
python -c "import google.generativeai; import sentence_transformers; print(' Packages OK')"
```

**Expected Output:**
```
Python 3.13.x
 Packages OK
```

---

### Test 1.2: Verify Environment Variables

```bash
# Check .env file exists
Test-Path .env

# Verify variables are loaded (without showing secrets)
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('DB:', 'SET' if os.getenv('DATABASE_URL') else 'MISSING'); print('API:', 'SET' if os.getenv('GEMINI_API_KEY') else 'MISSING')"
```

**Expected Output:**
```
True
DB: SET
API: SET
```

---

### Test 1.3: Database Connection

```bash
# Test database connectivity
python -c "import os; import psycopg2; from dotenv import load_dotenv; load_dotenv(); conn = psycopg2.connect(os.getenv('DATABASE_URL')); print(' Database connected'); conn.close()"
```

**Expected Output:**
```
 Database connected
```

---

## 2️⃣ Component Tests

### Test 2.1: Document Reading

**Test PDF:**
```bash
python -c "from embeddings.ingest import read_pdf; text = read_pdf('data/sample.pdf'); print(f' Read {len(text)} characters')"
```

**Test DOCX:**
```bash
python -c "from embeddings.ingest import read_docx; text = read_docx('data/science.1203877.docx'); print(f' Read {len(text)} characters')"
```

---

### Test 2.2: Text Chunking

```bash
python -c "
from embeddings.ingest import chunk_text
text = 'This is a test. ' * 200  # 3000 chars
chunks = chunk_text(text, 1000, 100)
print(f' Created {len(chunks)} chunks')
print(f'   Chunk 1 length: {len(chunks[0][\"text\"])} chars')
"
```

**Expected Output:**
```
 Created 3 chunks
   Chunk 1 length: ~1000 chars
```

---

### Test 2.3: Embedding Generation

```bash
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding = model.encode('test sentence')
print(f' Embedding dimension: {len(embedding)}')
print(f'   Sample values: {embedding[:5]}')
"
```

**Expected Output:**
```
 Embedding dimension: 384
   Sample values: [0.123, -0.456, ...]
```

---

### Test 2.4: Gemini API Connection

```bash
python -c "
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash')
response = model.generate_content('Say hello in one word')
print(f' Gemini response: {response.text}')
"
```

**Expected Output:**
```
 Gemini response: Hello
```

---

## 3️⃣ Integration Tests

### Test 3.1: Full Ingestion Pipeline

**Small Test Document:**
```bash
# Create test file
echo "Machine learning is a subset of AI. Deep learning uses neural networks." > test_doc.txt

# Process it
python embeddings/ingest.py --in test_doc.txt --out test_embeddings.jsonl --chunk 50

# Verify output
python -c "import json; records = [json.loads(line) for line in open('test_embeddings.jsonl')]; print(f' Created {len(records)} embeddings'); print(f'   First chunk: {records[0][\"text\"][:50]}...')"
```

**Expected Output:**
```
 Created 2 embeddings
   First chunk: Machine learning is a subset of AI. Deep lear...
```

---

### Test 3.2: Database Upload

```bash
# Upload test embeddings
python upload_to_db.py --input test_embeddings.jsonl

# Verify in database
python -c "
import os, psycopg2
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM documents')
count = cur.fetchone()[0]
print(f' Database has {count} documents')
conn.close()
"
```

---

### Test 3.3: Retrieval Test

```bash
# Test similarity search
python -c "
import os
from dotenv import load_dotenv
from qa_system import QASystem
load_dotenv()
qa = QASystem()
context = qa.retrieve_context('What is machine learning?')
print(f' Retrieved {len(context)} documents')
for i, doc in enumerate(context, 1):
    score = doc.metadata.get('score', 0)
    print(f'   {i}. Score: {score:.2%} | {doc.page_content[:50]}...')
"
```

---

## 4️⃣ End-to-End Tests

### Test 4.1: Automated E2E Test

```bash
# Run complete workflow test
python test_pdf_workflow.py
```

**This tests:**
1. Environment configuration
2. Document ingestion
3. Database upload
4. QA system operation

**Expected Output:**
```
============================================================
  PDF → QA System End-to-End Test
============================================================

Checking Prerequisites
 DATABASE_URL is set
 GEMINI_API_KEY is set

Step 1: PDF Ingestion (Create Embeddings)
 PDF ingestion successful!

Step 2: Upload to Database
 Database upload successful!

Step 3: Test QA System
 QA system initialized!
 Question 1: What is this research about?
 Answer: [Generated answer]...
 Sources used: 4

============================================================
   All Tests Passed!
============================================================
```

---

### Test 4.2: Interactive QA Test

```bash
# Start interactive session
python qa_system.py
```

**Test Questions:**
```
 Your question: What is the main topic of this research?
 Your question: What methods were used?
 Your question: What are the key findings?
 Your question: Who are the authors?
```

**Validation Checklist:**
- [ ] System initializes without errors
- [ ] Questions return relevant answers
- [ ] Source documents are displayed
- [ ] Similarity scores are reasonable (> 0.5 for good matches)
- [ ] Answers are grounded in context (no hallucinations)

---

## 5️⃣ Edge Case Tests

### Test 5.1: Empty Query

```bash
python -c "
from qa_system import QASystem
qa = QASystem()
result = qa.ask_question('')
print(' Handled empty query')
"
```

---

### Test 5.2: Question with No Matches

```bash
python -c "
from qa_system import QASystem
qa = QASystem()
result = qa.ask_question('xyzabc nonsense impossible match')
print(f'Answer: {result[\"answer\"]}')
print(' Handled no-match query')
"
```

**Expected:** Should gracefully say no relevant information found.

---

### Test 5.3: Very Long Question

```bash
python -c "
from qa_system import QASystem
qa = QASystem()
long_q = 'What is ' + ' '.join(['machine learning'] * 100) + '?'
result = qa.ask_question(long_q)
print(f' Handled long query ({len(long_q)} chars)')
"
```

---

### Test 5.4: Special Characters

```bash
python -c "
from qa_system import QASystem
qa = QASystem()
result = qa.ask_question('What is AI? #test @mention & special chars!')
print(' Handled special characters')
"
```

---

### Test 5.5: Database Disconnection

```bash
# Temporarily set wrong DATABASE_URL
python -c "
import os
os.environ['DATABASE_URL'] = 'postgresql://wrong:wrong@localhost:5432/wrong'
try:
    from qa_system import QASystem
    qa = QASystem()
    qa.ask_question('test')
except Exception as e:
    print(f' Error handled gracefully: {type(e).__name__}')
"
```

---

##  Performance Tests

### Test P1: Ingestion Speed

```bash
# Time a full document
python -c "
import time
start = time.time()
# Process document
from embeddings.ingest import read_document, chunk_text, embed_chunks
text = read_document('data/science.1203877.docx')
chunks = chunk_text(text, 1000, 100)
vectors = embed_chunks(chunks, 'sentence-transformers/all-MiniLM-L6-v2')
elapsed = time.time() - start
print(f' Processed {len(chunks)} chunks in {elapsed:.2f} seconds')
print(f'   Speed: {len(chunks)/elapsed:.1f} chunks/sec')
"
```

---

### Test P2: Query Speed

```bash
# Time 10 queries
python -c "
import time
from qa_system import QASystem
qa = QASystem()
questions = ['What is AI?'] * 10
start = time.time()
for q in questions:
    qa.ask_question(q)
elapsed = time.time() - start
print(f' Answered {len(questions)} questions in {elapsed:.2f} seconds')
print(f'   Speed: {elapsed/len(questions):.2f} sec/question')
"
```

---

##  Debugging Tests

### Debug 1: Check Database Schema

```sql
-- Run in PostgreSQL
\d documents

-- Should show:
-- Column    | Type           
-- ----------|----------------
-- doc_id    | text           
-- content   | text           
-- embedding | vector(384)    
-- metadata  | jsonb          
-- created_at| timestamp      
```

---

### Debug 2: Inspect Embeddings

```bash
python -c "
import json
records = [json.loads(line) for line in open('data/test_embedded.jsonl')]
print(f'Total records: {len(records)}')
print(f'\nFirst record:')
r = records[0]
print(f'  ID: {r[\"id\"]}')
print(f'  Text: {r[\"text\"][:100]}...')
print(f'  Metadata: {r[\"metadata\"]}')
print(f'  Embedding length: {len(r[\"embedding\"])}')
print(f'  Embedding sample: {r[\"embedding\"][:5]}')
"
```

---

### Debug 3: Check Vector Similarity

```bash
python -c "
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
v1 = model.encode('machine learning')
v2 = model.encode('deep learning')
v3 = model.encode('cooking recipes')
sim12 = 1 - cosine(v1, v2)
sim13 = 1 - cosine(v1, v3)
print(f'Similarity (ML vs DL): {sim12:.3f}  # Should be high')
print(f'Similarity (ML vs Cooking): {sim13:.3f}  # Should be low')
"
```

---

##  Success Criteria

Your system is working correctly if:

1. **All imports succeed** 
2. **Documents are processed without errors** 
3. **Embeddings have correct dimension (384)** 
4. **Database contains uploaded documents** 
5. **Queries return relevant results** 
6. **Similarity scores are reasonable (0.4-0.9)** 
7. **Gemini generates coherent answers** 
8. **No crashes or exceptions** 

---

##  Common Test Failures

### Failure 1: Model Download Timeout
**Symptom:** Hangs on first run
**Cause:** Downloading model from HuggingFace
**Solution:** Wait (1-2 GB download), or pre-download:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

---

### Failure 2: Out of Memory
**Symptom:** Process killed during embedding
**Cause:** Too many chunks or large batch size
**Solution:** Reduce batch size:
```bash
python embeddings/ingest.py --in doc.pdf --out out.jsonl --batch 32
```

---

### Failure 3: Slow Queries
**Symptom:** Queries take > 5 seconds
**Cause:** No vector index or too many documents
**Solution:** Create index:
```sql
CREATE INDEX documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

---

##  Test Reports

### Generate Test Report

```bash
# Run all tests and save output
python test_pdf_workflow.py > test_report.txt 2>&1
python test_qa_system.py >> test_report.txt 2>&1

# View report
cat test_report.txt
```

---

##  Next Steps

After all tests pass:

1. **Add more documents** to build knowledge base
2. **Tune parameters** (chunk size, retrieval count)
3. **Benchmark performance** on your specific use case
4. **Deploy to production** (see DOCUMENTATION.md)

---

**Happy Testing!** 

For more details, see:
- DOCUMENTATION.md - System architecture
- QUICKSTART.md - Quick start guide
- README.md - Project overview
