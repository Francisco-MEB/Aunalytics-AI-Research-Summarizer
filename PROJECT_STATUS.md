# Aunalytics AI Research Summarizer - Project Status Report

**Date**: January 2025  
**Status**: ✅ Development Complete | 🚧 Testing Phase | 🔄 Deployment Pending

---

## 📊 Executive Summary

This project is a **Retrieval-Augmented Generation (RAG) system** designed to help students understand their professor's research. The system can:

1. **Scrape professor websites** for research papers
2. **Extract abstracts** from Google Scholar, Academia.edu, and ResearchGate
3. **Store documents** with embeddings in Supabase PostgreSQL
4. **Answer questions** using semantic search + Gemini LLM
5. **Serve a web interface** through FastAPI backend

**Current Architecture**: Simple vector similarity search (no RAPTOR hierarchy)

---

## 🌿 Branch-by-Branch Analysis

### **1. Backend Branch (Your Work) - MAIN PRODUCTION CODE**
**Status**: ✅ Complete, ✅ Tested (API running), 🔄 Database connection needed

#### Files:
- `main.py` - FastAPI web server with 6 REST endpoints
- `qa_system.py` - RAG system with Gemini 2.5 Flash
- `scraper.py` - Web scraper using BeautifulSoup
- `test_scraper.py` - Testing utilities
- `Dockerfile` - Container configuration
- `.dockerignore` - Build exclusions
- `DEPLOYMENT.md` - Cloud Run guide
- `SCRAPER_DOCS.md` - Scraper documentation

#### API Endpoints:
```
GET  /                          → Health check
GET  /health                    → Database connectivity test
POST /api/upload                → Upload PDF/DOCX/TXT
POST /api/scrape                → Scrape professor website
POST /api/question              → Ask questions (RAG)
DELETE /api/documents/{user_id} → Delete user data
GET  /api/documents/{user_id}/count → Count user documents
```

#### Technology Stack:
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **BeautifulSoup4** - HTML parsing
- **Sentence Transformers** - Embeddings (all-MiniLM-L6-v2, 384-dim)
- **Gemini 2.5 Flash** - LLM for answers
- **Supabase PostgreSQL** - Vector database with pgvector
- **Docker** - Containerization
- **Google Cloud Run** - Deployment platform

#### Testing Results:
```
✅ Server starts successfully on port 8080
✅ Root endpoint (/) returns JSON: {"status": "online"}
✅ Health endpoint (/health) returns database error (expected - no local Postgres)
❌ Database connection not configured locally (Supabase only)
```

---

### **2. Origin/Scraper Branch - TEAM MEMBER'S APPROACH**
**Status**: ✅ Functional, 🔄 Integration pending

#### Key File: `scraper_modified.py`

#### Approach:
- Uses **scholarly** Python library (official Google Scholar API wrapper)
- **Playwright** browser automation as fallback
- Extracts **10 abstracts per professor**

#### Workflow:
1. Search for Google Scholar link on professor's website
2. If not found → extract professor name from URL/HTML
3. Search Google Scholar by name → present matches
4. If no matches → use Playwright to automate Bing search
5. Extract author ID from Scholar URL
6. Fetch 10 publications with abstracts

#### Advantages over Your Scraper:
- ✅ More reliable (uses official scholarly library)
- ✅ Gets abstracts directly (no HTML parsing)
- ✅ Handles rate limiting with proxies
- ✅ Playwright fallback for difficult cases

#### Integration Recommendation:
**Replace your `scraper.py` with `scraper_modified.py`** OR merge the two approaches:
- Use `scholarly` library as primary method
- Keep your BeautifulSoup approach as fallback

---

### **3. Origin/Author_Scraper Branch - BIO EXTRACTION**
**Status**: ✅ Fully functional, 🔄 Integration opportunity

#### Key File: `info_scraper.py`

#### Extracts:
- ✅ Professor name (URL, meta tags, H1)
- ✅ Email address
- ✅ Biography (2-4 paragraphs)
- ✅ Education history
- ✅ Research interests
- ✅ Office hours
- ✅ Course listings (codes + titles)
- ✅ Important links (CV, GitHub, LinkedIn)

#### Features:
- Smart name validation (avoids menu text, navigation)
- Follows "Teaching" links automatically
- Parses course tables
- Extracts structured JSON output
- Tested on 3 university websites

#### Integration Recommendation:
**Add a new endpoint** to your FastAPI:
```python
POST /api/scrape/bio
```
This would use `info_scraper.py` to get professor details before scraping papers.

**User Flow**:
1. User enters professor website URL
2. Backend calls `info_scraper.scrape_professor_website(url)`
3. Displays: Name, Bio, Email, Courses on frontend
4. Then scrapes papers with your existing `/api/scrape` endpoint

---

### **4. Origin/Frontend Branch - USER INTERFACE**
**Status**: ✅ Complete UI, 🔄 Backend integration needed

#### Files:
- `frontend/index.html` - Landing page
- `frontend/query.html` - Query interface
- `frontend/style.css` - Styling
- `frontend/script.js` - JavaScript logic
- `frontend/images/LOGO.png` - Aunalytics logo

#### Design:
- **Landing Page**: Hero section with "Get Started" button
- **Query Page**: 
  - URL input field
  - "Analyze Website" button
  - Summary output box
  - Research papers scroller (10 papers)
  - Chat interface for questions
  - File upload button (📎)

#### Current State:
- ✅ Beautiful UI with Montserrat font
- ✅ Responsive layout with split panels
- ⚠️ Hardcoded placeholder data
- ❌ No JavaScript implementation yet

#### Integration Requirements:
Your `script.js` needs to call:
```javascript
// When "Analyze Website" clicked
fetch('http://localhost:8080/api/scrape', {
    method: 'POST',
    body: JSON.stringify({
        professor_url: url,
        user_id: userId,
        max_papers: 10
    })
})

// When user asks question
fetch('http://localhost:8080/api/question', {
    method: 'POST',
    body: JSON.stringify({
        question: userQuestion,
        user_id: userId
    })
})

// When user uploads file
fetch('http://localhost:8080/api/upload', {
    method: 'POST',
    body: formData // with file and user_id
})
```

---

## 🔗 Integration Architecture

### **Proposed Full System**

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (Browser)                      │
│  index.html → query.html → script.js                       │
│  - URL Input                                               │
│  - Paper Display                                           │
│  - Chat Interface                                          │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP Requests
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              BACKEND - FastAPI (main.py)                    │
│  Port 8080, Uvicorn ASGI Server                            │
├─────────────────────────────────────────────────────────────┤
│  Endpoints:                                                 │
│  • POST /api/scrape/bio  → info_scraper.py (NEW)          │
│  • POST /api/scrape      → scraper_modified.py (UPGRADE)  │
│  • POST /api/upload      → store_to_supabase              │
│  • POST /api/question    → qa_system.py                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │ Google  │      │ Sentence│      │ Gemini  │
    │ Scholar │      │Transform│      │  2.5    │
    │   API   │      │   Model │      │  Flash  │
    └─────────┘      └─────────┘      └─────────┘
                           │
                           ↓
                  ┌──────────────────┐
                  │   Supabase DB    │
                  │  PostgreSQL      │
                  │  + pgvector      │
                  │  + RLS Policies  │
                  └──────────────────┘
```

### **Data Flow Example**

1. **User enters**: `https://sites.nd.edu/taeho-jung/`
2. **Frontend calls**: `POST /api/scrape/bio`
3. **info_scraper.py** extracts:
   ```json
   {
     "name": "Taeho Jung",
     "email": "tjung@nd.edu",
     "bio": "Assistant Professor...",
     "research_interests": ["Computer Security", "IoT"],
     "courses": [
       {"code": "CSE 40567", "title": "Computer Security"}
     ]
   }
   ```
4. **Frontend displays** professor info
5. **Frontend calls**: `POST /api/scrape` (papers)
6. **scraper_modified.py** finds Google Scholar → extracts 10 papers
7. **Backend embeds** abstracts → stores in Supabase
8. **Frontend shows** paper list
9. **User asks**: "What is Prof. Jung's main research focus?"
10. **Frontend calls**: `POST /api/question`
11. **qa_system.py**:
    - Embeds question
    - Searches Supabase for top 15 similar chunks
    - Sends to Gemini with context
    - Returns answer
12. **Frontend displays** answer in chat

---

## 🚧 What Still Needs to Be Done

### **Immediate (Before Deployment)**

1. **Fix store_to_supabase import** in `main.py`
   - Current: Import error "NoneType has no loader"
   - Solution: Check file location, import syntax

2. **Merge scraper implementations**
   - Replace BeautifulSoup scraper with `scholarly` library
   - Keep fallback logic

3. **Integrate info_scraper**
   - Add `/api/scrape/bio` endpoint
   - Call before paper scraping

4. **Connect frontend to backend**
   - Implement `script.js` functions
   - Handle API responses
   - Update UI dynamically

5. **Test end-to-end**
   - Upload document → verify embeddings
   - Scrape website → check papers stored
   - Ask question → validate RAG response

### **Optional Enhancements**

6. **Add user authentication**
   - Generate unique user_id per session
   - Store in localStorage/cookies

7. **Improve error handling**
   - Better error messages in UI
   - Retry logic for failed scrapes

8. **Add caching**
   - Cache professor data (avoid re-scraping)
   - Cache embeddings for common questions

---

## 🐳 Docker & Deployment

### **Current Docker Setup**

**Dockerfile** (backend branch):
```dockerfile
FROM python:3.11-slim
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y gcc g++ postgresql-client
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY qa_system.py .
COPY main.py .
COPY embeddings/ ./embeddings/

EXPOSE 8080
ENV PORT=8080

CMD ["python", "main.py"]
```

**Status**: ✅ Complete, 🔄 Not tested

### **Missing for Deployment**

1. **Environment variables** (not in Dockerfile):
   ```env
   DATABASE_URL=postgresql://...
   GEMINI_API_KEY=...
   ```

2. **scraper.py** not copied to Docker image
   ```dockerfile
   COPY scraper.py .  # ADD THIS
   ```

3. **Frontend files** not included
   ```dockerfile
   COPY frontend/ ./frontend/  # ADD THIS
   ```

---

## ☁️ Google Cloud Run Deployment Guide

### **What is Google Cloud Run?**

**Cloud Run** is a **serverless platform** that runs your Docker container without managing servers. You pay only when requests come in (unlike EC2/VM that runs 24/7).

**Key Features**:
- Automatically scales from 0 to thousands of containers
- Charges per 100ms of request time
- Handles HTTPS automatically
- Integrates with Google Cloud services

### **Prerequisites**

1. **Google Cloud Project**:
   ```bash
   # Create project (do this once)
   gcloud projects create aunalytics-rag-summarizer --name="Aunalytics RAG"
   
   # Set as active project
   gcloud config set project aunalytics-rag-summarizer
   ```

2. **Enable billing** (required for Cloud Run):
   - Go to: https://console.cloud.google.com/billing
   - Link your credit card (free tier: 2 million requests/month)

3. **Enable APIs**:
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

4. **Install gcloud CLI** (if not installed):
   - Download: https://cloud.google.com/sdk/docs/install
   - Run: `gcloud init`
   - Login: `gcloud auth login`

### **Deployment Steps**

#### **Step 1: Build Docker Image**

```bash
# Navigate to project directory
cd "C:\Users\vecer\OneDrive\Documents\GitHub\Aunalytics-AI-Research-Summarizer"

# Build image locally first (test)
docker build -t aunalytics-rag:local .

# Test locally
docker run -p 8080:8080 -e DATABASE_URL="your-supabase-url" -e GEMINI_API_KEY="your-key" aunalytics-rag:local

# If works, build for Cloud Run (pushed to Google Container Registry)
gcloud builds submit --tag gcr.io/aunalytics-rag-summarizer/aunalytics-rag:v1
```

**What this does**:
- `gcloud builds submit` uploads your code to Google Cloud
- Google builds the Docker image in the cloud (faster, no local Docker needed)
- Image is stored in **Google Container Registry** (GCR)
- `gcr.io/PROJECT-ID/IMAGE-NAME:TAG` is the naming format

#### **Step 2: Deploy to Cloud Run**

```bash
gcloud run deploy aunalytics-rag-api \
  --image gcr.io/aunalytics-rag-summarizer/aunalytics-rag:v1 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GEMINI_API_KEY=YOUR_GEMINI_KEY" \
  --set-env-vars "DATABASE_URL=YOUR_SUPABASE_URL" \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300s \
  --max-instances 10
```

**Explanation of flags**:
- `--image`: Which Docker image to use
- `--platform managed`: Use serverless Cloud Run (not GKE)
- `--region us-central1`: Where servers run (Iowa data center)
- `--allow-unauthenticated`: Anyone can access (for public API)
- `--set-env-vars`: Pass secrets to container
- `--memory 1Gi`: 1GB RAM per instance (embeddings need memory)
- `--cpu 1`: 1 CPU core
- `--timeout 300s`: 5 minutes max request time (for long scrapes)
- `--max-instances 10`: Scale up to 10 containers max

**Output**:
```
Deploying container to Cloud Run service [aunalytics-rag-api]
✓ Deploying new service... Done.
  ✓ Creating Revision...
  ✓ Routing traffic...
Service [aunalytics-rag-api] revision [aunalytics-rag-api-00001] deployed.
Service URL: https://aunalytics-rag-api-abc123-uc.a.run.app
```

#### **Step 3: Test Deployed API**

```bash
# Test health endpoint
curl https://aunalytics-rag-api-abc123-uc.a.run.app/health

# Test from PowerShell
Invoke-RestMethod -Uri "https://aunalytics-rag-api-abc123-uc.a.run.app/" -Method GET
```

#### **Step 4: Update Frontend**

Change `script.js` to use Cloud Run URL:
```javascript
const API_BASE_URL = "https://aunalytics-rag-api-abc123-uc.a.run.app";
```

#### **Step 5: Deploy Frontend**

**Option A: Google Cloud Storage (Static Hosting)**
```bash
# Create bucket
gsutil mb gs://aunalytics-rag-frontend

# Upload files
gsutil -m cp -r frontend/* gs://aunalytics-rag-frontend/

# Make public
gsutil iam ch allUsers:objectViewer gs://aunalytics-rag-frontend

# Access at:
# https://storage.googleapis.com/aunalytics-rag-frontend/index.html
```

**Option B: Serve from FastAPI**
Add to `main.py`:
```python
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

### **Cost Estimate**

**Cloud Run Pricing** (as of 2025):
- **CPU**: $0.00002400/vCPU-second
- **Memory**: $0.00000250/GB-second
- **Requests**: $0.40 per million requests

**Example for 1000 users/month**:
- 1000 scrape requests × 10 seconds each = 10,000 CPU-seconds = $0.24
- 1000 queries × 2 seconds each = 2,000 CPU-seconds = $0.05
- Total: **$0.29/month** (well within free tier!)

**Free Tier**:
- 2 million requests
- 360,000 GB-seconds memory
- 180,000 vCPU-seconds compute

---

## 📝 GitHub Sync Status

### **Current State**

```bash
# Commits on backend branch
99cd2a5 (HEAD -> backend, origin/backend) feat: Add professor website scraper
4bb0115 feat: Add FastAPI backend with Google Cloud Run deployment
52e6fe6 Changed backend to pre-raptor system
fd083f1 Added RLS to Supabase
```

**Status**: ✅ All commits pushed to GitHub

### **Branch Summary**

| Branch | Status | Description |
|--------|--------|-------------|
| `backend` | ✅ Up to date | Your FastAPI + scraper |
| `origin/scraper` | 🔄 Not merged | Team's scholarly scraper |
| `origin/author_scraper` | 🔄 Not merged | Team's bio scraper |
| `origin/frontend` | 🔄 Not merged | Team's HTML/CSS UI |
| `main` | 🔄 Behind | Base project |

### **Recommended Merge Strategy**

```bash
# 1. Create integration branch
git checkout -b integration

# 2. Merge your backend
git merge backend

# 3. Merge frontend
git merge origin/frontend --allow-unrelated-histories

# 4. Merge scraper improvements
git checkout origin/scraper -- scraper_modified.py
git add scraper_modified.py
git commit -m "feat: Add scholarly-based scraper"

# 5. Merge bio scraper
git checkout origin/author_scraper -- info_scraper.py
git add info_scraper.py
git commit -m "feat: Add professor bio scraper"

# 6. Test everything
python main.py
# Open http://localhost:8080

# 7. Push integration branch
git push -u origin integration

# 8. Create Pull Request on GitHub
# integration → main
```

---

## 🎯 Next Steps (Priority Order)

### **Phase 1: Integration (1-2 days)**
- [ ] Fix `store_to_supabase` import error
- [ ] Merge `scraper_modified.py` (replace BeautifulSoup)
- [ ] Add `/api/scrape/bio` endpoint with `info_scraper.py`
- [ ] Implement `script.js` frontend logic
- [ ] Test full pipeline locally

### **Phase 2: Testing (1 day)**
- [ ] Test with 3 professor websites
- [ ] Verify embeddings in Supabase
- [ ] Test RAG question answering
- [ ] Fix bugs, improve error handling

### **Phase 3: Deployment (1 day)**
- [ ] Update Dockerfile (add scraper.py, frontend/)
- [ ] Build Docker image locally
- [ ] Deploy to Google Cloud Run
- [ ] Test live API
- [ ] Deploy frontend to Cloud Storage

### **Phase 4: Polish (optional)**
- [ ] Add user authentication
- [ ] Improve UI/UX
- [ ] Add caching
- [ ] Monitor performance

---

## 🔧 Troubleshooting

### **Common Issues**

**1. ModuleNotFoundError: db_connection**
- **Cause**: Import path incorrect
- **Fix**: Add `sys.path.insert()` before import (already done)

**2. Database connection failed**
- **Cause**: No local PostgreSQL running
- **Fix**: Use Supabase connection string with DATABASE_URL env var

**3. Sentence transformer model slow**
- **Cause**: Downloads 90MB model on first run
- **Fix**: Pre-download in Dockerfile with RUN command

**4. CORS errors in frontend**
- **Cause**: Browser blocks cross-origin requests
- **Fix**: Already added CORS middleware to FastAPI

**5. Cloud Run timeout (504)**
- **Cause**: Scraping takes > 60 seconds
- **Fix**: Increase `--timeout` to 300s (done in deploy command)

---

## 📚 Key Documentation

- **Backend API**: `c:\...\main.py` (lines 1-443)
- **Scraper**: `SCRAPER_DOCS.md`
- **Deployment**: `DEPLOYMENT.md`
- **Database**: `documentation/RLS_IMPLEMENTATION.md`
- **Quick Ref**: `documentation/QUICK_REFERENCE.md`

---

## ✅ Testing Summary

| Component | Status | Notes |
|-----------|--------|-------|
| FastAPI Server | ✅ Working | Runs on port 8080 |
| Root endpoint (/) | ✅ Working | Returns JSON |
| Health endpoint | ⚠️ Partial | Needs DB connection |
| Upload endpoint | 🔄 Not tested | Needs store_to_supabase fix |
| Scrape endpoint | 🔄 Not tested | Needs real URL test |
| Question endpoint | 🔄 Not tested | Needs data in DB |
| Frontend UI | ✅ Complete | Not connected to backend |
| Dockerfile | 🔄 Not built | Ready to build |
| Cloud Run | 🔄 Not deployed | Guide complete |

---

## 🎓 Key Concepts Explained

### **What is RAG (Retrieval-Augmented Generation)?**

Traditional LLM:
```
User: "What is Prof. Jung's research?"
LLM: "I don't have information about specific professors."
```

RAG System:
```
User: "What is Prof. Jung's research?"
↓
1. Embed question into vector [0.23, -0.45, ...]
2. Search database for similar vectors
3. Retrieve top 15 matching chunks
4. Send to LLM: "Based on these papers: [chunks], answer: [question]"
↓
LLM: "Prof. Jung's research focuses on computer security and IoT privacy..."
```

### **What is Vector Similarity Search?**

Text → Numbers:
```
"machine learning"     → [0.23, 0.45, 0.67, ...]
"deep learning"        → [0.25, 0.47, 0.63, ...]  ← Similar!
"cooking recipes"      → [0.89, -0.12, 0.05, ...] ← Different!
```

Similarity = Cosine distance between vectors

### **What is RLS (Row Level Security)?**

Without RLS:
```sql
SELECT * FROM documents;  -- Returns EVERYONE's data
```

With RLS:
```sql
-- Policy: users can only see their own documents
CREATE POLICY user_isolation ON documents
  USING (user_id = current_user_id());

-- Now queries automatically filter:
SELECT * FROM documents;  -- Returns only YOUR data
```

### **What is CORS?**

Browser security:
```
Frontend (localhost:5500) → Backend (localhost:8080)
❌ BLOCKED by browser (different origin)

Solution: Backend adds header
Access-Control-Allow-Origin: *
✅ Now allowed
```

---

## 📞 Support Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com
- **Cloud Run Docs**: https://cloud.google.com/run/docs
- **Sentence Transformers**: https://www.sbert.net
- **Supabase Docs**: https://supabase.com/docs
- **Gemini API**: https://ai.google.dev/gemini-api/docs

---

**Report Generated**: January 2025  
**Project**: Aunalytics AI Research Summarizer  
**Team**: Backend (you), Scraper Team, Frontend Team, Bio Scraper Team
