# 🚀 Quick Start Guide - Testing PDF to QA System

## ✅ Setup Complete!

I've set up everything you need to test the QA system with a PDF. Here's what's been configured:

### 📦 What Was Installed
- ✅ Virtual environment (`venv/`)
- ✅ All dependencies (Google Generative AI, Sentence Transformers, LangChain, etc.)
- ✅ PDF processing tools (pypdf, docx2txt)
- ✅ Database tools (psycopg)

### 📄 New Files Created
1. **`upload_to_db.py`** - Uploads JSONL embeddings to PostgreSQL/pgvector
2. **`test_pdf_workflow.py`** - End-to-end test script  
3. **`.env`** - Environment variables file (⚠️ **YOU NEED TO FILL THIS IN**)

---

## 🔧 Before You Can Test

### Step 1: Set Up Supabase Database

**Follow the detailed guide:** [`SUPABASE_SETUP.md`](SUPABASE_SETUP.md)

**Quick version (5 minutes):**
1. Go to [supabase.com](https://supabase.com) and create a new project
2. In SQL Editor, run the script from `setup_supabase.sql` (enables pgvector + creates table)
3. Copy your **Connection Pooler** URL from Settings → Database (port 5432)

**Or run:** `python test_supabase_connection.py` to verify your setup

---

### Step 2: Configure Environment Variables

Open the **`.env`** file and add your credentials:

```bash
# Get from Supabase: Settings → Database → Connection string (Connection Pooler)
DATABASE_URL=postgresql://postgres.xxxxx:[YOUR-PASSWORD]@aws-0-region.pooler.supabase.com:5432/postgres

# Get from https://aistudio.google.com/app/apikey  
GEMINI_API_KEY=your_api_key_here
```

**Important Notes:**
- Use **Connection Pooler** URL (port 5432), not direct connection (5433)
- For Gemini, create a free API key at https://aistudio.google.com/app/apikey
- See `SUPABASE_SETUP.md` for detailed Supabase configuration

---

### Step 3: Activate Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

---

## 🧪 Running the Test

Once `.env` is configured, run the end-to-end test:

```powershell
python test_pdf_workflow.py
```

This script will:
1. ✅ Check environment variables
2. ✅ Process the document in `data/science.1203877.docx`
3. ✅ Create embeddings  
4. ✅ Upload to database
5. ✅ Test QA system with sample questions

---

## 📖 Manual Workflow (Step-by-Step)

If you want to run each step manually:

### 1. Process a PDF/Document

```powershell
python embeddings/ingest.py --in data/your_document.pdf --out data/output.jsonl
```

### 2. Upload to Database

```powershell
python upload_to_db.py --input data/output.jsonl
```

### 3. Use QA System Interactively

```powershell
python qa_system.py
```

Then ask questions like:
- "What is this research about?"
- "What are the main findings?"
- "What methods were used?"

---

## 🐛 Troubleshooting

### Supabase Connection Issues
- **Run:** `python test_supabase_connection.py` to diagnose
- Verify DATABASE_URL uses **Connection Pooler** (port 5432, not 5433)
- Check pgvector extension: `CREATE EXTENSION IF NOT EXISTS vector;` in Supabase SQL Editor
- See `SUPABASE_SETUP.md` for detailed troubleshooting

### Import Error: `google.generativeai`
- Make sure virtual environment is activated
- Rerun: `pip install google-generativeai`

### Database Connection Error
- Verify `DATABASE_URL` in `.env` is correct
- Check Supabase dashboard → Database → Connection pooler (port 5432)
- Ensure pgvector extension is enabled (run `test_supabase_connection.py`)

### No documents found in database
- Run `upload_to_db.py` first to populate the database
- Check table exists in Supabase: Table Editor → documents
- Verify uploads succeeded (check terminal output)

---

##  📝 Project Structure

```
Aunalytics-AI-Research-Summarizer/
├── venv/                         # Virtual environment  
├── .env                          # Environment variables (YOU MUST CONFIGURE)
├── .env.example                  # Template
├── SUPABASE_SETUP.md            # 📘 Detailed Supabase setup guide
├── setup_supabase.sql           # SQL script for Supabase
├── test_supabase_connection.py  # Test your Supabase setup
├── embeddings/
│   └── ingest.py                # PDF → Embeddings  
├── data/
│   └── science.1203877.docx     # Sample document
├── upload_to_db.py              # Embeddings → Database
├── qa_system.py                 # Main QA system
├── test_qa_system.py            # Component tests
└── test_pdf_workflow.py         # End-to-end test (RUN THIS!)
```

---

## 🎯 Next Steps

1. **Set up Supabase** - Follow [`SUPABASE_SETUP.md`](SUPABASE_SETUP.md) (5 minutes)
2. **Test connection** - Run `python test_supabase_connection.py`
3. **Fill in `.env`** with your DATABASE_URL and GEMINI_API_KEY
4. **Run end-to-end test** - `python test_pdf_workflow.py`
5. **Use interactively** - `python qa_system.py`
6. **Add your own PDFs** to `data/` directory
7. **Process & upload** them using the manual workflow

---

## 💡 Tips

- The system uses `sentence-transformers/all-MiniLM-L6-v2` (384-dim vectors)
- Default chunk size is 800 characters with 100 overlap
- Adjust these in `ingest.py` with `--chunk` and `--overlap` flags
- You can process .txt, .pdf, and .docx files

---

**Ready to test!** 🎉

Start with: `python test_pdf_workflow.py` (after configuring .env)
