# AI-Powered Professor Research Summarizer

Web application dedicated to easily-digestible summaries of complex academic papers. This project uses RAG (Retrieval-Augmented Generation) pipelines to retrieve, analyze, and summarize research papers.

## 🏗️ Tech Stack

- **Languages:** Python
- **AI/ML:** LangChain, Sentence Transformers, Google Gemini
- **Database:** Supabase (PostgreSQL + pgvector)
- **Vector Search:** pgvector with cosine similarity

## 📁 Project Structure

```
├── embeddings/          # Document processing and embedding generation
│   └── ingest.py       # PDF/DOCX → embeddings pipeline
├── qa_system.py        # Main RAG-based QA system
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── docs/              # 📚 Documentation (setup guides, architecture)
├── tests/             # 🧪 Test files and validation scripts
└── scripts/           # 🛠️ Utility scripts (database setup, uploads)
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Francisco-MEB/Aunalytics-AI-Research-Summarizer.git
cd Aunalytics-AI-Research-Summarizer

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file:
```bash
# Supabase Database (see docs/SUPABASE_SETUP.md for setup)
DATABASE_URL=postgresql://postgres.xxx:password@aws-region.pooler.supabase.com:5432/postgres

# Google Gemini API Key (get from https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=your_api_key_here
```

### 3. Usage

```bash
# Process a document
python embeddings/ingest.py --in your_paper.pdf --out data/embeddings.jsonl

# Upload to database
python scripts/upload_to_db.py --input data/embeddings.jsonl

# Ask questions
python qa_system.py
```

## 📖 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[Supabase Setup](docs/SUPABASE_SETUP.md)** - Database configuration
- **[System Architecture](docs/ARCHITECTURE.md)** - How everything works
- **[Testing Guide](docs/TESTING.md)** - Validation and testing

## 🧪 Testing

```bash
# Test Supabase connection
python tests/test_supabase_connection.py

# Test individual components
python tests/test_qa_system.py

# Test full pipeline
python tests/test_pdf_workflow.py
```

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## 📝 License

[Add license information]