# File Upload & Query System Setup

## Overview
The system now supports:
1. **File Upload** - Upload PDF, DOCX, or TXT files
2. **Automatic Chunking** - Splits documents into ~1000 character chunks with 100 character overlap
3. **Embeddings** - Generates 384-dimensional vectors using sentence-transformers
4. **Storage** - Stores in Supabase with Row Level Security (RLS)
5. **RAG Queries** - Retrieves relevant chunks and generates answers with Gemini AI

## Backend Setup

### 1. Environment Variables
Make sure your `.env` file has:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
GEMINI_API_KEY=your_gemini_api_key
```

### 2. Start Backend Server
```powershell
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The server will start at `http://127.0.0.1:8000`

## Frontend Setup

### 1. Open the Frontend
Open `frontend/query.html` in your browser or use a local server:
```powershell
cd frontend
python -m http.server 8080
```
Then visit `http://localhost:8080/query.html`

### 2. Using the System

#### Upload a Document:
1. Click the 📎 button in the chat interface
2. Select a PDF, DOCX, or TXT file
3. The file will be automatically:
   - Chunked into smaller pieces
   - Embedded using AI
   - Stored in your Supabase database
4. You'll see a success message with the number of chunks created

#### Query Your Documents:
1. Type a question in the chat input
2. Press Enter or click the send button
3. The system will:
   - Find relevant document chunks
   - Generate an answer using Gemini AI
   - Show you the response with source count

## API Endpoints

### POST /ingest/
Upload and process a document
- **file**: Document file (PDF/DOCX/TXT)
- **user_id**: UUID for user (required)
- **chunk_size**: Characters per chunk (default 1000)
- **chunk_overlap**: Overlap between chunks (default 100)

**Response:**
```json
{
  "status": "success",
  "num_chunks": 42,
  "filename": "research.pdf",
  "user_id": "your-uuid"
}
```

### POST /query/
Query the knowledge base
- **message**: Your question (required)
- **user_id**: UUID for user (required)
- **top_k**: Number of chunks to retrieve (default 4, use 15 for summaries)

**Response:**
```json
{
  "response": "AI generated answer based on your documents...",
  "num_sources": 4,
  "user_id": "your-uuid"
}
```

## How It Works

### File Upload Flow:
```
User uploads file
    ↓
Backend reads file (PDF/DOCX/TXT)
    ↓
Text is chunked using RecursiveCharacterTextSplitter
    ↓
Each chunk is embedded (384-dim vector)
    ↓
Chunks + embeddings stored in Supabase
    ↓
Success message returned to frontend
```

### Query Flow:
```
User asks question
    ↓
Question is embedded (384-dim vector)
    ↓
Vector similarity search in Supabase
    ↓
Top-K most relevant chunks retrieved
    ↓
Chunks sent to Gemini AI as context
    ↓
AI generates answer based on context
    ↓
Answer displayed to user
```

## User ID Management
- Each browser session gets a unique UUID stored in localStorage
- This ensures your documents are kept separate from others
- Check browser console to see your User ID

## Troubleshooting

### Backend won't start:
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify environment variables are set correctly
- Check if port 8000 is already in use

### File upload fails:
- Check file format (must be PDF, DOCX, or TXT)
- Verify Supabase connection
- Check browser console for error messages

### Query returns no results:
- Make sure you've uploaded documents first
- Check that you're using the same user_id
- Verify Supabase table has data

### CORS errors:
- Make sure backend is running on port 8000
- Check that CORS middleware is configured correctly in backend

## Testing the System

### 1. Test File Upload:
```bash
curl -X POST http://127.0.0.1:8000/ingest/ \
  -F "file=@test.txt" \
  -F "user_id=00000000-0000-0000-0000-000000000001"
```

### 2. Test Query:
```bash
curl -X POST http://127.0.0.1:8000/query/ \
  -F "message=What is this document about?" \
  -F "user_id=00000000-0000-0000-0000-000000000001"
```

## Files Modified

### Backend:
- ✅ `backend/api/router_ingest.py` - Complete rewrite with proper chunking and embedding
- ✅ `backend/api/router_query.py` - RAG query system with Gemini integration
- ✅ `backend/main.py` - Added query router

### Frontend:
- ✅ `frontend/script.js` - Complete rewrite with upload and query functionality
- ✅ User ID management with localStorage
- ✅ Visual feedback for uploads and queries

## Next Steps

1. ✅ Implement user authentication (replace localStorage UUID with real auth)
2. ✅ Add file management (view/delete uploaded files)
3. ✅ Add conversation history persistence
4. ✅ Add document metadata display
5. ✅ Implement RAPTOR hierarchical summarization
