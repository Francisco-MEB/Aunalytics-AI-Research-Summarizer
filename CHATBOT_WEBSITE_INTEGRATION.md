# Chatbot Website Integration

## What Changed

The system now automatically stores all scraped website content in your Supabase knowledge base, making it queryable through the chatbot.

## Features Added

### 1. **Homepage Content Storage**
- When you analyze a website, the entire homepage text is:
  - Chunked into ~1000 character pieces
  - Embedded with AI vectors
  - Stored in Supabase with metadata

### 2. **Research Papers Storage**
- All Google Scholar papers found are now:
  - Chunked (not just stored as single pieces)
  - Properly embedded
  - Tagged with paper title and type

### 3. **AI Summary Storage**
- The RAG-generated summary itself is:
  - Chunked and embedded
  - Stored as a separate document
  - Tagged as "ai_summary" type

### 4. **Unified Querying**
- The chatbot now searches across:
  - ✅ Uploaded PDF/DOCX/TXT files
  - ✅ Scraped website homepages
  - ✅ Research paper abstracts
  - ✅ AI-generated summaries

## How It Works

### Workflow:
```
User enters professor website URL
    ↓
Backend scrapes homepage text
    ↓
Homepage text → chunked → embedded → stored in Supabase
    ↓
Find Google Scholar link
    ↓
Extract research papers
    ↓
Each paper abstract → chunked → embedded → stored
    ↓
Generate RAG summary
    ↓
Summary → chunked → embedded → stored
    ↓
Display summary to user + show chunk count
```

### Querying:
```
User asks question in chatbot
    ↓
Question is embedded
    ↓
Vector search retrieves top-K chunks from ALL sources:
  - Uploaded files
  - Website content
  - Research papers
  - AI summaries
    ↓
Gemini AI generates answer using all relevant context
    ↓
User gets comprehensive answer
```

## Example Usage

### 1. Analyze a Website:
1. Go to `query.html`
2. Enter professor's website URL: `https://example.edu/~professor`
3. Click "Analyze Website"
4. System will:
   - Show you the summary
   - Display research papers
   - Store everything in your knowledge base
   - Show: "✓ Stored 45 chunks (Homepage: 12, Papers: 30, Summary: 3)"

### 2. Query the Chatbot:
**You:** "What research areas does this professor focus on?"

**Bot:** *Searches across website content + papers + summary and provides comprehensive answer*

**You:** "Tell me more about their work on machine learning"

**Bot:** *Retrieves relevant chunks from papers and website, generates detailed response*

## Metadata Tracking

Each stored chunk includes metadata:
- `source_name`: Where it came from (URL, paper title, etc.)
- `type`: Content type (`homepage`, `research_paper`, `ai_summary`)
- `url`: Original URL (for website content)
- `title`: Paper title (for research papers)
- `chunk_index`: Position in original document

## Files Modified

### Backend:
- ✅ `backend/api/router_scrape.py`
  - Added `chunk_text()` function
  - Added `embed_and_store_content()` function
  - Updated `/analyze` endpoint to chunk and store all content
  - Added chunk count tracking in response

### Frontend:
- ✅ `frontend/script.js`
  - Pass `user_id` to scrape endpoint
  - Display chunk storage confirmation to user

## Benefits

1. **No More Silos**: Website data and uploaded files are in the same knowledge base
2. **Comprehensive Answers**: Chatbot can reference website content and papers together
3. **Context Persistence**: Website analysis stays in your knowledge base for future queries
4. **Metadata Rich**: Every chunk knows where it came from
5. **Efficient Search**: Vector similarity works across all content types

## Testing

### Test the Integration:
1. Analyze a professor's website
2. Note the chunk counts displayed
3. Ask chatbot: "What is this professor's research about?"
4. Upload a related PDF
5. Ask: "How does this document relate to the professor's work?"
6. Bot should reference both sources!

## Database Structure

All content is stored in the same `documents` table:
```sql
{
  doc_id: uuid,
  user_id: uuid,
  content: text,
  embedding: vector(384),
  metadata: jsonb {
    source_name: "URL or filename",
    type: "homepage | research_paper | ai_summary | upload",
    url: "...",
    title: "...",
    chunk_index: 0
  },
  source: "website_scrape | google_scholar | upload"
}
```

## Next Steps

- ✅ Add source attribution in chatbot responses (show which sources were used)
- ✅ Add ability to delete/clear website data
- ✅ Add filtering by source type in queries
- ✅ Add conversation memory across sessions
