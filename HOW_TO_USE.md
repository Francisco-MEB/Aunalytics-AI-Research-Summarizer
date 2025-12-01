# How to Use the System Now

## What Just Got Fixed

### The Problem
- ResearchGate, ASEE, and other research sites **block automated scraping**
- This caused 403 Forbidden errors and crashes
- Users got confusing "500 Internal Server Error" messages

### The Solution
- **Better error handling**: 403 errors don't crash the system anymore
- **Multiple fallback strategies**: Tries Google Scholar → Academia → ResearchGate → Local site
- **Clearer error messages**: Users now see helpful guidance instead of "500 error"
- **Graceful degradation**: If one source fails, tries the next one

## How to Use the System

### ✅ OPTION 1: Upload Documents Directly (BEST OPTION)
1. Open `query.html` in your browser
2. Click "Upload Document" button
3. Select PDF, DOCX, or TXT files
4. Files are embedded and ready to query immediately

**This is the most reliable method!**

### ✅ OPTION 2: Use Google Scholar Directly
1. Find the professor's Google Scholar profile
2. Copy the Scholar URL (e.g., `https://scholar.google.com/citations?user=ABC123`)
3. Paste into the URL input
4. Click "Analyze Website"

### ⚠️ OPTION 3: Try Professor Website
1. Enter professor's personal website URL
2. Click "Analyze Website"
3. System will attempt to:
   - Find Google Scholar link on their page
   - Extract publication lists from their site
   - Scrape any found papers

**Note**: External research sites (ResearchGate, ASEE) will likely block this method.

## What Works & What Doesn't

### ✅ Works Great
- **File uploads** (PDF, DOCX, TXT)
- **Direct Google Scholar links** (if not blocked)
- **Asking questions** about already-uploaded documents
- **Summarization** of embedded papers

### ⚠️ May Work
- **Professor website scraping** (depends on site structure)
- **Local publication lists** (if website has them)

### ❌ Usually Blocked
- **ResearchGate** profiles (403 Forbidden)
- **Academia.edu** profiles (may be blocked)
- **ASEE Peer** paper links (403 Forbidden)
- **Direct paper URLs** from paywalled journals

## Current Server Status

**API Server**: Running on `http://localhost:8080`

**Endpoints**:
- `GET /` - Service info
- `GET /health` - Health check
- `POST /api/upload` - Upload documents (PDF/DOCX/TXT)
- `POST /api/scrape` - Scrape professor website
- `POST /api/question` - Ask questions
- `DELETE /api/documents/{user_id}` - Clear user documents

## Testing the System

### Test 1: Health Check
```bash
curl http://localhost:8080/health
```
Should return: `{"status":"healthy","database":"connected",...}`

### Test 2: Upload a Document
1. Open `frontend/query.html` in browser
2. Click "Upload Document"
3. Select a PDF or TXT file
4. Wait for "Document uploaded successfully"
5. Ask a question about it

### Test 3: Ask a Question
After uploading documents:
1. Type a question in the chat input
2. Press "SEND"
3. Get AI-generated answer based on your documents

## Browser Cache Issue (If You Still See Weird Text)

If you still see corrupted characters:
1. Open browser (Edge, Chrome, Firefox)
2. Press `Ctrl+Shift+Delete`
3. Select "Cached images and files"
4. Click "Clear data"
5. Reload `query.html`

OR use **Incognito/Private mode**:
- Chrome/Edge: `Ctrl+Shift+N`
- Firefox: `Ctrl+Shift+P`

## Next Steps

1. **Test file upload**: This is the most reliable feature
2. **Ask questions**: Once documents are embedded, queries work great
3. **Try different sources**: Google Scholar links when available
4. **Manual workflow**: Download papers as PDFs, upload them directly

## Why Research Sites Block Scraping

Research aggregation sites like ResearchGate, Academia.edu, and ASEE:
- Protect their business model (advertising, subscriptions)
- Prevent data mining and large-scale scraping
- Require user authentication for full access
- Use CAPTCHAs and rate limiting

**This is expected behavior** - most academic sites actively prevent bots.

## Future Improvements Possible

1. Use official APIs (Google Scholar API, CrossRef API)
2. Add Selenium for JavaScript-heavy sites
3. Implement proxy rotation
4. Add CAPTCHA solving
5. Use authentication for sites that allow API access

For now, **file upload is the best and most reliable method!**
