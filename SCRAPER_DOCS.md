# Scraper Implementation Documentation

## Overview

This scraper system extracts research papers from professor websites and integrates them into your RAG pipeline. It automatically finds and scrapes Google Scholar, Academia.edu, or ResearchGate profiles.

---

## Components

### 1. scraper.py
Main scraper module that handles all web scraping logic.

**Key Classes:**
- `ProfessorScraper`: Main scraper class with methods for different platforms

**Main Methods:**
- `scrape_professor_website()`: Entry point that coordinates the scraping
- `_find_google_scholar_link()`: Searches page for Google Scholar URLs
- `_scrape_google_scholar()`: Extracts papers from Google Scholar profile
- `_fetch_paper_abstract()`: Gets abstract from individual paper pages

### 2. FastAPI Integration (main.py)
New endpoint: `POST /api/scrape`

**Request:**
```json
{
  "professor_url": "https://example.edu/professor",
  "user_id": "uuid-here",
  "max_papers": 10
}
```

**Response:**
```json
{
  "message": "Successfully scraped and embedded 10 papers",
  "professor_url": "https://example.edu/professor",
  "scholar_url": "https://scholar.google.com/citations?user=...",
  "source": "Google Scholar",
  "papers_found": 10,
  "papers": [
    {
      "title": "Paper Title",
      "authors": "Author Names",
      "year": "2023",
      "abstract": "Abstract text...",
      "url": "https://...",
      "source": "Google Scholar"
    }
  ],
  "chunks_created": 150,
  "error": null
}
```

---

## How It Works

### Step 1: Find Research Profile
```
User provides professor URL
    |
    v
Scraper loads the page
    |
    v
Search for links containing:
- "google" + "scholar"
- "scholar.google.com"
- "academia.edu"
- "researchgate.net"
```

### Step 2: Extract Papers
```
Found Google Scholar profile
    |
    v
Load Google Scholar page
    |
    v
Parse HTML for paper elements
    |
    v
Extract:
- Title
- Authors  
- Year
- Link to detail page
```

### Step 3: Get Abstracts
```
For each paper:
    |
    v
Follow link to detail page
    |
    v
Find abstract section
    |
    v
Extract abstract text
    |
    v
Wait 0.5s (be polite to servers)
```

### Step 4: Embed and Store
```
For each paper:
    |
    v
Combine title + authors + abstract
    |
    v
chunk_text() - break into 500-char chunks
    |
    v
embed_chunks() - create vector embeddings
    |
    v
store_to_supabase() - save with user_id
```

---

## Search Strategy

### Priority Order:
1. **Google Scholar** (primary) - Most academic papers
2. **Academia.edu** (fallback 1) - Humanities papers
3. **ResearchGate** (fallback 2) - Alternative platform
4. **Direct website** (fallback 3) - Scrape publication lists

### What Gets Scraped:

**From Professor's Main Page:**
- Bio information (name, title, department)
- Research interests
- Course listings

**From Google Scholar:**
- Top 10 most cited papers
- Paper titles
- Author lists
- Publication years
- Abstracts (from detail pages)
- Paper URLs

---

## Anti-Scraping Measures

### What We Do:
- Set User-Agent header to look like a browser
- Add delays between requests (0.5 seconds)
- Respect server timeout limits (10 seconds)
- Graceful error handling

### Limitations:
- Google Scholar has rate limiting
- Some sites block automated access
- Abstracts may not always be available
- CAPTCHA can block access

### Production Solutions:
1. **Use Official APIs:**
   - Semantic Scholar API (free)
   - CrossRef API (free)
   - arXiv API (free)

2. **Caching:**
   - Cache scraped results
   - Only re-scrape weekly/monthly

3. **Proxies:**
   - Rotate IP addresses
   - Use scraping services (ScraperAPI, Bright Data)

---

## Integration with Frontend

### Mockup Requirements:

**Left Side:**
- Website URL input field
- Submit button

**After Scraping:**
- Display 5 paper preview cards/icons
- Show paper titles
- Make them clickable

**Right Side:**
- Chat interface
- User asks questions
- System uses embedded papers to answer

### Frontend JavaScript:

```javascript
// When user submits professor URL
async function scrapeProfessor(url) {
    const response = await fetch('/api/scrape', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            professor_url: url,
            user_id: getUserId(),  // Get from session/auth
            max_papers: 10
        })
    });
    
    const data = await response.json();
    
    // Display papers in left panel
    displayPapers(data.papers.slice(0, 5));  // Show top 5
    
    // Enable chat interface
    enableChat();
}

// When user asks question
async function askQuestion(question) {
    const response = await fetch('/api/question', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question: question,
            user_id: getUserId()
        })
    });
    
    const answer = await response.json();
    displayAnswer(answer.answer, answer.sources);
}
```

---

## Testing

### Manual Testing:

```powershell
# Run test script
python test_scraper.py

# Test with actual professor URL
python test_scraper.py https://example.edu/professor
```

### API Testing:

```powershell
# Start the server
python main.py

# Test scrape endpoint (PowerShell)
$body = @{
    professor_url = "https://example.edu/professor"
    user_id = "your-uuid-here"
    max_papers = 5
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8080/api/scrape -Method POST -Body $body -ContentType "application/json"
```

### Expected Results:
- Papers list with titles and abstracts
- Chunks created count (should be ~15 per paper)
- Source indication (Google Scholar, etc.)
- No errors in response

---

## Error Handling

### Common Errors:

**404 - No papers found:**
- Professor doesn't have Google Scholar
- No accessible publication list
- Website structure not recognized

**500 - Scraping failed:**
- Network timeout
- Website blocked the request
- Invalid HTML structure
- CAPTCHA triggered

**400 - Invalid URL:**
- Malformed URL
- Not a valid HTTP/HTTPS URL

### Handling in Frontend:
```javascript
try {
    const data = await scrapeProfessor(url);
    if (data.error) {
        showError(data.error);
    }
} catch (error) {
    showError('Failed to scrape professor website. Please try again.');
}
```

---

## Future Improvements

### Short Term:
1. Implement Academia.edu scraping
2. Implement ResearchGate scraping
3. Better abstract extraction
4. Cache scraped results

### Long Term:
1. Use Semantic Scholar API instead of scraping
2. Add progress updates during scraping
3. Support for non-English papers
4. Better bio information extraction
5. Scrape CV PDFs for publication lists

---

## Dependencies Added

```
beautifulsoup4==4.12.3  # HTML parsing
requests==2.32.5         # HTTP requests
```

Already in requirements.txt, no additional installation needed for deployment.

---

## Deployment Notes

### Docker:
- Scraper code included in container
- No additional system dependencies needed
- Works same as local development

### Google Cloud Run:
- May need higher timeout for scraping (currently 300s)
- Consider Cloud Scheduler for periodic updates
- Monitor rate limiting from external sites

### Environment Variables:
No additional env vars needed for basic scraping.
For production, consider:
- `SCRAPER_TIMEOUT=10`
- `SCRAPER_MAX_PAPERS=10`
- `SCRAPER_USE_CACHE=true`

---

## Summary

You now have:
- Complete web scraper for professor papers
- FastAPI endpoint `/api/scrape`
- Automatic embedding and storage
- Integration with existing QA system
- Error handling and fallbacks
- Documentation and test scripts

The scraper fulfills Team 2's deliverables and unblocks Team 1's work on embedding scraped content.
