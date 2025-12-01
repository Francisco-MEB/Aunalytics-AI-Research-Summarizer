# Scraper 403 Error Fix

## Problem Summary
The web scraper was failing with **403 Forbidden errors** when trying to access external research profile sites like:
- ResearchGate
- Academia.edu  
- ASEE Peer

These sites have anti-bot protection that blocks automated scraping attempts.

## Root Causes
1. **Anti-scraping protection**: Research sites actively block bots to protect their content
2. **Poor error handling**: 403 errors were causing the entire scrape to fail with 500 Internal Server Error
3. **No fallback strategy**: When one source failed, the scraper would crash instead of trying alternatives

## Solution Implemented

### 1. Improved Error Handling
- **Catch 403 errors specifically** instead of treating them as fatal errors
- **Continue to next fallback** when a site blocks us
- **Return 422 (Unprocessable Entity)** instead of 500 (Server Error) for scraping issues

### 2. Graceful Degradation
The scraper now tries multiple sources in order:
1. **Google Scholar** (best source, but also blocks bots)
2. **Academia.edu** (if Scholar fails or is blocked)
3. **ResearchGate** (if Academia fails or is blocked)
4. **Professor's website directly** (local scraping of publication lists)

If one source returns 403, it logs a warning and tries the next one.

### 3. Better User Feedback
Instead of generic "500 Internal Server Error", users now see:
- **404**: No papers found at all
- **422**: Papers couldn't be accessed due to blocking (with helpful message)
- **200**: Success with papers (even if some sources were blocked)

Error message now says:
```
No papers found. External research profile sites (ResearchGate, ASEE) are blocking 
automated access. Please provide direct links to papers or Google Scholar profile.
```

## Files Modified
- `scraper.py`: Added try-catch for 403 errors in fallback methods
- `main.py`: Changed HTTP status code from 500 to 422 for scraping errors
- `.env`: Added `API_PORT=8080` to avoid conflict with database port

## Testing
1. ✅ Server starts on correct port (8080)
2. ✅ 403 errors don't crash the scraper
3. ✅ User gets helpful error message instead of "500 Internal Server Error"
4. ⚠️ External sites still block automated access (this is expected behavior)

## Recommendations

### For Users
1. **Use Google Scholar links directly** if available
2. **Upload PDFs manually** for best results (File Upload feature works great!)
3. **Provide specific paper URLs** instead of profile pages

### Future Improvements
1. Implement **rate limiting and delays** to be more respectful to websites
2. Add **Selenium/Playwright** for sites that require JavaScript rendering
3. Use **official APIs** (Google Scholar API, ResearchGate API) when available
4. Add **proxy rotation** to avoid IP-based blocking
5. Implement **CAPTCHA solving** for sites that use it

## Why This Happens
Research aggregation sites like ResearchGate and ASEE:
- Want to protect their data from being scraped
- Require users to view papers through their platform (advertising revenue)
- Block automated access using User-Agent detection, rate limiting, and CAPTCHAs
- May require authentication (login) to access full content

**This is normal and expected behavior** - these sites don't want bots scraping them!
