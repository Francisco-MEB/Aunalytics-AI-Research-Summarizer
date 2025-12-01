"""
Test script for the professor scraper
Demonstrates how the scraping system works
"""
from scraper import ProfessorScraper


def test_scraper_basic():
    """
    Basic test of the scraper functionality
    Shows what data gets extracted and how it flows into your system
    """
    print("\n" + "="*70)
    print("SCRAPER TEST - Understanding the Workflow")
    print("="*70 + "\n")
    
    # Initialize the scraper
    scraper = ProfessorScraper()
    
    # Example test URLs (these are just examples - replace with real ones)
    test_urls = [
        "https://www.cs.princeton.edu/~arvindn/",  # Example CS professor
        # Add more test URLs here
    ]
    
    print("This scraper does the following:\n")
    print("1. Takes a professor's website URL")
    print("2. Searches the page for links to:")
    print("   - Google Scholar (primary)")
    print("   - Academia.edu (fallback 1)")
    print("   - ResearchGate (fallback 2)")
    print("3. Scrapes the top 10 research papers")
    print("4. Extracts: title, authors, year, abstract, URL")
    print("5. Returns structured data for embedding\n")
    
    print("="*70)
    print("HOW IT INTEGRATES WITH YOUR SYSTEM")
    print("="*70 + "\n")
    
    print("Flow:")
    print("Frontend -> POST /api/scrape with professor URL")
    print("    |")
    print("    v")
    print("Scraper extracts papers + abstracts")
    print("    |")
    print("    v")
    print("chunk_text() breaks abstracts into chunks")
    print("    |")
    print("    v")
    print("embed_chunks() creates vectors")
    print("    |")
    print("    v")
    print("store_to_supabase() saves to database with user_id")
    print("    |")
    print("    v")
    print("User can now ask questions about the papers!")
    print("\n")
    
    print("="*70)
    print("WHAT THE SCRAPER LOOKS FOR")
    print("="*70 + "\n")
    
    print("On professor's website:")
    print("- <a> tags with keywords: 'google', 'scholar', 'citations'")
    print("- Links containing 'scholar.google.com'")
    print("- Links to 'academia.edu' or 'researchgate.net'")
    print("- Publication sections with paper titles\n")
    
    print("On Google Scholar page:")
    print("- Paper titles in class 'gsc_a_at'")
    print("- Authors and publication info")
    print("- Year published")
    print("- Link to paper detail page\n")
    
    print("On paper detail page:")
    print("- Abstract text (typically in specific div/section)")
    print("- Full paper metadata\n")
    
    print("="*70)
    print("ANTI-SCRAPING CONSIDERATIONS")
    print("="*70 + "\n")
    
    print("Google Scholar has anti-bot measures:")
    print("- Includes User-Agent header to look like a browser")
    print("- Adds 0.5 second delay between requests")
    print("- For production, may need:")
    print("  - Rotating proxies")
    print("  - Google Scholar API (if available)")
    print("  - Cached results\n")
    
    print("Alternative approach:")
    print("- Use Semantic Scholar API (free, no scraping needed)")
    print("- Use CrossRef API for paper metadata")
    print("- Use arXiv API for preprints\n")
    
    print("="*70)
    print("API ENDPOINT USAGE")
    print("="*70 + "\n")
    
    print("Frontend JavaScript example:")
    print('''
const response = await fetch('http://your-api.com/api/scrape', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        professor_url: 'https://example.edu/professor',
        user_id: 'user-uuid-here',
        max_papers: 10
    })
});

const data = await response.json();
// data.papers = [{title, authors, year, abstract, url}, ...]
// data.chunks_created = 150 (total chunks embedded)
// data.source = "Google Scholar" (where papers came from)
    ''')
    
    print("\n" + "="*70)
    print("TESTING WITH REAL URL")
    print("="*70 + "\n")
    
    print("To test with a real professor:")
    print("1. Find a university professor's webpage")
    print("2. Make sure they have a Google Scholar link")
    print("3. Run: python test_scraper.py <their-url>")
    print("4. Check the output to see what gets scraped\n")
    
    print("Example URLs to try:")
    print("- CS professors at major universities")
    print("- Look for 'Google Scholar' link on their homepage")
    print("- Make sure they have publications listed\n")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # User provided a URL to test
        test_url = sys.argv[1]
        print(f"\nTesting scraper with: {test_url}\n")
        
        scraper = ProfessorScraper()
        result = scraper.scrape_professor_website(test_url, max_papers=5)
        
        print("\n" + "="*70)
        print("SCRAPE RESULTS")
        print("="*70)
        print(f"\nSource: {result['source']}")
        print(f"Papers found: {len(result['papers'])}")
        
        if result['error']:
            print(f"Error: {result['error']}")
        
        for i, paper in enumerate(result['papers'], 1):
            print(f"\n[{i}] {paper['title']}")
            print(f"    Authors: {paper['authors']}")
            print(f"    Year: {paper['year']}")
            print(f"    Abstract: {paper['abstract'][:150]}...")
    else:
        # No URL provided, show explanation
        test_scraper_basic()
