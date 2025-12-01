"""
Web scraper for professor research papers
Scrapes Google Scholar, Academia.edu, and ResearchGate for paper abstracts
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import re
import time
from urllib.parse import urljoin, urlparse


class ProfessorScraper:
    """
    Scrapes professor websites to find research papers and abstracts
    Supports Google Scholar, Academia.edu, and ResearchGate
    """
    
    def __init__(self):
        # Set up headers to mimic a browser (some sites block scrapers)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.timeout = 10  # seconds
    
    def scrape_professor_website(self, url: str, max_papers: int = 10) -> Dict:
        """
        Main entry point: scrapes a professor's website for research papers
        
        Args:
            url: Professor's website URL
            max_papers: Maximum number of papers to retrieve (default 10)
        
        Returns:
            Dictionary with professor info and list of papers
        """
        print(f"[SCRAPER] Starting scrape of: {url}")
        
        result = {
            "professor_url": url,
            "scholar_url": None,
            "papers": [],
            "bio_info": None,
            "source": None,
            "error": None
        }
        
        try:
            # Step 1: Get the professor's main page
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Step 2: Extract bio/info from main page
            result["bio_info"] = self._extract_bio_info(soup)
            
            # Step 3: Find Google Scholar link
            scholar_url = self._find_google_scholar_link(soup, url)
            
            if scholar_url:
                print(f"[SCRAPER] Found Google Scholar: {scholar_url}")
                result["scholar_url"] = scholar_url
                result["source"] = "Google Scholar"
                result["papers"] = self._scrape_google_scholar(scholar_url, max_papers)
            
            # Fallback 1: Try Academia.edu
            if not result["papers"]:
                print("[SCRAPER] No Google Scholar, trying Academia.edu...")
                academia_url = self._find_academia_link(soup, url)
                if academia_url:
                    result["source"] = "Academia.edu"
                    result["papers"] = self._scrape_academia(academia_url, max_papers)
            
            # Fallback 2: Try ResearchGate
            if not result["papers"]:
                print("[SCRAPER] No Academia.edu, trying ResearchGate...")
                rg_url = self._find_researchgate_link(soup, url)
                if rg_url:
                    result["source"] = "ResearchGate"
                    result["papers"] = self._scrape_researchgate(rg_url, max_papers)
            
            # Fallback 3: Try scraping papers directly from professor's site
            if not result["papers"]:
                print("[SCRAPER] No external profiles, scraping main site...")
                result["source"] = "Professor Website"
                result["papers"] = self._scrape_local_papers(soup, url, max_papers)
            
            print(f"[SCRAPER] Complete. Found {len(result['papers'])} papers from {result['source']}")
            
        except requests.RequestException as e:
            result["error"] = f"Failed to fetch URL: {str(e)}"
            print(f"[SCRAPER ERROR] {result['error']}")
        except Exception as e:
            result["error"] = f"Scraping error: {str(e)}"
            print(f"[SCRAPER ERROR] {result['error']}")
        
        return result
    
    def _find_google_scholar_link(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """
        Search page for Google Scholar profile link
        Looks for anchor tags with keywords: google, scholar, citations
        """
        # Keywords that indicate a Google Scholar link
        keywords = ['google', 'scholar', 'citations']
        
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            link_text = link.get_text().lower()
            
            # Check if href or link text contains keywords
            if any(keyword in href for keyword in keywords) or any(keyword in link_text for keyword in keywords):
                # Build full URL
                full_url = urljoin(base_url, link['href'])
                if 'scholar.google' in full_url:
                    return full_url
        
        return None
    
    def _find_academia_link(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """Find Academia.edu profile link"""
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'academia.edu' in href.lower():
                return urljoin(base_url, href)
        return None
    
    def _find_researchgate_link(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """Find ResearchGate profile link"""
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'researchgate.net' in href.lower():
                return urljoin(base_url, href)
        return None
    
    def _extract_bio_info(self, soup: BeautifulSoup) -> Dict:
        """
        Extract biographical information from professor's main page
        Looks for: name, title, department, research interests, courses
        """
        bio = {
            "name": None,
            "title": None,
            "department": None,
            "research_interests": None,
            "courses": []
        }
        
        # Try to find name from h1 or title
        h1 = soup.find('h1')
        if h1:
            bio["name"] = h1.get_text().strip()
        
        # Look for common professor info patterns
        text_content = soup.get_text()
        
        # Search for research interests section
        if 'research interest' in text_content.lower():
            # This is a simplified extraction - would need more sophisticated parsing
            bio["research_interests"] = "Found research interests section"
        
        return bio
    
    def _scrape_google_scholar(self, scholar_url: str, max_papers: int) -> List[Dict]:
        """
        Scrape Google Scholar profile for papers
        Note: Google Scholar has anti-scraping measures, this is a basic implementation
        """
        papers = []
        
        try:
            response = requests.get(scholar_url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Google Scholar structure: papers are in divs with class 'gsc_a_tr'
            paper_elements = soup.find_all('tr', class_='gsc_a_tr')[:max_papers]
            
            for idx, paper_elem in enumerate(paper_elements):
                try:
                    # Extract title
                    title_elem = paper_elem.find('a', class_='gsc_a_at')
                    title = title_elem.get_text() if title_elem else f"Paper {idx+1}"
                    
                    # Get paper URL
                    paper_url = urljoin(scholar_url, title_elem['href']) if title_elem and title_elem.get('href') else None
                    
                    # Extract authors and publication info
                    authors_elem = paper_elem.find('div', class_='gs_gray')
                    authors = authors_elem.get_text() if authors_elem else "Unknown"
                    
                    # Extract year
                    year_elem = paper_elem.find('span', class_='gsc_a_h')
                    year = year_elem.get_text() if year_elem else "N/A"
                    
                    # Try to get abstract by following paper link
                    abstract = self._fetch_paper_abstract(paper_url) if paper_url else "Abstract not available"
                    
                    papers.append({
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "abstract": abstract,
                        "url": paper_url,
                        "source": "Google Scholar"
                    })
                    
                    print(f"[SCRAPER] Scraped paper {idx+1}: {title[:50]}...")
                    
                    # Be polite - don't hammer the server
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"[SCRAPER WARNING] Failed to parse paper {idx+1}: {e}")
                    continue
        
        except Exception as e:
            print(f"[SCRAPER ERROR] Google Scholar scraping failed: {e}")
        
        return papers
    
    def _scrape_academia(self, academia_url: str, max_papers: int) -> List[Dict]:
        """
        Scrape Academia.edu profile
        Academia.edu structure varies, this is a basic implementation
        """
        papers = []
        print("[SCRAPER] Academia.edu scraping not fully implemented yet")
        # TODO: Implement Academia.edu scraping based on their HTML structure
        return papers
    
    def _scrape_researchgate(self, rg_url: str, max_papers: int) -> List[Dict]:
        """
        Scrape ResearchGate profile
        ResearchGate has strict anti-scraping, may require API access
        """
        papers = []
        print("[SCRAPER] ResearchGate scraping not fully implemented yet")
        # TODO: Implement ResearchGate scraping or use their API
        return papers
    
    def _scrape_local_papers(self, soup: BeautifulSoup, base_url: str, max_papers: int) -> List[Dict]:
        """
        Try to scrape papers directly from professor's website
        Looks for publication lists, CV links, etc.
        """
        papers = []
        
        # Look for common publication section keywords
        pub_keywords = ['publication', 'paper', 'research', 'article', 'journal']
        
        # Search for sections that might contain papers
        for section in soup.find_all(['section', 'div', 'article']):
            section_text = section.get_text().lower()
            
            # Check if this section is about publications
            if any(keyword in section_text for keyword in pub_keywords):
                # Look for links within this section
                links = section.find_all('a', href=True)
                
                for link in links[:max_papers]:
                    title = link.get_text().strip()
                    if len(title) > 10:  # Probably a paper title if it's long enough
                        papers.append({
                            "title": title,
                            "authors": "Unknown",
                            "year": "N/A",
                            "abstract": "Abstract not available from local scraping",
                            "url": urljoin(base_url, link['href']),
                            "source": "Professor Website"
                        })
                
                if papers:
                    break  # Found papers, stop looking
        
        return papers[:max_papers]
    
    def _fetch_paper_abstract(self, paper_url: str) -> str:
        """
        Fetch abstract from a paper's detail page
        This is called for each individual paper
        """
        if not paper_url:
            return "Abstract not available"
        
        try:
            response = requests.get(paper_url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Google Scholar abstracts are in div with id 'gsc_vcd_tse'
            abstract_elem = soup.find('div', id='gsc_vcd_tse')
            if abstract_elem:
                return abstract_elem.get_text().strip()
            
            # Fallback: look for common abstract sections
            for section in soup.find_all(['div', 'section', 'p']):
                text = section.get_text().lower()
                if 'abstract' in text:
                    return section.get_text().strip()[:500]  # Limit to 500 chars
            
            return "Abstract not found on detail page"
        
        except Exception as e:
            print(f"[SCRAPER WARNING] Failed to fetch abstract: {e}")
            return "Failed to fetch abstract"


def test_scraper():
    """Test function to verify scraper works"""
    scraper = ProfessorScraper()
    
    # Test with a sample professor URL
    test_url = "https://example.edu/professor"
    result = scraper.scrape_professor_website(test_url, max_papers=5)
    
    print("\n=== SCRAPER TEST RESULTS ===")
    print(f"Source: {result['source']}")
    print(f"Papers found: {len(result['papers'])}")
    if result['error']:
        print(f"Error: {result['error']}")
    
    return result


if __name__ == "__main__":
    test_scraper()
