from bs4 import BeautifulSoup
import requests
from scholarly import scholarly
from urllib.parse import urlparse, parse_qs

def scrape_professor_website(url: str):
    """
    Returns:
    {
        "name": "...",
        "abstracts": ["...", "..."]
    }
    """

    # fetch professor site
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")

    # find scholar link
    scholar_url = None
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if "scholar.google" in href:
            scholar_url = href
            break

    if not scholar_url:
        return None

    # extract Google Scholar user ID
    parsed = urlparse(scholar_url)
    qs = parse_qs(parsed.query)
    if "user" not in qs:
        return None

    scholar_id = qs["user"][0]

    # fetch scholarly profile
    author = scholarly.search_author_id(scholar_id)
    scholarly.fill(author, sections=["publications"])

    abstracts = []
    for pub in author["publications"][:10]:
        scholarly.fill(pub)
        abs = pub["bib"].get("abstract")
        if abs:
            abstracts.append(abs)

    return {
        "name": author.get("name", "Unknown"),
        "abstracts": abstracts
    }
