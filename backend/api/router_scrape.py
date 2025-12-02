import os
import uuid
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from fastapi import APIRouter
from dotenv import load_dotenv
from supabase import create_client
from scholarly import scholarly
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
import google.generativeai as genai   # GEMINI SUMMARIZER

load_dotenv()

router = APIRouter()

# --- Initialize models + Supabase ---
model_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))

# Gemini init
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-2.0-flash")


# ================================================================
# Extract text from webpage
# ================================================================
def extract_text(url: str):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except:
        return ""


# ================================================================
# Summarize homepage content
# ================================================================
def summarize_with_gemini(text: str) -> str:
    if not text:
        return "No summary available."

    try:
        prompt = f"""
        Summarize the following professor homepage content into a clean,
        readable, academic summary focusing on:
        - research fields
        - expertise
        - notable contributions
        - lab description
        - what the professor studies
    
        Rewrite the following text as a single smooth academic paragraph.
        Do NOT use headings. 
        Do NOT use bullets. 
        Do NOT bold anything. 
        Do NOT format anything. 
        Just use plain text. 
        Blend the information naturally into prose.

        Remove:
        - menus
        - headers/footers
        - contact info

        Text:
        {text}
        """

        response = gemini.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print("Gemini summary error:", e)
        return text[:600] + "..."


# ================================================================
# Detect direct Google Scholar link
# ================================================================
def getScholar(websiteUrl):
    try:
        response = requests.get(websiteUrl, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("a"):
            href = link.get("href")
            if href and "scholar.google" in href.lower():
                return href
    except:
        pass

    return None


# ================================================================
# Validate a name
# ================================================================
def is_valid_name(name: str):
    if not name or len(name) > 50 or len(name) < 4:
        return False

    invalid = ["menu", "about", "contact", "header", "footer", "main"]
    if any(t in name.lower() for t in invalid):
        return False

    if len(name.split()) < 2:
        return False

    return bool(re.match(r"^[A-Za-z\s\-.]+$", name))


# ================================================================
# Extract professor name
# ================================================================
def extract_professor_name(websiteUrl):
    try:
        response = requests.get(websiteUrl, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
    except:
        return None

    # From URL path
    for part in websiteUrl.split("/"):
        if (
            part and
            "edu" not in part and
            "." not in part and
            len(part) > 2
        ):
            candidate = part.replace("~", "").replace("-", " ").title()
            if is_valid_name(candidate):
                return candidate

    # From H1/H2
    for tag in ["h1", "h2"]:
        el = soup.find(tag)
        if el:
            name = el.get_text(strip=True)
            if is_valid_name(name):
                return name

    return None


# ================================================================
# Extract Scholar ID
# ================================================================
def extractAuthor(scholarUrl):
    try:
        parsed = urlparse(scholarUrl)
        params = parse_qs(parsed.query)
        return params["user"][0]
    except:
        return None


# ================================================================
# Search scholar by name
# ================================================================
def search_scholar_by_name(name, limit=3):
    try:
        query = scholarly.search_author(name)
        for _ in range(limit):
            match = next(query, None)
            if match:
                return match["scholar_id"]
    except:
        pass
    return None


# ================================================================
# Playwright fallback
# ================================================================
def playwrightSearch(name):
    if not name:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("https://www.bing.com")

            page.fill("input[name='q']", f"{name} Google Scholar")
            page.keyboard.press("Enter")
            page.wait_for_selector(".b_algo", timeout=5000)

            items = page.locator(".b_algo")
            for i in range(items.count()):
                text = items.nth(i).inner_text().lower()
                if "scholar.google" in text:
                    link = items.nth(i).locator("h2 a").first.get_attribute("href")
                    browser.close()
                    return link

            browser.close()
    except:
        pass
    return None


# ================================================================
# Scholar descriptions/abstracts
# ================================================================
def get_scholar_descriptions(authorId):
    try:
        author = scholarly.search_author_id(authorId)
        scholarly.fill(author, sections=["publications"])

        descriptions = []
        pubs = author.get("publications", [])
        for pub in pubs[:10]:
            scholarly.fill(pub)

            desc = pub["bib"].get("abstract") or pub["bib"].get("description")
            title = pub["bib"].get("title", "Untitled")

            if not desc:
                desc = "No description available."

            descriptions.append({
                "title": title,
                "description": desc
            })

        return descriptions

    except Exception as e:
        print("Scholar description error:", e)
        return []


# ================================================================
# ⭐ MAIN SCRAPE ROUTE ⭐
# ================================================================
@router.post("/analyze")
async def analyze(data: dict):

    url = data["url"]
    user_id = data.get("user_id", None)

    homepage_text = extract_text(url)
    summary = summarize_with_gemini(homepage_text)

    scholar_url = getScholar(url)
    author_id = None

    if scholar_url:
        author_id = extractAuthor(scholar_url)
    else:
        name = extract_professor_name(url)
        if name:
            author_id = search_scholar_by_name(name)
        if not author_id:
            scholar_url = playwrightSearch(name)
            if scholar_url:
                author_id = extractAuthor(scholar_url)

    if not author_id:
        return {
            "summary": summary,
            "papers": [],
            "error": "Unable to locate Google Scholar profile."
        }

    papers = get_scholar_descriptions(author_id)

    # Insert into Supabase
    for p in papers:
        text = p["description"]
        title = p["title"]

        chunk_id = str(uuid.uuid4())
        doc_id = str(uuid.uuid4())
        embedding = model_embedder.encode([text])[0].tolist()

        supabase.table("documents").insert({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "user_id": user_id,
            "content": text,
            "metadata": {"title": title},
            "source": "google_scholar",
            "embedding": embedding
        }).execute()

    return {
        "summary": summary,
        "papers": papers
    }


# ================================================================
# ⭐ SUMMARIZE A PAPER (CLICK EVENT) ⭐
# ================================================================
@router.post("/summarize-paper")
async def summarize_paper(data: dict):
    text = data.get("text", "")

    if not text:
        return { "summary": "No content to summarize." }

    try:
        prompt = f"""
        Rewrite the following text as a single smooth, readable academic paragraph. 
        Do NOT use section headers. Do NOT use bullet points. Do NOT use bold or markdown or italics; just use plain text. 
        Do NOT list information — instead blend it into prose. 
        Preserve only the meaning and relevant research-related details.

        Text to summarize:
        {text}
        """
        result = gemini.generate_content(prompt)
        return { "summary": result.text.strip() }

    except Exception as e:
        print("Paper summary error:", e)
        return { "summary": text[:600] + "..." }
