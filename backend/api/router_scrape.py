import os
import uuid
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from fastapi import APIRouter
from dotenv import load_dotenv
from supabase import create_client
from scholarly import scholarly
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from backend.core.summarizer import rag_summarize  # <-- NEW IMPORT

load_dotenv()

router = APIRouter()

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-2.0-flash")

def extract_text(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        return soup.get_text(" ", strip=True)
    except:
        return ""
    

def find_scholar_link(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        for a in soup.find_all("a"):
            href = (a.get("href") or "").strip()
            if "scholar.google" in href:
                if href.startswith("http"):
                    return href
                return "https://scholar.google.com" + href
    except:
        return None
    return None


def extract_author_id(url):
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        return qs.get("user", [None])[0]
    except:
        return None


def get_scholar_papers(author_id):
    try:
        author = scholarly.search_author_id(author_id)
        scholarly.fill(author, sections=["publications"])
        pubs = author.get("publications", [])

        out = []
        for pub in pubs[:10]:
            scholarly.fill(pub)
            title = pub["bib"].get("title", "Untitled")
            desc = (
                pub["bib"].get("abstract")
                or pub["bib"].get("description")
                or ""
            )
            out.append({"title": title, "description": desc})
        return out

    except Exception as e:
        print("Scholar error:", e)
        return []





@router.post("/analyze")
async def analyze(data: dict):
    url = data["url"]
    user_id = data.get("user_id", "default")

  
    homepage_text = extract_text(url)

   
    scholar_url = find_scholar_link(url)
    papers = []
    if scholar_url:
        author_id = extract_author_id(scholar_url)
        if author_id:
            papers = get_scholar_papers(author_id)

    
    for p in papers:
        text = p["description"]
        title = p["title"]

        if not text:
            continue

        vec = embedder.encode([text])[0].tolist()

        supabase.table("documents").insert({
            "chunk_id": str(uuid.uuid4()),
            "doc_id": str(uuid.uuid4()),
            "user_id": user_id,
            "content": text,
            "metadata": {"title": title},
            "source": "google_scholar",
            "embedding": vec
        }).execute()
    
    rag_summary = rag_summarize(homepage_text, user_id=user_id, k=8)

    return {
        "summary": rag_summary,
        "papers": papers,
    }
