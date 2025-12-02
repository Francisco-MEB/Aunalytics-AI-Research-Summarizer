import os
import uuid
import requests
from typing import List, Dict
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from fastapi import APIRouter
from dotenv import load_dotenv
from supabase import create_client
from scholarly import scholarly
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

from backend.core.summarizer import rag_summarize

load_dotenv()

router = APIRouter()

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
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


# ------------------------------
# Helper: Chunk and embed text
# ------------------------------
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Dict]:
    """Split text into overlapping chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.create_documents([text], metadatas=[{}])
    return [{"text": d.page_content, "metadata": d.metadata} for d in docs]


def embed_and_store_content(text: str, user_id: str, source_name: str, metadata: dict = None):
    """Chunk text, generate embeddings, and store in Supabase."""
    if not text or not text.strip():
        return 0
    
    # Chunk the text
    chunks = chunk_text(text)
    
    # Generate embeddings for all chunks
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
    
    # Store each chunk
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata.update({
            "source_name": source_name,
            "chunk_index": i
        })
        
        supabase.table("documents").insert({
            "doc_id": str(uuid.uuid4()),
            "chunk_id": str(uuid.uuid4()),
            "user_id": user_id,
            "content": chunk["text"],
            "metadata": chunk_metadata,
            "source": "website_scrape",
            "embedding": vector.tolist()
        }).execute()
    
    return len(chunks)





@router.post("/analyze")
async def analyze(data: dict):
    url = data["url"]
    user_id = data.get("user_id", "default")
    
    print(f"\n{'='*70}")
    print(f"SCRAPING WEBSITE")
    print(f"{'='*70}")
    print(f"URL: {url}")
    print(f"User ID: {user_id}")
    print(f"{'='*70}\n")

    # 1. Extract homepage text
    homepage_text = extract_text(url)
    
    # 2. Chunk and store homepage content
    homepage_chunks = 0
    if homepage_text:
        homepage_chunks = embed_and_store_content(
            text=homepage_text,
            user_id=user_id,
            source_name=url,
            metadata={"url": url, "type": "homepage"}
        )
        print(f"Stored {homepage_chunks} chunks from homepage")

    # 3. Find and process Google Scholar papers
    scholar_url = find_scholar_link(url)
    papers = []
    if scholar_url:
        author_id = extract_author_id(scholar_url)
        if author_id:
            papers = get_scholar_papers(author_id)

    # 4. Store each paper's abstract as chunks
    paper_chunks = 0
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
    
    # 6. Store the summary itself as a document
    if rag_summary:
        summary_chunks = embed_and_store_content(
            text=rag_summary,
            user_id=user_id,
            source_name=f"Summary of {url}",
            metadata={"url": url, "type": "ai_summary"}
        )
        print(f"Stored {summary_chunks} chunks from summary")

    return {
        "summary": rag_summary,
        "papers": papers,
        "chunks_stored": {
            "homepage": homepage_chunks,
            "papers": paper_chunks,
            "summary": summary_chunks if rag_summary else 0
        }
    }
