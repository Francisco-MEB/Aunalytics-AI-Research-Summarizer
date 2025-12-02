# backend/api/router_scrape.py

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


# ------------------------------
# Helper: Extract homepage text
# ------------------------------
def extract_text(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        return soup.get_text(" ", strip=True)
    except:
        return ""


# ------------------------------
# Helper: Find Google Scholar link
# ------------------------------
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
    try:
        url = data["url"]
        user_id = data.get("user_id", "default")
        
        print(f"\n{'='*70}")
        print(f"SCRAPING WEBSITE")
        print(f"{'='*70}")
        print(f"URL: {url}")
        print(f"User ID: {user_id}")
        print(f"{'='*70}\n")

        # 1. Extract homepage text
        print("Step 1: Extracting homepage text...")
        homepage_text = extract_text(url)
        print(f"Extracted {len(homepage_text)} characters")
        
        # 2. Chunk and store homepage content
        homepage_chunks = 0
        if homepage_text:
            print("Step 2: Storing homepage chunks...")
            homepage_chunks = embed_and_store_content(
                text=homepage_text,
                user_id=user_id,
                source_name=url,
                metadata={"url": url, "type": "homepage"}
            )
            print(f"Stored {homepage_chunks} chunks from homepage")

        # 3. Find and process Google Scholar papers
        print("Step 3: Finding Google Scholar papers...")
        scholar_url = find_scholar_link(url)
        papers = []
        if scholar_url:
            author_id = extract_author_id(scholar_url)
            if author_id:
                print(f"Found author ID: {author_id}")
                papers = get_scholar_papers(author_id)
                print(f"Found {len(papers)} papers")

        # 4. Store each paper's abstract as chunks
        paper_chunks = 0
        for p in papers:
            text = p["description"]
            title = p["title"]

            if not text:
                continue

            chunks_stored = embed_and_store_content(
                text=text,
                user_id=user_id,
                source_name=f"Paper: {title}",
                metadata={"title": title, "type": "research_paper"}
            )
            paper_chunks += chunks_stored

        print(f"Stored {paper_chunks} chunks from {len(papers)} papers")

        # 5. Generate summary using RAG
        print("Step 5: Generating RAG summary...")
        rag_summary = rag_summarize(homepage_text, user_id=user_id, k=8)
        print(f"Generated summary: {len(rag_summary) if rag_summary else 0} characters")
        
        # 6. Store the summary itself as a document
        summary_chunks = 0
        if rag_summary:
            print("Step 6: Storing summary chunks...")
            summary_chunks = embed_and_store_content(
                text=rag_summary,
                user_id=user_id,
                source_name=f"Summary of {url}",
                metadata={"url": url, "type": "ai_summary"}
            )
            print(f"Stored {summary_chunks} chunks from summary")

        print(f"\n{'='*70}")
        print("SCRAPING COMPLETE")
        print(f"{'='*70}\n")

        return {
            "summary": rag_summary,
            "papers": papers,
            "chunks_stored": {
                "homepage": homepage_chunks,
                "papers": paper_chunks,
                "summary": summary_chunks
            }
        }
    
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR IN SCRAPE ENDPOINT")
        print(f"{'='*70}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
        
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))
