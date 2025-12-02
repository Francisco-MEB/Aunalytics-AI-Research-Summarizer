import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from .retrieval import retrieve_chunks_rpc
import os

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-2.0-flash")

def build_rag_prompt(homepage_summary, retrieved_chunks):
    """Construct a clean RAG-enhanced prompt for Gemini."""

    context_block = "\n\n".join(
        f"- {c.get('content', '')}" for c in retrieved_chunks
    )

    prompt = f"""
You are generating a research overview for a university professor.
Use the homepage summary AND the retrieved scholarly paper abstracts.

---------------------
HOMEPAGE SUMMARY:
{homepage_summary}

---------------------
RELEVANT SCHOLARLY CONTEXT (retrieved via embeddings):
{context_block}
---------------------
Produce a SINGLE, smooth, academic summary paragraph describing:
- the professor's research areas
- themes & patterns across papers
- major contributions
- main domains or methods

Write in a professional academic tone.
Do NOT mention that you used retrieved chunks.
    """

    return prompt


def rag_summarize(homepage_text: str, user_id="default", k=8):
    """Full RAG pipeline → embed homepage → retrieve → summarize."""
 
    query_vec = embedder.encode([homepage_text])[0].tolist()

    retrieved = retrieve_chunks_rpc(query_vec, user_id=user_id, k=k)

    prompt = build_rag_prompt(homepage_text, retrieved)

    try:
        out = llm.generate_content(prompt)
        return out.text.strip()
    except Exception as e:
        print("RAG summary error:", e)
        return homepage_text[:600]
