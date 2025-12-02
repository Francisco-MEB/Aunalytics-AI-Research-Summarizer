# backend/api/router_query.py

import os
import json
from typing import List
from fastapi import APIRouter, Form, HTTPException
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()

router = APIRouter(tags=["Query"])
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-2.0-flash")


def retrieve_context(question: str, user_id: str, top_k: int = 4) -> List[dict]:
    """
    Retrieves the top-k most similar text chunks from the database
    based on the user's question. Respects RLS if user_id is set.
    
    Args:
        question: The user's question
        user_id: User UUID for RLS filtering
        top_k: Number of documents to retrieve (default 4, higher for summaries)
    
    Returns:
        List of documents with content and similarity scores
    """
    # Embed the user's question
    query_vector = embedder.encode(question, normalize_embeddings=True).tolist()
    
    try:
        # Query Supabase with vector similarity search
        response = supabase.rpc(
            'match_chunks',
            {
                'query_embedding': query_vector,
                'match_threshold': 0.0,
                'match_count': top_k,
                'filter_user_id': user_id
            }
        ).execute()
        
        # Return the matched documents
        return response.data if response.data else []
        
    except Exception as e:
        print(f"Retrieval error: {e}")
        # Fallback to direct query if RPC doesn't exist
        try:
            response = supabase.table("documents") \
                .select("doc_id, content, metadata") \
                .eq("user_id", user_id) \
                .limit(top_k) \
                .execute()
            return response.data if response.data else []
        except:
            return []


def generate_answer(question: str, context_docs: List[dict]) -> str:
    """Generate answer using Gemini LLM based on retrieved context"""
    if not context_docs:
        return "I couldn't find any relevant information to answer your question. Please make sure you've uploaded documents first."
    
    # Prepare context from retrieved documents
    context_text = ""
    for i, doc in enumerate(context_docs, 1):
        similarity = doc.get('similarity', 'N/A')
        if isinstance(similarity, (int, float)):
            context_text += f"\n--- Source {i} (Similarity: {similarity:.3f}) ---\n"
        else:
            context_text += f"\n--- Source {i} ---\n"
        context_text += doc.get('content', '')
        context_text += "\n"
    
    # Create the prompt for Gemini
    prompt = f"""You are a helpful research assistant. Based on the provided context, answer the user's question accurately and comprehensively.

Context from research documents:
{context_text}

User Question: {question}

Instructions:
- Provide a clear, accurate answer based ONLY on the information in the context above
- If the context doesn't contain enough information to fully answer the question, say so
- Include relevant details and examples from the context when possible
- If you mention specific information, you can reference it as coming from "the research documents"
- Keep your answer focused and well-structured
- Do NOT use markdown formatting, bullet points, or section headers - just plain text prose

Answer:"""

    try:
        # Generate response using Gemini
        response = llm.generate_content(prompt)
        
        if response.text:
            return response.text.strip()
        else:
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback response if Gemini fails
        preview = "\n\n".join([doc.get('content', '')[:200] + "..." for doc in context_docs[:2]])
        return f"I found {len(context_docs)} relevant documents but encountered an error generating the response. Here's a preview:\n\n{preview}"


@router.post("/")
async def query_documents(
    message: str = Form(...),
    user_id: str = Form(...),
    top_k: int = Form(4)
):
    """
    Query the knowledge base with RAG (Retrieval Augmented Generation)
    
    Parameters:
    - message: The user's question
    - user_id: User UUID for Row Level Security (required)
    - top_k: Number of document chunks to retrieve (default 4, use 10-15 for summaries)
    """
    
    try:
        # Check if this is a summarization request
        is_summary = any(word in message.lower() for word in ['summarize', 'summary', 'overview', 'what is this about'])
        
        if is_summary:
            top_k = max(top_k, 15)  # Use more chunks for summaries
        
        # Retrieve relevant context
        context_docs = retrieve_context(message, user_id=user_id, top_k=top_k)
        
        # Generate answer
        answer = generate_answer(message, context_docs)
        
        return {
            "response": answer,
            "num_sources": len(context_docs),
            "user_id": user_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
