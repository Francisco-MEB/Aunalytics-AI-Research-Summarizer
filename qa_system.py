import os
import sys
from pathlib import Path
from typing import List, TypedDict
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import psycopg2
import google.generativeai as genai

# Add database connection path
sys.path.insert(0, str(Path(__file__).parent / "embeddings" / "database connection"))
from db_connection import get_db_connection


class QASystem:
    def __init__(self, user_id: str = None):
        """Initialize the QA system with embedding model, database connection, and Gemini
        
        Args:
            user_id: Optional UUID for Row Level Security. If provided, only queries user's documents.
        """
        # Initialize embedding model (same as ingest.py)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Store user_id for RLS
        self.user_id = user_id
        
        # Initialize Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    
    def retrieve_context(self, question: str, top_k: int = 4) -> List[Document]:
        """
        Retrieves the top-k most similar text chunks from the pgvector database
        based on the user's question. Respects RLS if user_id is set.
        
        Args:
            question: The user's question
            top_k: Number of documents to retrieve (default 4, higher for summaries)
        """
        # Embed the user's question
        query_vector = self.model.encode(question, normalize_embeddings=True).tolist()
        
        # Connect to PostgreSQL and search
        try:
            with get_db_connection() as conn, conn.cursor() as cur:
                if self.user_id:
                    cur.execute(
                        """
                        SELECT doc_id, content, 1 - (embedding <=> %s::vector) AS similarity
                        FROM documents
                        WHERE user_id = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (query_vector, self.user_id, query_vector, top_k)
                    )
                else:
                    cur.execute(
                        """
                        SELECT doc_id, content, 1 - (embedding <=> %s::vector) AS similarity
                        FROM documents
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (query_vector, query_vector, top_k)
                    )
                rows = cur.fetchall()
                retrieved_docs = [
                    Document(
                        page_content=row[1],
                        metadata={"doc_id": row[0], "score": row[2]}
                    )
                    for row in rows
                ]
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Database error: {e}")
            return []
    
    def generate_answer(self, question: str, context: List[Document]) -> str:
        """Generate answer using Gemini LLM based on retrieved context"""
        if not context:
            return "I couldn't find any relevant information to answer your question."
        
        # Prepare context from retrieved documents
        context_text = ""
        for i, doc in enumerate(context, 1):
            context_text += f"\n--- Source {i} (Similarity: {doc.metadata.get('score', 'N/A'):.3f}) ---\n"
            context_text += doc.page_content
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

Answer:"""

        try:
            # Generate response using Gemini
            response = self.gemini_model.generate_content(prompt)
            
            if response.text:
                return response.text.strip()
            else:
                return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                
        except Exception as e:
            print(f"Gemini API error: {e}")
            # Fallback response if Gemini fails
            return f"I found {len(context)} relevant documents but encountered an error generating the response. Here's a summary of what I found:\n\n" + "\n".join([doc.page_content[:200] + "..." for doc in context[:2]])
    
    def ask_question(self, question: str) -> dict:
        """Orchestrate the whole QA process"""
        # Check if this is a summarization request
        is_summary = any(word in question.lower() for word in ['summarize', 'summary', 'overview', 'what is this about'])
        
        if is_summary:
            print("[PROCESSING] Summarization request detected")
            print("[RETRIEVAL]  Fetching top 15 most relevant documents...")
            context = self.retrieve_context(question, top_k=15)
            print(f"[RETRIEVED]  {len(context)} documents found")
        else:
            print("[PROCESSING] Searching knowledge base...")
            context = self.retrieve_context(question, top_k=4)
            print(f"[RETRIEVED]  {len(context)} relevant documents")
        
        print("[GENERATING] Creating response with AI...")
        answer = self.generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "context": context,
            "num_sources": len(context)
        }


def main():
    """Main function to run the QA system"""
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Research QA System with RLS support")
    parser.add_argument("--user-id", dest="user_id", type=str, required=True,
                       help="User UUID for Row Level Security (required)")
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print(" "*20 + "RESEARCH QA SYSTEM")
    print("="*70)
    
    try:
        qa_system = QASystem(user_id=args.user_id)
        print("\n[SYSTEM] Initialization successful")
        if args.user_id:
            print(f"[USER]   {args.user_id[:8]}...{args.user_id[-8:]}")
        print("\n" + "-"*70)
        print("Type 'quit' or 'exit' to close the system")
        print("-"*70)
        
        while True:
            question = input("\n[QUESTION] > ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n" + "="*70)
                print("Thank you for using the Research QA System")
                print("="*70 + "\n")
                break
            
            if not question:
                continue
            
            print("\n" + "-"*70)
            # Process the question
            result = qa_system.ask_question(question)
            
            # Display results with better formatting
            print("\n[ANSWER]")
            print("-"*70)
            print(result['answer'])
            print("-"*70)
            
            if result['context']:
                num_sources = result['num_sources']
                print(f"\n[SOURCES] {num_sources} document(s) retrieved")
                
                # Only show source previews if there are 10 or fewer
                if num_sources <= 10:
                    print("-"*70)
                    for i, doc in enumerate(result['context'], 1):
                        score = doc.metadata.get('score', 0)
                        print(f"\n  [{i}] Relevance: {score:.3f}")
                        preview = doc.page_content[:150].replace('\n', ' ')
                        print(f"      {preview}...")
                else:
                    print(f"      (Too many to display - used {num_sources} chunks)")
            print("-"*70)
                    
    except Exception as e:
        print("\n" + "="*70)
        print("[ERROR] Failed to initialize QA system")
        print("="*70)
        print(f"\nDetails: {e}")
        print("\nPlease verify:")
        print("  - Database credentials in .env file")
        print("  - Network connection to database")
        print("  - User ID is valid")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()


