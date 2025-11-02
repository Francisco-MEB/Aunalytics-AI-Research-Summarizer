#!/usr/bin/env python3
"""
Question-Answering System with RAG (Retrieval-Augmented Generation)
==================================================================
Combines vector similarity search with Gemini LLM for accurate answers.

Architecture:
1. User asks a question
2. Question is embedded using sentence-transformers
3. Vector similarity search retrieves relevant chunks from database
4. Retrieved chunks + question sent to Gemini for answer generation
5. Gemini generates contextual answer based on retrieved information
"""
import os
from typing import List
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import psycopg2
import google.generativeai as genai


class QASystem:
    """
    Main QA System orchestrating retrieval and generation.
    
    Components:
    - Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
    - Database: PostgreSQL with pgvector extension
    - LLM: Google Gemini 2.5 Flash
    """
    
    def __init__(self):
        """Initialize the QA system with all required components."""
        print("🔧 Initializing QA System...")
        
        # Load embedding model (same model used in ingestion)
        print("📥 Loading embedding model...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Get database connection string
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError(
                "DATABASE_URL environment variable not set. "
                "Please set it in your .env file."
            )
        
        # Configure Gemini API
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Please set it in your .env file."
            )
        
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        print("✅ QA System initialized successfully!\n")
    
    def retrieve_context(self, question: str, top_k: int = 4) -> List[Document]:
        """
        Retrieve most similar text chunks from the database.
        
        Uses cosine similarity (via pgvector's <=> operator) to find
        the most relevant chunks for the given question.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve (default: 4)
            
        Returns:
            List of Document objects with content and metadata
        """
        # Embed the question using the same model as ingestion
        query_vector = self.model.encode(
            question,
            normalize_embeddings=True
        ).tolist()

        # Query database for similar vectors
        try:
            with psycopg2.connect(self.db_url) as conn, conn.cursor() as cur:
                # pgvector's <=> operator computes cosine distance
                # 1 - distance = similarity score
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

            # Convert to LangChain Document objects
            retrieved_docs = [
                Document(
                    page_content=row[1],
                    metadata={"doc_id": row[0], "score": row[2]}
                )
                for row in rows
            ]
            
            return retrieved_docs
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            return []
    
    def generate_answer(self, question: str, context: List[Document]) -> str:
        """
        Generate answer using Gemini LLM based on retrieved context.
        
        Creates a prompt with context documents and the question,
        then uses Gemini to generate a grounded answer.
        
        Args:
            question: User's question
            context: Retrieved relevant documents
            
        Returns:
            Generated answer string
        """
        if not context:
            return "I couldn't find any relevant information to answer your question."
        
        # Format context from retrieved documents
        context_text = ""
        for i, doc in enumerate(context, 1):
            similarity = doc.metadata.get('score', 0)
            context_text += f"\n--- Source {i} (Relevance: {similarity:.2%}) ---\n"
            context_text += doc.page_content
            context_text += "\n"
        
        # Create prompt for Gemini
        prompt = f"""You are a helpful research assistant analyzing academic documents. 
Based on the provided context, answer the user's question accurately and comprehensively.

Context from research documents:
{context_text}

User Question: {question}

Instructions:
- Provide a clear, accurate answer based ONLY on the information in the context above
- If the context doesn't contain enough information to fully answer the question, say so
- Include relevant details and examples from the context when possible
- If you mention specific information, you can reference it as coming from "the research documents"
- Keep your answer focused and well-structured
- Do not make up or infer information not present in the context

Answer:"""

        try:
            # Generate response using Gemini
            response = self.gemini_model.generate_content(prompt)
            
            if response.text:
                return response.text.strip()
            else:
                return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            # Fallback: return context summary if Gemini fails
            return (
                f"I found {len(context)} relevant documents but encountered an error generating the response. "
                f"Here's a summary of what I found:\n\n" +
                "\n".join([doc.page_content[:200] + "..." for doc in context[:2]])
            )
    
    def ask_question(self, question: str) -> dict:
        """
        Complete QA workflow: retrieve context and generate answer.
        
        Args:
            question: User's question
            
        Returns:
            Dict with keys: question, answer, context, num_sources
        """
        print(f"🔍 Searching for relevant context...")
        context = self.retrieve_context(question)
        
        print(f"📚 Found {len(context)} relevant documents")
        for i, doc in enumerate(context, 1):
            score = doc.metadata.get('score', 0)
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"   {i}. Relevance: {score:.2%} | {preview}...")
        
        print(f"\n🤖 Generating answer with Gemini...")
        answer = self.generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "context": context,
            "num_sources": len(context)
        }


def main():
    """Main function to run the QA system interactively."""
    print("=" * 70)
    print("  🤖 AI Research Summarizer - Question Answering System")
    print("=" * 70)
    
    try:
        qa_system = QASystem()
        
        print("\n� Tip: Ask questions about the documents in your database")
        print("💡 Type 'quit', 'exit', or 'q' to stop\n")
        
        while True:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the QA system. Goodbye!")
                break
            
            if not question:
                continue
            
            print()  # Empty line for readability
            
            # Process the question
            result = qa_system.ask_question(question)
            
            # Display results
            print(f"\n{'='*70}")
            print(f"💡 ANSWER:")
            print(f"{'='*70}")
            print(result['answer'])
            print(f"\n{'='*70}")
            print(f"📖 Sources used: {result['num_sources']}")
            print(f"{'='*70}\n")
                    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting QA system: {e}")
        print("💡 Make sure your .env file is configured with:")
        print("   - DATABASE_URL (PostgreSQL connection string)")
        print("   - GEMINI_API_KEY (Google Gemini API key)")


if __name__ == "__main__":
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    main()

