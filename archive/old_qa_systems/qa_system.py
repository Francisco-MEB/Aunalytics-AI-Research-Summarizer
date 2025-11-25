import os
from typing import List, TypedDict
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import psycopg2
import google.generativeai as genai


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
        
        # Initialize database connection
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        # Initialize Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    
    def retrieve_context(self, question: str) -> List[Document]:
        """
        Retrieves the top-k most similar text chunks from the pgvector database
        based on the user's question. Respects RLS if user_id is set.
        """
        # Embed the user's question
        query_vector = self.model.encode(question, normalize_embeddings=True).tolist()

        # Connect to PostgreSQL and search
        try:
            with psycopg2.connect(self.db_url) as conn, conn.cursor() as cur:
                # If user_id is set, filter by it (RLS will also enforce this)
                if self.user_id:
                    cur.execute(
                        """
                        SELECT doc_id, content, 1 - (embedding <=> %s::vector) AS similarity
                        FROM documents
                        WHERE user_id = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT 4;
                        """,
                        (query_vector, self.user_id, query_vector)
                    )
                else:
                    # No user_id filter - search all accessible documents
                    cur.execute(
                        """
                        SELECT doc_id, content, 1 - (embedding <=> %s::vector) AS similarity
                        FROM documents
                        ORDER BY embedding <=> %s::vector
                        LIMIT 4;
                        """,
                        (query_vector, query_vector)
                    )
                rows = cur.fetchall()

            # Wrap rows into LangChain Document objects
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
        print(f" Searching for relevant context...")
        context = self.retrieve_context(question)
        
        print(f" Found {len(context)} relevant documents")
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
    
    parser = argparse.ArgumentParser(description="Research QA System with RLS support")
    parser.add_argument("--user-id", dest="user_id", type=str, default=None,
                       help="User UUID for Row Level Security (optional)")
    args = parser.parse_args()
    
    try:
        qa_system = QASystem(user_id=args.user_id)
        print(" QA System initialized successfully!")
        if args.user_id:
            print(f" RLS enabled - Querying documents for user: {args.user_id}")
        else:
            print(" RLS disabled - Querying all accessible documents")
        print(" Make sure your DATABASE_URL environment variable is set")
        
        while True:
            question = input("\n Enter your question (or 'quit' to exit): ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print(" Goodbye!")
                break
            
            if not question:
                continue
            
            # Process the question
            result = qa_system.ask_question(question)
            
            # Display results
            print(f"\n Answer: {result['answer']}")
            if result['context']:
                print(f"\n Sources ({result['num_sources']}):")
                for i, doc in enumerate(result['context'], 1):
                    print(f"  {i}. Score: {doc.metadata.get('score', 'N/A'):.3f}")
                    print(f"     {doc.page_content[:100]}...")
                    
    except Exception as e:
        print(f" Error starting QA system: {e}")
        print(" Make sure your DATABASE_URL is set and your database is accessible")


if __name__ == "__main__":
    main()


