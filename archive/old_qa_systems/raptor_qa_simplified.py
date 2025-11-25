"""
Simplified RAPTOR QA System - Clean, maintainable version
Key improvements:
- Removed complex retrieval logic
- Simplified summarization (direct per-document approach)
- Better error messages
- Cleaner code structure
"""
import os
import json
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class SimplifiedQASystem:
    """Simplified QA with focus on clarity and maintainability"""
    
    def __init__(self, user_id: str):
        """Initialize with user_id for RLS"""
        self.user_id = user_id
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Database config
        self.db_config = {
            'user': os.getenv('user'),
            'password': os.getenv('password'),
            'host': os.getenv('host'),
            'port': int(os.getenv('port', '5432')),
            'dbname': os.getenv('dbname')
        }
        
        # Gemini setup
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)
    
    
    def list_documents(self) -> List[Dict]:
        """List all documents uploaded by the user"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT metadata->>'source_file' as source_file,
                           metadata->>'file_hash' as file_hash,
                           COUNT(*) as chunk_count
                    FROM documents
                    WHERE user_id = %s AND hierarchy_level = 0
                    GROUP BY metadata->>'source_file', metadata->>'file_hash'
                    ORDER BY metadata->>'source_file'
                """, (self.user_id,))
                return cur.fetchall()
    
    
    def get_diagnostics(self) -> Dict:
        """Get database diagnostics"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Count by level
                cur.execute(
                    "SELECT hierarchy_level, COUNT(*) as count FROM documents WHERE user_id = %s GROUP BY hierarchy_level ORDER BY hierarchy_level",
                    (self.user_id,)
                )
                levels = {row['hierarchy_level']: row['count'] for row in cur.fetchall()}
                
                # Distinct documents
                cur.execute(
                    "SELECT COUNT(DISTINCT metadata->>'source_file') as distinct_docs FROM documents WHERE user_id = %s AND hierarchy_level = 0",
                    (self.user_id,)
                )
                distinct_docs = cur.fetchone()['distinct_docs']
                
                # Missing embeddings
                cur.execute(
                    "SELECT COUNT(*) as missing FROM documents WHERE user_id = %s AND embedding IS NULL",
                    (self.user_id,)
                )
                missing_embeddings = cur.fetchone()['missing']
                
                return {
                    'levels': levels,
                    'distinct_docs': distinct_docs,
                    'missing_embeddings': missing_embeddings
                }
    
    
    def retrieve_chunks(self, question: str, top_k: int = 5, hierarchy_level: int = 0) -> List[Dict]:
        """
        Simple retrieval: get top-k chunks by similarity
        
        Args:
            question: Query text
            top_k: Number of chunks to return
            hierarchy_level: Which level to search (0 = raw chunks, 1+ = summaries)
        """
        query_vector = self.model.encode(question, normalize_embeddings=True).tolist()
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT doc_id, content, metadata, hierarchy_level,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM documents
                    WHERE user_id = %s AND hierarchy_level = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_vector, self.user_id, hierarchy_level, query_vector, top_k))
                
                return cur.fetchall()
    
    
    def retrieve_chunks_for_document(self, source_file: str, top_k: int = 5) -> List[Dict]:
        """
        Get top chunks for a specific document
        Simple and direct - no complicated merging
        """
        # Use a generic "summarize" query to get representative chunks
        query_vector = self.model.encode("summarize this document", normalize_embeddings=True).tolist()
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT doc_id, content, metadata, hierarchy_level,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM documents
                    WHERE user_id = %s 
                      AND hierarchy_level = 0 
                      AND metadata->>'source_file' = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_vector, self.user_id, source_file, query_vector, top_k))
                
                return cur.fetchall()
    
    
    def summarize_document(self, source_file: str, chunks: List[Dict]) -> str:
        """
        Generate summary for a single document using Gemini
        Simple and focused
        """
        if not chunks:
            return f"No content found for {os.path.basename(source_file)}"
        
        # Combine chunks
        combined_text = "\n\n---\n\n".join([chunk['content'] for chunk in chunks])
        
        # Limit length to avoid token issues
        if len(combined_text) > 6000:
            combined_text = combined_text[:6000] + "\n\n[Content truncated...]"
        
        # Simple prompt
        prompt = f"""Summarize this research document in 2-3 concise paragraphs. Focus on the main findings, methods, and conclusions.

Document: {os.path.basename(source_file)}

Content:
{combined_text}

Summary:"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating summary: {e}"
    
    
    def summarize_all_documents(self) -> Dict[str, str]:
        """
        Summarize each document individually
        Returns a dict mapping source_file -> summary
        
        This is MUCH simpler than the old approach:
        1. Get list of documents
        2. For each document, get top 5 chunks
        3. Summarize those chunks
        4. Done!
        """
        docs = self.list_documents()
        
        if not docs:
            return {}
        
        summaries = {}
        
        print(f"\n Summarizing {len(docs)} documents...\n")
        
        for doc in docs:
            source_file = doc['source_file']
            file_name = os.path.basename(source_file)
            
            print(f" Processing: {file_name}...")
            
            # Get top chunks for this document
            chunks = self.retrieve_chunks_for_document(source_file, top_k=5)
            
            if not chunks:
                summaries[source_file] = f"No chunks retrieved for {file_name}"
                print(f"   ️  Warning: No chunks found")
                continue
            
            print(f"   Retrieved {len(chunks)} chunks (avg similarity: {sum(c['similarity'] for c in chunks)/len(chunks):.3f})")
            
            # Generate summary
            summary = self.summarize_document(source_file, chunks)
            summaries[source_file] = summary
            print(f"    Summary generated\n")
        
        return summaries
    
    
    def ask(self, question: str, verbose: bool = True) -> Dict:
        """
        Answer a question using simple retrieval + LLM
        
        This is simpler than before:
        1. Retrieve top chunks (no complex classification)
        2. Build context
        3. Ask Gemini
        """
        if verbose:
            print(f"\n Searching for: {question}")
        
        # Check if hierarchy exists
        diag = self.get_diagnostics()
        has_hierarchy = len(diag['levels']) > 1
        
        # For summary questions, try hierarchy first if it exists
        if any(word in question.lower() for word in ['summarize', 'overview', 'about', 'main points']):
            if has_hierarchy:
                # Try to get high-level summaries
                chunks = self.retrieve_chunks(question, top_k=3, hierarchy_level=max(diag['levels'].keys()))
                if verbose:
                    print(f"   Using hierarchy Level {max(diag['levels'].keys())}")
            else:
                # No hierarchy, use raw chunks
                chunks = self.retrieve_chunks(question, top_k=5, hierarchy_level=0)
                if verbose:
                    print(f"   No hierarchy found, using Level 0 chunks")
        else:
            # For specific questions, always use Level 0 (detailed chunks)
            chunks = self.retrieve_chunks(question, top_k=5, hierarchy_level=0)
            if verbose:
                print(f"   Using Level 0 (detailed chunks)")
        
        if not chunks:
            return {
                'question': question,
                'answer': "No relevant documents found. Try uploading documents first.",
                'num_chunks': 0
            }
        
        if verbose:
            print(f"   Retrieved {len(chunks)} chunks")
            for i, chunk in enumerate(chunks[:3], 1):
                fname = os.path.basename(chunk['metadata'].get('source_file', 'Unknown'))
                print(f"     {i}. {fname} (similarity: {chunk['similarity']:.3f})")
        
        # Build context
        context_parts = []
        for chunk in chunks:
            fname = os.path.basename(chunk['metadata'].get('source_file', 'Unknown'))
            context_parts.append(f"[Source: {fname}]\n{chunk['content']}\n")
        
        context = "\n---\n".join(context_parts)
        
        # Ask Gemini
        prompt = f"""You are a research assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer based on the context above
- Cite sources when relevant (e.g., "According to science.1203877.docx...")
- If the context doesn't contain the answer, say so clearly
- Keep your answer focused and well-structured

Answer:"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            answer = f"Error generating answer: {e}"
        
        return {
            'question': question,
            'answer': answer,
            'num_chunks': len(chunks)
        }


def main():
    """Interactive CLI"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python raptor_qa_simplified.py <user_id>")
        sys.exit(1)
    
    user_id = sys.argv[1]
    qa = SimplifiedQASystem(user_id)
    
    print("=" * 60)
    print("Simplified RAPTOR QA System")
    print("=" * 60)
    print("\nCommands:")
    print("  • Type a question to get an answer")
    print("  • 'list' - Show all uploaded documents")
    print("  • 'diagnose' - Show database statistics")
    print("  • 'summarize_all' - Summarize each document")
    print("  • 'help' - Show this message")
    print("  • 'quit' - Exit\n")
    
    try:
        while True:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            # LIST
            if question.lower() in ['list', 'docs']:
                print("\n" + "-" * 60)
                print("UPLOADED DOCUMENTS:")
                print("-" * 60)
                docs = qa.list_documents()
                if docs:
                    for doc in docs:
                        fname = os.path.basename(doc['source_file']) if doc['source_file'] else 'Unknown'
                        fhash = doc['file_hash'][:8] if doc['file_hash'] else 'no-hash'
                        print(f"  • {fname} ({doc['chunk_count']} chunks) [hash: {fhash}...]")
                else:
                    print("  No documents found.")
                print("-" * 60)
                continue
            
            # DIAGNOSE
            if question.lower() in ['diagnose', 'diag']:
                print("\n" + "-" * 60)
                print("DATABASE DIAGNOSTICS:")
                print("-" * 60)
                diag = qa.get_diagnostics()
                print(f"Hierarchy levels: {diag['levels']}")
                print(f"Distinct documents: {diag['distinct_docs']}")
                print(f"Missing embeddings: {diag['missing_embeddings']}")
                print("-" * 60)
                continue
            
            # SUMMARIZE ALL
            if question.lower() in ['summarize_all', 'summarizeall', 'sum_all']:
                summaries = qa.summarize_all_documents()
                
                if not summaries:
                    print("\nNo documents to summarize.")
                    continue
                
                print("\n" + "=" * 60)
                print("DOCUMENT SUMMARIES")
                print("=" * 60)
                
                for source_file, summary in summaries.items():
                    fname = os.path.basename(source_file)
                    print(f"\n {fname}")
                    print("-" * 60)
                    print(summary)
                    print()
                
                print("=" * 60)
                continue
            
            # HELP
            if question.lower() in ['help', 'h', '?']:
                print("\nCommands:")
                print("  • Ask any question about your documents")
                print("  • 'list' - Show uploaded documents")
                print("  • 'diagnose' - Database statistics")
                print("  • 'summarize_all' - Summarize each document")
                print("  • 'quit' - Exit")
                continue
            
            # Default: answer question
            result = qa.ask(question, verbose=True)
            
            print("\n" + "-" * 60)
            print("ANSWER:")
            print("-" * 60)
            print(result['answer'])
            print("\n" + "=" * 60)
    
    except (KeyboardInterrupt, EOFError):
        print('\nGoodbye!')
        return


if __name__ == "__main__":
    main()
