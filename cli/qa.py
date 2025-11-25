"""
RAPTOR QA System for Incremental Architecture
Works with chunks, tree_nodes, and chunk_to_leaf tables
"""
import os
import json
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor
import google.generativeai as genai
from enum import Enum
import re
from dotenv import load_dotenv
from auth import AuthManager

load_dotenv()


class QuestionType(Enum):
    FACTUAL = "factual"          # Specific details → Level 0 (chunks)
    SUMMARY = "summary"          # Overview → Level 1-2 (tree_nodes)
    COMPARISON = "comparison"    # Compare concepts → Multi-level
    ANALYTICAL = "analytical"    # Deep analysis → Adaptive


class RaptorQAIncremental:
    """QA system for incremental RAPTOR architecture (chunks + tree_nodes)"""
    
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

        # Assistant persona
        self.assistant_profile = (
            "You are a Research Assistant helping users explore research papers. "
            "Cite source documents (by filename) for factual answers. Ask clarifying questions if ambiguous. "
            "Clearly separate: facts from uploaded documents vs. your own knowledge. "
            "Mark assistant knowledge as '[From my knowledge base]' and document facts as '[From: filename]'."
        )
        
        # Conversation history
        self.session_history: List[Tuple[str, str]] = []
    
    
    def list_documents(self) -> List[Dict]:
        """List all unique documents uploaded by the user"""
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get distinct documents by source_file
                # Note: In incremental schema, doc_id in chunks table is the actual document ID
                # Each document has multiple chunks with the same doc_id
                cur.execute("""
                    SELECT 
                        doc_id,
                        COUNT(*) as chunk_count,
                        MIN(created_at) as uploaded_at,
                        MAX(metadata->>'source_file') as source_file
                    FROM chunks
                    WHERE user_id = %s AND deleted = FALSE
                    GROUP BY doc_id
                    ORDER BY uploaded_at DESC
                """, (self.user_id,))
                
                return cur.fetchall()
        finally:
            conn.close()
    
    
    def show_db_diagnostics(self) -> Dict:
        """Show diagnostics about database state for current user"""
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get tree statistics
                cur.execute("SELECT * FROM get_tree_stats(%s)", (self.user_id,))
                stats = cur.fetchone()
                
                # Get sample documents
                cur.execute("""
                    SELECT doc_id, 
                           COUNT(*) as chunk_count,
                           metadata->>'source_file' as source_file
                    FROM chunks
                    WHERE user_id = %s AND deleted = FALSE
                    GROUP BY doc_id, metadata->>'source_file'
                    ORDER BY chunk_count DESC
                    LIMIT 5
                """, (self.user_id,))
                examples = cur.fetchall()
                
                return {
                    'total_chunks': stats['total_chunks'] if stats else 0,
                    'active_chunks': stats['active_chunks'] if stats else 0,
                    'deleted_chunks': stats['deleted_chunks'] if stats else 0,
                    'tree_levels': stats['tree_levels'] if stats else 0,
                    'nodes_per_level': stats['nodes_per_level'] if stats else {},
                    'examples': examples
                }
        finally:
            conn.close()
    
    
    def classify_question(self, question: str) -> QuestionType:
        """Classify question type using hybrid keyword + LLM approach"""
        q = question.lower()
        
        # Keyword patterns
        if re.search(r'\b(compare|versus|vs\.?|difference between)\b', q):
            return QuestionType.COMPARISON
        
        if re.search(r'\b(overview|summarize|main points?|about|explain)\b', q):
            return QuestionType.SUMMARY
        
        if re.search(r'^\s*(why|how does|what causes)\b', q):
            return QuestionType.ANALYTICAL
        
        if re.search(r'\b(what is|define|when|where|who|which)\b', q):
            return QuestionType.FACTUAL
        
        # Ambiguous - use Gemini
        try:
            prompt = f"""Classify this question as ONE word: FACTUAL, SUMMARY, COMPARISON, or ANALYTICAL

Question: {question}

Classification:"""
            
            response = self.gemini_model.generate_content(prompt)
            result = response.text.strip().upper()
            
            if "SUMMARY" in result:
                return QuestionType.SUMMARY
            elif "COMPARISON" in result:
                return QuestionType.COMPARISON
            elif "ANALYTICAL" in result:
                return QuestionType.ANALYTICAL
            else:
                return QuestionType.FACTUAL
        except:
            return QuestionType.FACTUAL
    
    
    def retrieve_chunks(self, question: str, top_k: int = 5) -> List[Dict]:
        """Retrieve from chunks table (Level 0)"""
        query_vector = self.model.encode(question, normalize_embeddings=True).tolist()
        
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Use match_chunks function
                cur.execute("""
                    SELECT * FROM match_chunks(%s::vector, %s::uuid, %s, FALSE)
                """, (query_vector, self.user_id, top_k))
                
                results = []
                for row in cur.fetchall():
                    results.append({
                        'chunk_id': row['chunk_id'],
                        'content': row['text'],
                        'metadata': row['metadata'],
                        'similarity': row['similarity'],
                        'level': 0  # chunks are always level 0
                    })
                
                return results
        finally:
            conn.close()
    
    
    def retrieve_tree_nodes(self, question: str, level: int, top_k: int = 5) -> List[Dict]:
        """Retrieve from tree_nodes table (Level 1+)"""
        query_vector = self.model.encode(question, normalize_embeddings=True).tolist()
        
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Use match_tree_nodes function
                cur.execute("""
                    SELECT * FROM match_tree_nodes(%s::vector, %s::uuid, %s, %s)
                """, (query_vector, self.user_id, level, top_k))
                
                results = []
                for row in cur.fetchall():
                    if row['summary_text']:  # Only include nodes with summaries
                        results.append({
                            'node_id': row['node_id'],
                            'content': row['summary_text'],
                            'metadata': {},
                            'similarity': row['similarity'],
                            'level': row['level']
                        })
                
                return results
        finally:
            conn.close()
    
    
    def retrieve_multi_level(self, question: str, levels: List[int], k_per_level: int = 3) -> List[Dict]:
        """Retrieve from multiple levels (0 = chunks, 1+ = tree_nodes)"""
        all_results = []
        
        for level in levels:
            if level == 0:
                results = self.retrieve_chunks(question, k_per_level)
            else:
                results = self.retrieve_tree_nodes(question, level, k_per_level)
            all_results.extend(results)
        
        # Sort by similarity
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        return all_results
    
    
    def retrieve_adaptive(self, question: str, question_type: QuestionType) -> List[Dict]:
        """Adaptive retrieval based on question type"""
        
        if question_type == QuestionType.FACTUAL:
            # Detailed chunks from Level 0
            return self.retrieve_chunks(question, top_k=5)
        
        elif question_type == QuestionType.SUMMARY:
            # Try high-level summaries first, fallback to chunks
            # Get max level from stats
            try:
                stats = self.show_db_diagnostics()
                max_level = stats.get('tree_levels', 0)
                
                if max_level >= 3:
                    # Try Level 2-3 first (highest summaries)
                    results = self.retrieve_tree_nodes(question, 2, top_k=2)
                    if not results and max_level >= 2:
                        results = self.retrieve_tree_nodes(question, 1, top_k=3)
                elif max_level >= 2:
                    # Try Level 1
                    results = self.retrieve_tree_nodes(question, 1, top_k=3)
                else:
                    results = []
                
                # Fallback to chunks if no summaries
                if not results:
                    results = self.retrieve_chunks(question, top_k=5)
                
                return results
            except:
                return self.retrieve_chunks(question, top_k=5)
        
        elif question_type == QuestionType.COMPARISON:
            # Mix of chunks and summaries
            return self.retrieve_multi_level(question, [0, 1], k_per_level=4)
        
        elif question_type == QuestionType.ANALYTICAL:
            # All levels
            try:
                stats = self.show_db_diagnostics()
                max_level = min(stats.get('tree_levels', 1), 3)
                levels = list(range(max_level + 1))
                return self.retrieve_multi_level(question, levels, k_per_level=2)
            except:
                return self.retrieve_chunks(question, top_k=5)
        
        else:
            return self.retrieve_chunks(question, top_k=5)
    
    
    def compress_context(self, documents: List[Dict], question: str, max_chunks: int = 4) -> List[Dict]:
        """Context compression: rerank and filter by relevance"""
        if len(documents) <= max_chunks:
            return documents
        
        # Score each document for relevance
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        
        scored_docs = []
        for doc in documents:
            content_lower = doc['content'].lower()
            content_words = set(re.findall(r'\b\w+\b', content_lower))
            
            # Keyword overlap score
            overlap = len(question_words & content_words)
            keyword_score = overlap / len(question_words) if question_words else 0
            
            # Combined score (vector similarity + keyword overlap)
            combined_score = doc['similarity'] * 0.7 + keyword_score * 0.3
            
            scored_docs.append((combined_score, doc))
        
        # Sort and take top chunks
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored_docs[:max_chunks]]
    
    
    def generate_answer(self, question: str, context_docs: List[Dict]) -> Tuple[str, List[str]]:
        """Generate answer using Gemini with context"""
        
        if not context_docs:
            # No documents - use knowledge base
            prompt = f"""{self.assistant_profile}

No relevant documents found in the uploaded files.
Answer using your knowledge base.

Question: {question}

Answer (mark as '[From my knowledge base]'):"""
            
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip(), []
            except Exception as e:
                return f"Error generating answer: {e}", []
        
        # Build context string
        context_parts = []
        sources = set()
        
        for i, doc in enumerate(context_docs, 1):
            level = doc.get('level', 0)
            content = doc['content'][:400]
            
            # Extract source
            try:
                metadata = doc.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                source = metadata.get('source_file', 'Unknown')
                if source != 'Unknown':
                    source_name = os.path.basename(source)
                    sources.add(source_name)
                else:
                    source_name = f'Cluster {i}'
            except:
                source_name = f'Chunk {i}'
            
            context_parts.append(f"[Source: {source_name}, Level {level}]:\n{content}\n")
        
        context = "\n".join(context_parts)
        
        # Build conversation history
        history_text = ""
        if self.session_history:
            last_turns = self.session_history[-2:]
            history_lines = []
            for uq, ua in last_turns:
                history_lines.append(f"User: {uq}\nAssistant: {ua}")
            history_text = "\n\nConversation so far:\n" + "\n".join(history_lines) + "\n\n"
        
        # Generate answer
        prompt = f"""{self.assistant_profile}
{history_text}
Answer based on the provided research content.

Research Content:
{context}

Question: {question}

Instructions:
- Cite source documents (e.g., "[From: w27392.pdf]")
- If synthesizing multiple sources, mention each
- If uncertain, say so

Answer:"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip(), list(sources)
        except Exception as e:
            return f"Error generating answer: {e}", list(sources)
    
    
    def summarize_all_documents(self) -> str:
        """Use tree hierarchy summaries for overview"""
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get tree stats
                cur.execute("SELECT * FROM get_tree_stats(%s)", (self.user_id,))
                stats = cur.fetchone()
                
                if not stats or stats['tree_levels'] == 0:
                    return "No hierarchy found. The system needs to build the tree structure first."
                
                max_level = stats['tree_levels']
                
                # Try to get high-level summaries (Level 2 or highest available)
                target_level = min(max_level, 2)
                
                cur.execute("""
                    SELECT summary_text, level
                    FROM tree_nodes
                    WHERE user_id = %s AND level = %s AND summary_text IS NOT NULL
                    ORDER BY node_id
                """, (self.user_id, target_level))
                
                summaries = cur.fetchall()
                
                if not summaries:
                    # Fallback to Level 1
                    cur.execute("""
                        SELECT summary_text, level
                        FROM tree_nodes
                        WHERE user_id = %s AND level = 1 AND summary_text IS NOT NULL
                        ORDER BY node_id
                    """, (self.user_id,))
                    summaries = cur.fetchall()
                    target_level = 1
                
                if not summaries:
                    return "No summaries found in the hierarchy. Please rebuild the tree."
                
                print(f"   Retrieved {len(summaries)} summaries from Level {target_level}")
                
                # Combine summaries
                combined_parts = [f"[Cluster {i+1}]\n{s['summary_text']}" 
                                 for i, s in enumerate(summaries)]
                combined = "\n\n---\n\n".join(combined_parts)
                
                # Synthesize with Gemini
                prompt = f"""{self.assistant_profile}

You have been given pre-computed summaries from the RAPTOR hierarchy (Level {target_level}). 
Synthesize these into a comprehensive overview:

{combined}

Provide a clear, well-organized synthesis covering:
1. Main topics and themes across all documents
2. Key findings and insights
3. Important methodologies or approaches mentioned

Synthesis:"""
                
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
        finally:
            conn.close()
    
    
    def ask(self, question: str, verbose: bool = False) -> Dict:
        """Main QA pipeline"""
        
        # Step 1: Classify question
        question_type = self.classify_question(question)
        
        if verbose:
            print(f"\n[Classification: {question_type.value}]")
        
        # Step 2: Check if documents exist
        try:
            stats = self.show_db_diagnostics()
            if stats.get('active_chunks', 0) == 0:
                answer = "No documents found. Please upload documents using add_document.py or batch_ingest.py."
                return {
                    'question': question,
                    'question_type': question_type.value,
                    'num_retrieved': 0,
                    'num_used': 0,
                    'answer': answer,
                    'source_files': [],
                    'sources': []
                }
        except Exception as e:
            if verbose:
                print(f"[Warning] Could not fetch diagnostics: {e}")
        
        # Step 3: Adaptive retrieval
        retrieved_docs = self.retrieve_adaptive(question, question_type)
        
        if verbose:
            print(f"[Retrieved: {len(retrieved_docs)} documents]")
            for doc in retrieved_docs[:3]:
                source = 'Chunk' if doc.get('level') == 0 else f"Node-L{doc.get('level')}"
                print(f"  - {source}, similarity: {doc['similarity']:.3f}")
        
        # Step 4: Context compression
        max_chunks = 6 if question_type == QuestionType.SUMMARY else 4
        compressed_docs = self.compress_context(retrieved_docs, question, max_chunks=max_chunks)
        
        if verbose:
            print(f"[Compressed to: {len(compressed_docs)} chunks]")
        
        # Step 5: Generate answer
        answer, source_files = self.generate_answer(question, compressed_docs)
        
        # Save to history
        self.session_history.append((question, answer))
        if len(self.session_history) > 20:
            self.session_history = self.session_history[-20:]
        
        return {
            'question': question,
            'question_type': question_type.value,
            'num_retrieved': len(retrieved_docs),
            'num_used': len(compressed_docs),
            'answer': answer,
            'source_files': source_files,
            'sources': compressed_docs
        }


def main():
    """Interactive QA session"""
    import sys
    
    # Check if user is logged in
    auth = AuthManager()
    user_id = auth.load_session()
    
    if not user_id:
        print("=" * 60)
        print(" AUTHENTICATION REQUIRED")
        print("=" * 60)
        print("\nYou need to login first:")
        print("  python auth.py login <username> <password>")
        print("\nOr register a new account:")
        print("  python auth.py register <username> <password>")
        print("\n" + "=" * 60)
        sys.exit(1)
    
    qa = RaptorQAIncremental(user_id)
    
    print("=" * 60)
    print("RAPTOR QA System (Incremental Architecture)")
    print("=" * 60)
    print(f"Logged in as: {auth.current_user['username']}")
    print("\nCommands:")
    print("  - Type a question to get an answer")
    print("  - 'list' to see all uploaded documents")
    print("  - 'diagnose' to show DB diagnostics")
    print("  - 'summarize_all' to get overview of all documents")
    print("  - 'quit' to exit\n")
    
    try:
        while True:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # LIST
            if question.lower() in ['list', 'docs', 'documents']:
                print("\n" + "-" * 60)
                print("UPLOADED DOCUMENTS:")
                print("-" * 60)
                docs = qa.list_documents()
                if docs:
                    for doc in docs:
                        source = doc.get('source_file') or 'Unknown'
                        # Handle None values
                        if source == 'None' or not source:
                            source = f"Document {str(doc['doc_id'])[:8]}"
                        else:
                            source = os.path.basename(source)
                        
                        doc_id_short = str(doc['doc_id'])[:8]
                        print(f"  • {source}")
                        print(f"    - Doc ID: {doc_id_short}...")
                        print(f"    - {doc['chunk_count']} chunks")
                        print(f"    - Uploaded: {doc['uploaded_at']}")
                else:
                    print("  No documents found.")
                print("=" * 60)
                continue
            
            # DIAGNOSE
            if question.lower() in ['diagnose', 'diag']:
                try:
                    stats = qa.show_db_diagnostics()
                    print("\n" + "-" * 60)
                    print("DATABASE DIAGNOSTICS:")
                    print("-" * 60)
                    print(f"Total chunks: {stats['total_chunks']}")
                    print(f"Active chunks: {stats['active_chunks']}")
                    print(f"Deleted chunks: {stats['deleted_chunks']}")
                    print(f"Tree levels: {stats['tree_levels']}")
                    print(f"Nodes per level: {stats['nodes_per_level']}")
                    print("\nExample documents:")
                    for ex in stats['examples']:
                        source = ex.get('source_file', 'Unknown')
                        print(f"  • {source} ({ex['chunk_count']} chunks)")
                    print("-" * 60)
                except Exception as e:
                    print(f"Error fetching diagnostics: {e}")
                continue
            
            # SUMMARIZE ALL
            if question.lower() in ['summarize_all', 'summarizeall']:
                print('\n' + '-' * 60)
                print(' Using RAPTOR hierarchy for summarization...')
                print('-' * 60)
                answer = qa.summarize_all_documents()
                print(answer)
                print('\n' + '-' * 60)
                continue
            
            # HELP
            if question.lower() in ['help', 'h', '?']:
                print("\n" + "-" * 60)
                print("HELP:")
                print("-" * 60)
                print("Commands:")
                print("  - Type a question to get an answer")
                print("  - 'list' to see all uploaded documents")
                print("  - 'diagnose' to show DB diagnostics")
                print("  - 'summarize_all' to get overview of all documents")
                print("  - 'quit' to exit")
                print("\nQuestion types:")
                print("  - Factual: Ask specific questions about document content")
                print("  - Summary: Get overviews and main points")
                print("  - Comparison: Compare information across documents")
                print("  - Analytical: Deep analysis with multi-level context")
                print("=" * 60)
                continue
            
            if not question:
                continue
            
            # Regular question
            result = qa.ask(question, verbose=True)
            
            print("\n" + "-" * 60)
            print("ANSWER:")
            print("-" * 60)
            print(result['answer'])
            if result.get('source_files'):
                print("\n[Sources: " + ", ".join(result['source_files']) + "]")
            print("\n" + "=" * 60)
    
    except (KeyboardInterrupt, EOFError):
        print('\nGoodbye!')


if __name__ == "__main__":
    main()
