"""
Enhanced RAPTOR QA System with Hierarchy-Aware Retrieval and Context Compression
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

load_dotenv()


class QuestionType(Enum):
    FACTUAL = "factual"          # Specific details → Level 0
    SUMMARY = "summary"          # Overview → Level 1-2
    COMPARISON = "comparison"    # Compare concepts → Multi-level
    ANALYTICAL = "analytical"    # Deep analysis → Adaptive


class RaptorQASystem:
    """Advanced QA with RAPTOR hierarchy, context compression, and adaptive retrieval"""
    
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

        # Assistant persona & helper prompt: used in all LLM calls
        # This sets self-awareness so the model behaves like a research assistant
        self.assistant_profile = (
            "You are a Research Assistant helping users explore research papers. "
            "Cite source documents (by filename) for factual answers. Ask clarifying questions if ambiguous. "
            "Clearly separate: facts from uploaded documents vs. your own knowledge. "
            "Mark assistant knowledge as '[From my knowledge base]' and document facts as '[From: filename]'."
        )
        # Maintain short-lived conversational history (for the current CLI session)
        self.session_history: List[Tuple[str, str]] = []  # list of (user_q, assistant_a)
    
    
    def list_documents(self) -> List[Dict]:
        """List all unique documents uploaded by the user"""
        import json
        
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get distinct source files from metadata
                cur.execute("""
                    SELECT DISTINCT metadata->>'source_file' as source_file,
                           COUNT(*) as chunk_count,
                           MIN(hierarchy_level) as min_level,
                           MAX(hierarchy_level) as max_level
                    FROM documents
                    WHERE user_id = %s AND hierarchy_level = 0
                    GROUP BY metadata->>'source_file'
                    ORDER BY metadata->>'source_file'
                """, (self.user_id,))
                
                return cur.fetchall()
        finally:
            conn.close()


    def show_db_diagnostics(self) -> Dict:
        """Show diagnostics about database state for current user.

        Returns stats: total chunks/levels, example files and counts, embedding null counts.
        """
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Count per hierarchy level
                cur.execute(
                    "SELECT hierarchy_level, COUNT(*) as count FROM documents WHERE user_id = %s GROUP BY hierarchy_level ORDER BY hierarchy_level",
                    (self.user_id,)
                )
                level_counts = {row['hierarchy_level']: row['count'] for row in cur.fetchall()}

                # Total distinct documents (by source file)
                cur.execute(
                    "SELECT COUNT(DISTINCT metadata->>'source_file') as distinct_docs FROM documents WHERE user_id = %s AND metadata->>'source_file' IS NOT NULL",
                    (self.user_id,)
                )
                distinct_docs = cur.fetchone()['distinct_docs']

                # Example files
                cur.execute(
                    "SELECT metadata->>'source_file' as source_file, COUNT(*) as chunk_count FROM documents WHERE user_id = %s AND hierarchy_level = 0 GROUP BY metadata->>'source_file' ORDER BY 2 DESC LIMIT 5",
                    (self.user_id,)
                )
                examples = cur.fetchall()

                # Check for null embeddings count
                cur.execute(
                    "SELECT COUNT(*) as missing_embeddings FROM documents WHERE user_id = %s AND (embedding IS NULL OR array_length(embedding,1) = 0)",
                    (self.user_id,)
                )
                missing_embeddings = cur.fetchone()['missing_embeddings']

                return {
                    'levels': level_counts,
                    'distinct_docs': distinct_docs,
                    'examples': examples,
                    'missing_embeddings': missing_embeddings
                }
        finally:
            conn.close()
    
    
    def classify_question(self, question: str) -> QuestionType:
        """Classify question type using hybrid keyword + LLM approach"""
        q = question.lower()
        
        # Keyword patterns for clear cases
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
    
    
    def retrieve_from_level(self, question: str, level: int, top_k: int = 5) -> List[Dict]:
        """Retrieve documents from specific hierarchy level"""
        query_vector = self.model.encode(question, normalize_embeddings=True).tolist()
        
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT doc_id, content, metadata, hierarchy_level,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM documents
                    WHERE user_id = %s AND hierarchy_level = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_vector, self.user_id, level, query_vector, top_k))
                
                rows = cur.fetchall()
                if not rows:
                    # Log diagnostic: no rows. We don't raise — return empty list but print diagnostics.
                    try:
                        stats = self.show_db_diagnostics()
                        print(f"[Diagnostics] No rows returned for level {level}. DB levels: {stats['levels']}, distinct_docs: {stats['distinct_docs']}, missing_embeddings: {stats['missing_embeddings']}")
                    except Exception:
                        pass
                return rows
        finally:
            conn.close()
    
    
    def retrieve_multi_level(self, question: str, levels: List[int], k_per_level: int = 3) -> List[Dict]:
        """Retrieve from multiple hierarchy levels"""
        all_results = []
        
        for level in levels:
            results = self.retrieve_from_level(question, level, k_per_level)
            all_results.extend(results)
        
        # Sort by similarity
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        return all_results


    def retrieve_chunks_for_file(self, source_file: str, level: int = 0, top_k: int = 5) -> List[Dict]:
        """Retrieve top chunks for a given source file at a specified hierarchy level"""
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT doc_id, content, metadata, hierarchy_level,
                           1 - (embedding <=> %s::vector) AS similarity
                       FROM documents
                       WHERE user_id = %s AND hierarchy_level = %s AND metadata->>'source_file' = %s
                       ORDER BY embedding <=> %s::vector
                       LIMIT %s""",
                    (self.model.encode("summarize", normalize_embeddings=True).tolist(), self.user_id, level, source_file, self.model.encode("summarize", normalize_embeddings=True).tolist(), top_k)
                )
                return cur.fetchall()
        finally:
            conn.close()
    
    
    def summarize_all_documents_from_hierarchy(self) -> str:
        """Use RAPTOR hierarchy summaries (Level 1-2) instead of re-summarizing Level 0"""
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # First try Level 2 (high-level summaries)
                cur.execute("""
                    SELECT content, metadata
                    FROM documents
                    WHERE user_id = %s AND hierarchy_level = 2
                    ORDER BY doc_id
                """, (self.user_id,))
                summaries = cur.fetchall()
                
                level_used = 2
                if not summaries:
                    # Fallback: use Level 1 if no Level 2
                    print("  ℹ️  No Level 2 summaries found, using Level 1...")
                    cur.execute("""
                        SELECT content, metadata
                        FROM documents
                        WHERE user_id = %s AND hierarchy_level = 1
                        ORDER BY doc_id
                    """, (self.user_id,))
                    summaries = cur.fetchall()
                    level_used = 1
                
                if not summaries:
                    return "No hierarchy summaries found. Please run batch_ingest.py with --raptor-mode clustering to build the hierarchy."
                
                print(f"   Retrieved {len(summaries)} summaries from Level {level_used}")
                
                # Combine summaries
                combined_parts = []
                for idx, s in enumerate(summaries, 1):
                    try:
                        metadata = s['metadata']
                        if isinstance(metadata, str):
                            import json
                            metadata = json.loads(metadata)
                        source = metadata.get('source_file', f'Cluster {idx}')
                    except:
                        source = f'Cluster {idx}'
                    
                    combined_parts.append(f"[Cluster {idx} - {os.path.basename(source)}]\n{s['content']}")
                
                combined = "\n\n---\n\n".join(combined_parts)
                
                prompt = f"""{self.assistant_profile}

You have been given pre-computed summaries from the RAPTOR hierarchy (Level {level_used}). Synthesize these into a comprehensive overview:

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
    
    
    def diversify_by_document(self, results: List[Dict], max_per_doc: int = 2) -> List[Dict]:
        """Ensure diversity by limiting chunks per source document"""
        import json
        
        doc_counts = {}
        diversified = []
        
        for result in results:
            # Extract source document
            try:
                metadata = result.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                source = metadata.get('source_file', 'Unknown')
            except:
                source = 'Unknown'
            
            # Track count per document
            if source not in doc_counts:
                doc_counts[source] = 0
            
            # Only add if we haven't exceeded max per document
            if doc_counts[source] < max_per_doc:
                diversified.append(result)
                doc_counts[source] += 1
        
        return diversified


    def retrieve_one_chunk_per_doc(self, question: str, level: int = 0, max_docs: int = 10) -> List[Dict]:
        """Retrieve the single most relevant chunk per distinct source_file up to max_docs.

        This helps summarization across multiple documents by ensuring all documents are represented.
        """
        # Get list of distinct source files for the user
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT DISTINCT metadata->>'source_file' as source_file FROM documents WHERE user_id = %s AND hierarchy_level = %s AND metadata->>'source_file' IS NOT NULL",
                    (self.user_id, level)
                )
                rows = cur.fetchall()
                sources = [r['source_file'] for r in rows]
        finally:
            conn.close()

        chunks = []
        # For each source file, get the top chunk
        for source in sources[:max_docs]:
            # Use the existing method to fetch top chunks for a file
            file_chunks = self.retrieve_chunks_for_file(source, level=level, top_k=1)
            if file_chunks:
                chunks.append(file_chunks[0])
        return chunks
    
    
    def retrieve_adaptive(self, question: str, question_type: QuestionType) -> List[Dict]:
        """Adaptive retrieval based on question type"""
        
        if question_type == QuestionType.FACTUAL:
            # Detailed chunks from Level 0
            return self.retrieve_from_level(question, 0, top_k=5)
        
        elif question_type == QuestionType.SUMMARY:
            # High-level summaries from Level 1-2
            # Try Level 2 first (most abstract), fallback to Level 1 and then Level 0
            results = self.retrieve_from_level(question, 2, top_k=2)
            if not results:
                results = self.retrieve_from_level(question, 1, top_k=3)
            # If no summary-level nodes exist, fall back to Level 0 chunks (detailed)
            if not results:
                results = self.retrieve_from_level(question, 0, top_k=5)
            # If multiple docs are present, ensure we represent all docs by taking top chunk per doc
            try:
                stats = self.show_db_diagnostics()
                if stats.get('distinct_docs', 0) > 1:
                    # Get one representative chunk per document
                    per_doc_chunks = self.retrieve_one_chunk_per_doc(question, level=0, max_docs=stats['distinct_docs'])
                    # Merge and dedupe, preferring high-similarity results
                    # Use union of per_doc_chunks + results (existing ranking) and sort
                    id_set = {r['doc_id'] for r in results}
                    merged = list(results)
                    for c in per_doc_chunks:
                        if c['doc_id'] not in id_set:
                            merged.append(c)
                    # sort merged by similarity
                    merged.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                    return merged
            except Exception:
                pass
            return results
        
        elif question_type == QuestionType.COMPARISON:
            # Mix of Level 0 (details) and Level 1 (context)
            # Get more results initially, then diversify to ensure multiple documents
            results = self.retrieve_multi_level(question, [0, 1], k_per_level=4)
            # Limit to 2 chunks per document for better cross-document comparison
            return self.diversify_by_document(results, max_per_doc=2)
        
        elif question_type == QuestionType.ANALYTICAL:
            # All levels for comprehensive understanding
            return self.retrieve_multi_level(question, [0, 1, 2], k_per_level=2)
        
        else:
            return self.retrieve_from_level(question, 0, top_k=5)


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
    
    
    def _generate_knowledge_only_answer(self, question: str) -> str:
        """Generate answer using only Gemini's knowledge base (no documents)"""
        history_text = ""
        if self.session_history:
            _last = self.session_history[-2:]
            history_lines = []
            for uq, ua in _last:
                history_lines.append(f"User: {uq}\nAssistant: {ua}")
            history_text = "\n\nConversation so far:\n" + "\n".join(history_lines) + "\n\n"
        
        prompt = f"""{self.assistant_profile}
{history_text}
No relevant documents found in the uploaded files.
Answer using your knowledge base.

Question: {question}

Answer (mark as '[From my knowledge base]'):"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating answer: {e}"
    
    
    def generate_answer(self, question: str, context_docs: List[Dict], use_fallback: bool = False) -> Tuple[str, List[str]]:
        """Generate answer using Gemini with context, return answer and sources
        
        Args:
            question: User's question
            context_docs: Retrieved document chunks
            use_fallback: If True, also provide additional context from Gemini's knowledge
        """
        import json
        
        if not context_docs:
            # No documents at all - only use knowledge fallback
            return self._generate_knowledge_only_answer(question), []
        
        # Extract unique source files
        sources = set()
        
        # Build context string with source information
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            level = doc.get('hierarchy_level', 0)
            content = doc['content'][:350]  # Reduced from 500 to 350 for token savings
            
            # Extract source file from metadata
            try:
                metadata = doc.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                source_file = metadata.get('source_file', 'Unknown')
                source_name = os.path.basename(source_file) if source_file != 'Unknown' else 'Unknown'
                sources.add(source_name)
            except:
                source_name = 'Unknown'
            
            context_parts.append(f"[Source: {source_name}, Level {level}]:\n{content}\n")
        
        context = "\n".join(context_parts)
        
        # Attach a short conversation history (last 2 turns) - reduced from 4 for efficiency
        history_text = ""
        if self.session_history:
            # Only keep a short window
            _last = self.session_history[-2:]
            history_lines = []
            for uq, ua in _last:
                history_lines.append(f"User: {uq}\nAssistant: {ua}")
            history_text = "\n\nConversation so far:\n" + "\n".join(history_lines) + "\n\n"

        # Build prompt based on whether we have documents or using knowledge fallback
        if use_fallback:
            # Hybrid mode: show document answer first (with warning), then add knowledge context
            max_sim = max(doc.get('similarity', 0) for doc in context_docs) if context_docs else 0
            
            prompt = f"""{self.assistant_profile}
{history_text}
Provide a TWO-PART answer:

Research Content (NOTE: Low similarity score {max_sim:.2f} - may not directly address question):
{context}

Question: {question}

Format:
---
**Part 1 - Based on uploaded documents:**
(Note: Low relevance score {max_sim:.2f})
[Analyze what the documents show, acknowledging limited relevance]

---
**Part 2 - Additional context from my knowledge base:**
[Provide relevant information from your training to give complete context]

Answer:"""
        else:
            # Normal mode: answer from uploaded documents only
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
    
    
    def ask(self, question: str, verbose: bool = False) -> Dict:
        """Main QA pipeline"""
        import json
        
        # Step 1: Classify question
        question_type = self.classify_question(question)
        
        if verbose:
            print(f"\n[Classification: {question_type.value}]")
        
        # Diagnostics: check if there are ANY documents in DB for this user
        try:
            stats = self.show_db_diagnostics()
            if stats.get('distinct_docs', 0) == 0:
                advice = (
                    "No documents were found in your account. "
                    "Try `python embeddings/batch_ingest.py --in <path> --user-id <your-id>` to add documents, "
                    "or `list` to view uploaded docs."
                )
                if verbose:
                    print(f"[Diagnostics] {advice}")
                # Return early with knowledge-only answer and useful hint
                return {
                    'question': question,
                    'question_type': question_type.value,
                    'num_retrieved': 0,
                    'num_used': 0,
                    'answer': self._generate_knowledge_only_answer(question),
                    'source_files': [],
                    'sources': []
                }
        except Exception:
            # If diagnostics fail, continue with retrieval; avoid breaking QA
            pass

        # Step 2: Adaptive retrieval
        retrieved_docs = self.retrieve_adaptive(question, question_type)
        
        if verbose:
            print(f"[Retrieved: {len(retrieved_docs)} documents]")
            for doc in retrieved_docs[:3]:
                # Extract source from metadata
                try:
                    metadata = doc.get('metadata', {})
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    source = os.path.basename(metadata.get('source_file', 'Unknown'))
                except:
                    source = 'Unknown'
                print(f"  - Level {doc['hierarchy_level']}, similarity: {doc['similarity']:.3f}, source: {source}")
        
        # Step 3: Context compression
        # If this is a summary query across multiple documents, increase max_chunks to include one per doc
        max_chunks = 4
        try:
            stats = self.show_db_diagnostics()
            distinct_docs = stats.get('distinct_docs', 0)
            if question_type == QuestionType.SUMMARY and distinct_docs > 1:
                # Allow one chunk per document up to a cap to ensure representation across docs
                max_chunks = min(max(4, distinct_docs), 12)
        except Exception:
            max_chunks = 4

        compressed_docs = self.compress_context(retrieved_docs, question, max_chunks=max_chunks)
        if verbose:
            print(f"[Compressing to {max_chunks} chunks (distinct_docs hint)]")
        
        if verbose:
            print(f"[Compressed to: {len(compressed_docs)} chunks]")
        
        # Check if we should use knowledge fallback (low similarity scores)
        use_fallback = False
        if compressed_docs:
            max_similarity = max(doc.get('similarity', 0) for doc in compressed_docs)
            if max_similarity < 0.3:
                use_fallback = True
                if verbose:
                    print(f"[Low similarity ({max_similarity:.3f}) - will use knowledge fallback]")
        else:
            use_fallback = True
            if verbose:
                print("[No documents found - will use knowledge fallback]")
                # If DB does contain documents but none were retrieved, offer diagnostics
                try:
                    stats = self.show_db_diagnostics()
                    if stats.get('distinct_docs', 0) > 0:
                        print("[Diagnostics] Documents exist in DB, but no documents were retrieved for this question.")
                        print(f"[Diagnostics] Run 'diagnose' to view DB stats. Levels: {stats['levels']}, missing embeddings: {stats['missing_embeddings']}")
                except Exception:
                    pass
                # Show diagnostics to help the user understand why
                try:
                    stats = self.show_db_diagnostics()
                    print(f"[Diagnostics] DB levels: {stats['levels']}, distinct_docs: {stats['distinct_docs']}, missing_embeddings: {stats['missing_embeddings']}")
                except Exception:
                    pass
        
        # If this is a summary across multiple docs, ensure at least one chunk per doc is present
        try:
            if question_type == QuestionType.SUMMARY:
                stats = self.show_db_diagnostics()
                distinct_docs = stats.get('distinct_docs', 0)
                if distinct_docs > 1:
                    # Build set of source files already included
                    sources_included = set()
                    for d in compressed_docs:
                        try:
                            md = d.get('metadata', {})
                            if isinstance(md, str):
                                md = json.loads(md)
                            sources_included.add(md.get('source_file', 'Unknown'))
                        except Exception:
                            pass
                    # If some sources missing, add one chunk for them
                    if len(sources_included) < distinct_docs:
                        per_doc_chunks = self.retrieve_one_chunk_per_doc(question, level=0, max_docs=distinct_docs)
                        for c in per_doc_chunks:
                            md = c.get('metadata', {})
                            if isinstance(md, str):
                                md = json.loads(md)
                            if md.get('source_file') not in sources_included:
                                compressed_docs.append(c)
                                sources_included.add(md.get('source_file'))
        except Exception:
            pass

        # Step 5: Generate answer with fallback mode if needed
        answer, source_files = self.generate_answer(question, compressed_docs, use_fallback=use_fallback)
        # Save into session history so the assistant retains context in this CLI session
        try:
            if isinstance(answer, str):
                self.session_history.append((question, answer))
            else:
                self.session_history.append((question, str(answer)))
            # Keep memory bounded
            if len(self.session_history) > 20:
                self.session_history = self.session_history[-20:]
        except Exception:
            pass
        
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
    
    if len(sys.argv) < 2:
        print("Usage: python raptor_qa.py <user_id>")
        sys.exit(1)
    
    user_id = sys.argv[1]
    qa = RaptorQASystem(user_id)
    
    print("=" * 60)
    print("RAPTOR QA System - Interactive Mode")
    print("=" * 60)
    # Show assistant persona summary for the user
    print("Assistant: \n  I am your Research Assistant. You can talk to me conversationally — ask questions, request summaries, or ask for comparisons. If I need more information, I will ask a clarifying question.")
    print("Commands:")
    print("  - Type a question to get an answer")
    print("  - 'list' or 'docs' to see all uploaded documents")
    print("  - 'diagnose' to show DB diagnostics (counts per level, missing embeddings)")
    print("  - 'help' to print this message again")
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
                        source = doc['source_file'] or 'Unknown'
                        # Attempt to show file hash if present (may not be available for old docs)
                        try:
                            metadata_sample = json.loads(doc.get('metadata', '{}')) if doc.get('metadata') else {}
                            file_hash = metadata_sample.get('file_hash') or 'no-hash'
                        except Exception:
                            file_hash = 'no-hash'
                        print(f"  • {os.path.basename(source)} [hash: {file_hash[:8]}...]")
                        print(f"    - {doc['chunk_count']} chunks (Level {doc['min_level']}-{doc['max_level']})")
                else:
                    print("  No documents found.")
                print("=" * 60)
                continue

            # SUMMARIZE ALL - USE HIERARCHY SUMMARIES
            if question.lower() in ['summarize_all', 'summarizeall', 'summarize_docs']:
                print('\n' + '-' * 60)
                print(' Using RAPTOR hierarchy for summarization...')
                print('-' * 60)
                answer = qa.summarize_all_documents_from_hierarchy()
                print(answer)
                print('\n' + '-' * 60)
                continue

            # HELP
            if question.lower() in ['help', 'h', '?']:
                print("\n" + "-" * 60)
                print("Assistant help:")
                print("  - You can ask the assistant to summarize a document: 'Summarize w27392.pdf' or 'Give me main points of the uploaded papers' ")
                print("  - Ask factual questions: 'What is the sample size in w27392.pdf?' ")
                print("  - Ask comparative questions: 'Compare findings across documents' ")
                print("  - Ask clarifying questions if the assistant asks you for more info")
                print("  - 'list' shows uploaded documents")
                print("  - 'diagnose' prints DB diagnostics: chunk counts by level, missing embeddings etc.")
                print("  - 'quit' to exit")
                print("=" * 60)
                continue

            # CLEAR / RESET
            if question.lower() in ['clear', 'reset']:
                qa.session_history = []
                print("\nConversation context cleared. You can start fresh.")
                continue

            # DIAGNOSE
            if question.lower() in ['diagnose', 'diag']:
                try:
                    stats = qa.show_db_diagnostics()
                    print("\n" + "-" * 60)
                    print("DATABASE DIAGNOSTICS:")
                    print("-" * 60)
                    print(f"Distinct documents: {stats['distinct_docs']}")
                    print(f"Chunks per level: {stats['levels']}")
                    print(f"Missing embeddings: {stats['missing_embeddings']}")
                    print("Example files (top 5 by chunks):")
                    for ex in stats['examples']:
                        print(f"  • {ex['source_file']} ({ex['chunk_count']} chunks)")
                    print("-" * 60)
                except Exception as e:
                    print(f"Error fetching diagnostics: {e}")
                continue

            if not question:
                continue

            # Default: forward to QA engine
            result = qa.ask(question, verbose=True)

            print("\n" + "-" * 60)
            print("ANSWER:")
            print("-" * 60)
            print(result['answer'])
            if result.get('source_files'):
                print("\n[Sources: " + ", ".join(result['source_files']) + "]")
            print("\n" + "=" * 60)

    except (KeyboardInterrupt, EOFError):
        print('\nGoodbye! (interrupted)')
        return


if __name__ == "__main__":
    main()
