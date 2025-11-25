"""
Streaming Answer Generation for RAPTOR QA System
Provides real-time streamed responses from Gemini for better UX
"""
import os
import sys
import json
from typing import Generator, List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


@dataclass
class StreamConfig:
    """Configuration for streaming"""
    model_name: str = "gemini-2.5-flash"
    chunk_by_sentence: bool = True      # Yield complete sentences
    min_chunk_size: int = 10            # Minimum characters per yield
    show_sources_first: bool = True     # Show sources before streaming
    typing_effect_delay: float = 0.0    # Optional delay for typing effect (0 = disabled)


class StreamingAnswerGenerator:
    """
    Generate streaming answers from Gemini.
    Yields text chunks in real-time for progressive display.
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.config.model_name)
        
        # Assistant persona
        self.assistant_profile = (
            "You are a Research Assistant helping users explore research papers. "
            "Cite source documents (by filename) for factual answers. "
            "Clearly separate: facts from uploaded documents vs. your own knowledge. "
            "Mark assistant knowledge as '[From my knowledge base]' and document facts as '[From: filename]'."
        )
    
    def _build_context(self, documents: List[Dict]) -> Tuple[str, List[str]]:
        """Build context string from documents"""
        if not documents:
            return "", []
        
        context_parts = []
        sources = set()
        
        for i, doc in enumerate(documents, 1):
            level = doc.get('level', 0)
            content = doc['content'][:500]  # Limit per doc
            
            # Extract source
            try:
                metadata = doc.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                source = metadata.get('source_file', 'Unknown')
                if source and source != 'Unknown':
                    source_name = os.path.basename(source)
                    sources.add(source_name)
                else:
                    source_name = f'Cluster {i}'
            except:
                source_name = f'Chunk {i}'
            
            context_parts.append(f"[Source: {source_name}, Level {level}]:\n{content}\n")
        
        return "\n".join(context_parts), list(sources)
    
    def _build_prompt(self, question: str, context: str, 
                      history: Optional[List[Tuple[str, str]]] = None) -> str:
        """Build complete prompt for Gemini"""
        
        # Conversation history
        history_text = ""
        if history:
            last_turns = history[-2:]
            history_lines = []
            for uq, ua in last_turns:
                history_lines.append(f"User: {uq}\nAssistant: {ua}")
            history_text = "\n\nConversation so far:\n" + "\n".join(history_lines) + "\n\n"
        
        if context:
            return f"""{self.assistant_profile}
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
        else:
            return f"""{self.assistant_profile}

No relevant documents found in the uploaded files.
Answer using your knowledge base.

Question: {question}

Answer (mark as '[From my knowledge base]'):"""
    
    def stream_answer(self, question: str, 
                      context_docs: List[Dict],
                      history: Optional[List[Tuple[str, str]]] = None,
                      on_source: Optional[Callable[[List[str]], None]] = None) -> Generator[str, None, None]:
        """
        Stream answer generation.
        
        Args:
            question: User question
            context_docs: Retrieved context documents
            history: Conversation history
            on_source: Optional callback when sources are determined
        
        Yields:
            Text chunks as they are generated
        """
        # Build context
        context, sources = self._build_context(context_docs)
        
        # Notify about sources
        if on_source and sources:
            on_source(sources)
        
        # Show sources first if configured
        if self.config.show_sources_first and sources:
            yield f"[Sources: {', '.join(sources)}]\n\n"
        
        # Build prompt
        prompt = self._build_prompt(question, context, history)
        
        # Stream from Gemini
        try:
            response = self.model.generate_content(prompt, stream=True)
            
            buffer = ""
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    text = chunk.text
                    
                    if self.config.chunk_by_sentence:
                        # Accumulate and yield complete sentences
                        buffer += text
                        
                        # Find sentence boundaries
                        while True:
                            # Look for sentence endings
                            for end_char in ['. ', '? ', '! ', '.\n', '?\n', '!\n']:
                                idx = buffer.find(end_char)
                                if idx != -1:
                                    # Yield up to and including the end char
                                    yield buffer[:idx + len(end_char)]
                                    buffer = buffer[idx + len(end_char):]
                                    break
                            else:
                                # No complete sentence found
                                break
                    else:
                        # Yield raw chunks
                        if len(text) >= self.config.min_chunk_size:
                            yield text
                        else:
                            buffer += text
                            if len(buffer) >= self.config.min_chunk_size:
                                yield buffer
                                buffer = ""
            
            # Yield remaining buffer
            if buffer:
                yield buffer
                
        except Exception as e:
            yield f"\n\n[Error generating response: {e}]"
    
    def generate_answer(self, question: str, 
                        context_docs: List[Dict],
                        history: Optional[List[Tuple[str, str]]] = None) -> Tuple[str, List[str]]:
        """
        Non-streaming answer generation (for compatibility).
        
        Returns:
            (answer_text, source_list)
        """
        context, sources = self._build_context(context_docs)
        prompt = self._build_prompt(question, context, history)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip(), sources
        except Exception as e:
            return f"Error generating answer: {e}", sources


class StreamingQA:
    """
    Complete streaming QA system.
    Combines retrieval + streaming generation.
    """
    
    def __init__(self, user_id: str, stream_config: Optional[StreamConfig] = None):
        self.user_id = user_id
        self.generator = StreamingAnswerGenerator(config=stream_config)
        
        # Import QA system components
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from raptor_qa_incremental import RaptorQAIncremental
        
        self.qa = RaptorQAIncremental(user_id)
    
    def ask_streaming(self, question: str, 
                      verbose: bool = False,
                      on_source: Optional[Callable[[List[str]], None]] = None) -> Generator[str, None, Dict]:
        """
        Ask a question with streaming response.
        
        Args:
            question: User question
            verbose: Show debug info
            on_source: Callback when sources are known
        
        Yields:
            Text chunks as generated
        
        Returns:
            Final result dict (via generator.send() or after iteration)
        """
        # Classify question
        question_type = self.qa.classify_question(question)
        
        if verbose:
            yield f"[Classification: {question_type.value}]\n"
        
        # Retrieve context
        retrieved_docs = self.qa.retrieve_adaptive(question, question_type)
        
        if verbose:
            yield f"[Retrieved: {len(retrieved_docs)} documents]\n"
        
        # Compress context
        compressed_docs = self.qa.compress_context(retrieved_docs, question)
        
        if verbose:
            yield f"[Using: {len(compressed_docs)} chunks]\n\n"
        
        # Stream answer
        full_answer = ""
        for chunk in self.generator.stream_answer(
            question, 
            compressed_docs,
            self.qa.session_history,
            on_source
        ):
            full_answer += chunk
            yield chunk
        
        # Save to history
        self.qa.session_history.append((question, full_answer))
        
        # Return result
        return {
            'question': question,
            'question_type': question_type.value,
            'num_retrieved': len(retrieved_docs),
            'num_used': len(compressed_docs),
            'answer': full_answer
        }


def print_streaming(generator: Generator[str, None, None]):
    """Helper to print streaming output"""
    import sys
    for chunk in generator:
        print(chunk, end='', flush=True)
    print()  # Final newline


# Interactive CLI
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from auth import AuthManager
    
    # Check auth
    auth = AuthManager()
    user_id = auth.load_session()
    
    if not user_id:
        print(" Not logged in! Please run:")
        print("   python auth.py login <username> <password>")
        sys.exit(1)
    
    print("=" * 60)
    print("STREAMING RAPTOR QA")
    print("=" * 60)
    print(f"Logged in as: {auth.current_user['username']}")
    print("\nType 'quit' to exit\n")
    
    streaming_qa = StreamingQA(user_id)
    
    try:
        while True:
            question = input("\n Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\n Answer: ", end='', flush=True)
            
            # Stream the answer
            for chunk in streaming_qa.ask_streaming(question, verbose=False):
                print(chunk, end='', flush=True)
            
            print("\n")
    
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
