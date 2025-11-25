"""
Advanced Chunking Strategies for Research Documents
Supports semantic-aware chunking, sentence boundaries, and configurable parameters
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    FIXED = "fixed"              # Simple fixed-size chunks (original)
    SEMANTIC = "semantic"        # Respects sentence/paragraph boundaries
    HIERARCHICAL = "hierarchical"  # Creates nested chunks (paragraph -> sentence)
    SLIDING_WINDOW = "sliding"   # Overlapping sliding window with context


@dataclass
class ChunkConfig:
    """Configuration for chunking behavior"""
    chunk_size: int = 1000       # Target chunk size in characters
    chunk_overlap: int = 200     # Overlap between chunks
    min_chunk_size: int = 100    # Minimum chunk size (avoid tiny chunks)
    max_chunk_size: int = 2000   # Maximum chunk size (avoid huge chunks)
    respect_sentences: bool = True   # Try to break at sentence boundaries
    respect_paragraphs: bool = True  # Try to break at paragraph boundaries
    include_metadata: bool = True    # Add position/context metadata
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC


class SemanticChunker:
    """
    Intelligent chunking that respects document structure.
    Produces better embeddings by keeping semantic units together.
    """
    
    # Sentence boundary patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    PARAGRAPH_BOUNDARY = re.compile(r'\n\s*\n')
    
    # Section header patterns (for research papers)
    SECTION_HEADERS = re.compile(
        r'^\s*(?:'
        r'(?:\d+\.?\s*)?(?:abstract|introduction|background|methodology|methods|'
        r'results|discussion|conclusion|references|acknowledgment|appendix)'
        r'|(?:\d+\.)+\s+\w+'  # Numbered sections like "1.2 Methods"
        r')\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove null bytes
        text = text.replace('\x00', '')
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove excessive newlines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while handling edge cases"""
        # Handle common abbreviations that shouldn't trigger sentence breaks
        protected_text = text
        abbreviations = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'vs.', 'etc.', 
                        'i.e.', 'e.g.', 'Fig.', 'et al.', 'Inc.', 'Ltd.', 'Jr.', 'Sr.']
        
        placeholders = {}
        for i, abbr in enumerate(abbreviations):
            placeholder = f"__ABBR{i}__"
            placeholders[placeholder] = abbr
            protected_text = protected_text.replace(abbr, placeholder)
        
        # Split on sentence boundaries
        sentences = self.SENTENCE_ENDINGS.split(protected_text)
        
        # Restore abbreviations
        restored_sentences = []
        for sent in sentences:
            for placeholder, abbr in placeholders.items():
                sent = sent.replace(placeholder, abbr)
            sent = sent.strip()
            if sent:
                restored_sentences.append(sent)
        
        return restored_sentences
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = self.PARAGRAPH_BOUNDARY.split(text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _find_section_boundaries(self, text: str) -> List[Tuple[int, str]]:
        """Find section headers in research paper format"""
        boundaries = []
        for match in self.SECTION_HEADERS.finditer(text):
            boundaries.append((match.start(), match.group().strip()))
        return boundaries
    
    def _merge_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Merge chunks that are too small"""
        if not chunks:
            return chunks
        
        merged = []
        current = None
        
        for chunk in chunks:
            if current is None:
                current = chunk.copy()
            elif len(current['text']) + len(chunk['text']) + 1 <= self.config.max_chunk_size:
                # Merge with current
                current['text'] = current['text'] + '\n\n' + chunk['text']
                current['metadata']['end_idx'] = chunk['metadata'].get('end_idx', 0)
            else:
                # Save current and start new
                if len(current['text']) >= self.config.min_chunk_size:
                    merged.append(current)
                current = chunk.copy()
        
        # Don't forget the last chunk
        if current and len(current['text']) >= self.config.min_chunk_size:
            merged.append(current)
        
        return merged
    
    def chunk_fixed(self, text: str) -> List[Dict]:
        """Simple fixed-size chunking (original behavior)"""
        text = self._clean_text(text)
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            
            # Adjust to word boundary
            if end < len(text):
                # Look for last space within chunk
                space_idx = text.rfind(' ', start, end)
                if space_idx > start + self.config.min_chunk_size:
                    end = space_idx
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'chunk_index': chunk_idx,
                        'start_char': start,
                        'end_char': end,
                        'strategy': 'fixed'
                    }
                })
                chunk_idx += 1
            
            # Move start with overlap
            start = end - self.config.chunk_overlap
            if start <= 0:
                start = end
        
        return chunks
    
    def chunk_semantic(self, text: str) -> List[Dict]:
        """
        Semantic chunking that respects document structure.
        Prioritizes: paragraph boundaries > sentence boundaries > word boundaries
        """
        text = self._clean_text(text)
        
        # First, split into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_idx = 0
        char_pos = 0
        
        for para in paragraphs:
            para_len = len(para)
            
            # If paragraph alone exceeds max, split by sentences
            if para_len > self.config.max_chunk_size:
                # Save current chunk first
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            'chunk_index': chunk_idx,
                            'start_char': current_start,
                            'end_char': char_pos,
                            'strategy': 'semantic',
                            'boundary_type': 'paragraph'
                        }
                    })
                    chunk_idx += 1
                    current_chunk = ""
                    current_start = char_pos
                
                # Split large paragraph by sentences
                sentences = self._split_into_sentences(para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 <= self.config.chunk_size:
                        current_chunk = (current_chunk + ' ' + sent).strip()
                    else:
                        if current_chunk.strip():
                            chunks.append({
                                'text': current_chunk.strip(),
                                'metadata': {
                                    'chunk_index': chunk_idx,
                                    'start_char': current_start,
                                    'end_char': char_pos,
                                    'strategy': 'semantic',
                                    'boundary_type': 'sentence'
                                }
                            })
                            chunk_idx += 1
                        current_chunk = sent
                        current_start = char_pos
                
            # If adding paragraph fits in target size
            elif len(current_chunk) + para_len + 2 <= self.config.chunk_size:
                current_chunk = (current_chunk + '\n\n' + para).strip()
            
            # Paragraph doesn't fit - save current and start new
            else:
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            'chunk_index': chunk_idx,
                            'start_char': current_start,
                            'end_char': char_pos,
                            'strategy': 'semantic',
                            'boundary_type': 'paragraph'
                        }
                    })
                    chunk_idx += 1
                current_chunk = para
                current_start = char_pos
            
            char_pos += para_len + 2  # Account for paragraph separator
        
        # Don't forget last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    'chunk_index': chunk_idx,
                    'start_char': current_start,
                    'end_char': char_pos,
                    'strategy': 'semantic',
                    'boundary_type': 'end'
                }
            })
        
        return self._merge_small_chunks(chunks)
    
    def chunk_sliding_window(self, text: str) -> List[Dict]:
        """
        Sliding window with larger overlap for better context continuity.
        Good for maintaining context across chunk boundaries.
        """
        text = self._clean_text(text)
        sentences = self._split_into_sentences(text)
        
        chunks = []
        chunk_idx = 0
        i = 0
        
        while i < len(sentences):
            # Build chunk from sentences
            chunk_sentences = []
            chunk_len = 0
            start_sent_idx = i
            
            while i < len(sentences) and chunk_len < self.config.chunk_size:
                sent = sentences[i]
                if chunk_len + len(sent) + 1 <= self.config.max_chunk_size:
                    chunk_sentences.append(sent)
                    chunk_len += len(sent) + 1
                    i += 1
                else:
                    break
            
            if chunk_sentences:
                chunk_text = ' '.join(chunk_sentences)
                
                # Calculate overlap in terms of sentences
                overlap_sentences = max(1, int(len(chunk_sentences) * 0.3))  # 30% overlap
                
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'chunk_index': chunk_idx,
                        'start_sentence_idx': start_sent_idx,
                        'end_sentence_idx': i - 1,
                        'num_sentences': len(chunk_sentences),
                        'strategy': 'sliding_window',
                        'overlap_sentences': overlap_sentences
                    }
                })
                chunk_idx += 1
                
                # Move back by overlap
                i = max(start_sent_idx + 1, i - overlap_sentences)
        
        return chunks
    
    def chunk_hierarchical(self, text: str) -> List[Dict]:
        """
        Hierarchical chunking: creates parent-child chunk relationships.
        Parent = larger context, Children = detailed sentences.
        Useful for multi-resolution retrieval.
        """
        text = self._clean_text(text)
        paragraphs = self._split_into_paragraphs(text)
        
        chunks = []
        chunk_idx = 0
        parent_idx = 0
        
        for para in paragraphs:
            # Create parent chunk (full paragraph or paragraph group)
            parent_id = f"parent_{parent_idx}"
            
            if len(para) > self.config.min_chunk_size:
                chunks.append({
                    'text': para,
                    'metadata': {
                        'chunk_index': chunk_idx,
                        'hierarchy_level': 'parent',
                        'parent_id': parent_id,
                        'strategy': 'hierarchical'
                    }
                })
                chunk_idx += 1
                
                # Create child chunks (sentences)
                sentences = self._split_into_sentences(para)
                for sent_idx, sent in enumerate(sentences):
                    if len(sent) >= 50:  # Skip very short sentences
                        chunks.append({
                            'text': sent,
                            'metadata': {
                                'chunk_index': chunk_idx,
                                'hierarchy_level': 'child',
                                'parent_id': parent_id,
                                'sentence_index': sent_idx,
                                'strategy': 'hierarchical'
                            }
                        })
                        chunk_idx += 1
                
                parent_idx += 1
        
        return chunks
    
    def chunk(self, text: str, strategy: Optional[ChunkingStrategy] = None) -> List[Dict]:
        """
        Main chunking method - dispatches to appropriate strategy.
        
        Args:
            text: Input text to chunk
            strategy: Override default strategy (optional)
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        strategy = strategy or self.config.strategy
        
        if strategy == ChunkingStrategy.FIXED:
            return self.chunk_fixed(text)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self.chunk_semantic(text)
        elif strategy == ChunkingStrategy.SLIDING_WINDOW:
            return self.chunk_sliding_window(text)
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            return self.chunk_hierarchical(text)
        else:
            return self.chunk_semantic(text)  # Default


def chunk_text(text: str, 
               chunk_size: int = 1000, 
               chunk_overlap: int = 200,
               strategy: str = "semantic") -> List[Dict]:
    """
    Convenience function for chunking text.
    Drop-in replacement for original chunk_text function.
    
    Args:
        text: Input text
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        strategy: One of "fixed", "semantic", "sliding", "hierarchical"
    
    Returns:
        List of chunk dictionaries
    """
    strategy_map = {
        "fixed": ChunkingStrategy.FIXED,
        "semantic": ChunkingStrategy.SEMANTIC,
        "sliding": ChunkingStrategy.SLIDING_WINDOW,
        "hierarchical": ChunkingStrategy.HIERARCHICAL
    }
    
    config = ChunkConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy_map.get(strategy, ChunkingStrategy.SEMANTIC)
    )
    
    chunker = SemanticChunker(config)
    return chunker.chunk(text)


# Quick test
if __name__ == "__main__":
    sample_text = """
    Introduction
    
    This is the first paragraph of the introduction. It contains multiple sentences 
    about the topic. Dr. Smith and colleagues conducted research on this subject.
    
    The second paragraph continues the discussion. It provides additional context
    and background information. See Fig. 1 for the results visualization.
    
    Methods
    
    We used a novel approach combining machine learning with traditional methods.
    The data was collected from various sources including academic papers.
    Statistical analysis was performed using standard techniques.
    
    Results
    
    Our findings show significant improvement over baseline methods. The accuracy
    increased by 15% compared to previous work. This represents a major advancement
    in the field.
    """
    
    print("=" * 60)
    print("TESTING CHUNKING STRATEGIES")
    print("=" * 60)
    
    config = ChunkConfig(chunk_size=300, chunk_overlap=50)
    chunker = SemanticChunker(config)
    
    for strategy in ChunkingStrategy:
        print(f"\n{strategy.value.upper()} CHUNKING:")
        print("-" * 40)
        chunks = chunker.chunk(sample_text, strategy)
        for i, chunk in enumerate(chunks):
            preview = chunk['text'][:100].replace('\n', ' ')
            print(f"  Chunk {i}: {len(chunk['text'])} chars - {preview}...")
        print(f"  Total: {len(chunks)} chunks")
