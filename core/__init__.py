"""
Core RAPTOR System Components
=============================
This package contains the main RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
implementation with incremental updates support.

Main Components:
- IncrementalRaptorBuilder: Fast document add/delete without full tree rebuild
- Chunking utilities: Multiple strategies for document chunking
- Clustering utilities: Smart clustering with multiple algorithms
- Caching: Embedding and query caching for performance
"""

from .raptor import IncrementalRaptorBuilder
from .chunking import chunk_text, SemanticChunker, ChunkingStrategy
from .clustering import SmartClusterer, ClusteringMethod
from .caching import EmbeddingCache, QueryCache

__all__ = [
    'IncrementalRaptorBuilder',
    'chunk_text',
    'SemanticChunker', 
    'ChunkingStrategy',
    'SmartClusterer',
    'ClusteringMethod',
    'EmbeddingCache',
    'QueryCache'
]
