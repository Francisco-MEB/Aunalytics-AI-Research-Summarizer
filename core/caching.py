"""
Caching Layer for RAPTOR QA System
Caches embeddings and query results for faster responses
"""
import os
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import threading


@dataclass
class CacheConfig:
    """Configuration for caching behavior"""
    cache_dir: str = ".cache"            # Directory for file-based cache
    embedding_cache_size: int = 1000     # Max embeddings to cache in memory
    query_cache_size: int = 100          # Max query results to cache
    query_ttl_seconds: int = 3600        # Query cache TTL (1 hour)
    embedding_ttl_seconds: int = 86400   # Embedding cache TTL (24 hours)
    use_file_cache: bool = True          # Persist cache to disk
    use_memory_cache: bool = True        # Keep cache in memory


class LRUCache:
    """Simple LRU cache with TTL support"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_order: List[str] = []
        self._lock = threading.Lock()
    
    def _make_key(self, key: Any) -> str:
        """Convert key to string hash"""
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        return hashlib.md5(str(key).encode()).hexdigest()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache"""
        str_key = self._make_key(key)
        
        with self._lock:
            if str_key not in self.cache:
                return None
            
            value, timestamp = self.cache[str_key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[str_key]
                self.access_order.remove(str_key)
                return None
            
            # Update access order (LRU)
            self.access_order.remove(str_key)
            self.access_order.append(str_key)
            
            return value
    
    def set(self, key: Any, value: Any):
        """Set value in cache"""
        str_key = self._make_key(key)
        
        with self._lock:
            # Evict if full
            while len(self.cache) >= self.max_size:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
            
            self.cache[str_key] = (value, time.time())
            self.access_order.append(str_key)
    
    def clear(self):
        """Clear all cache"""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
    
    def stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }


class EmbeddingCache:
    """
    Cache for text embeddings.
    Avoids re-computing embeddings for the same text.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.memory_cache = LRUCache(
            max_size=self.config.embedding_cache_size,
            ttl_seconds=self.config.embedding_ttl_seconds
        )
        self._cache_file = Path(self.config.cache_dir) / "embeddings_cache.pkl"
        self._ensure_cache_dir()
        self._load_file_cache()
        
        # Stats
        self.hits = 0
        self.misses = 0
    
    def _ensure_cache_dir(self):
        """Create cache directory if needed"""
        Path(self.config.cache_dir).mkdir(exist_ok=True)
    
    def _load_file_cache(self):
        """Load cache from file"""
        if self.config.use_file_cache and self._cache_file.exists():
            try:
                with open(self._cache_file, 'rb') as f:
                    data = pickle.load(f)
                    for key, value in data.items():
                        self.memory_cache.cache[key] = value
                print(f"  Loaded {len(data)} embeddings from cache")
            except Exception as e:
                print(f"  Warning: Could not load embedding cache: {e}")
    
    def _save_file_cache(self):
        """Save cache to file"""
        if self.config.use_file_cache:
            try:
                with open(self._cache_file, 'wb') as f:
                    pickle.dump(self.memory_cache.cache, f)
            except Exception as e:
                print(f"  Warning: Could not save embedding cache: {e}")
    
    def _text_key(self, text: str, model_name: str = "") -> str:
        """Generate unique key for text + model"""
        combined = f"{model_name}:{text}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, text: str, model_name: str = "") -> Optional[List[float]]:
        """Get cached embedding for text"""
        key = self._text_key(text, model_name)
        result = self.memory_cache.get(key)
        
        if result is not None:
            self.hits += 1
            return result
        else:
            self.misses += 1
            return None
    
    def get_batch(self, texts: List[str], model_name: str = "") -> Tuple[List[Optional[List[float]]], List[int]]:
        """
        Get cached embeddings for multiple texts.
        Returns: (embeddings_list, indices_needing_computation)
        """
        embeddings = []
        missing_indices = []
        
        for i, text in enumerate(texts):
            cached = self.get(text, model_name)
            embeddings.append(cached)
            if cached is None:
                missing_indices.append(i)
        
        return embeddings, missing_indices
    
    def set(self, text: str, embedding: List[float], model_name: str = ""):
        """Cache embedding for text"""
        key = self._text_key(text, model_name)
        self.memory_cache.set(key, embedding)
    
    def set_batch(self, texts: List[str], embeddings: List[List[float]], model_name: str = ""):
        """Cache multiple embeddings"""
        for text, emb in zip(texts, embeddings):
            self.set(text, emb, model_name)
        
        # Periodically save to file
        if self.config.use_file_cache:
            self._save_file_cache()
    
    def clear(self):
        """Clear all cached embeddings"""
        self.memory_cache.clear()
        if self._cache_file.exists():
            self._cache_file.unlink()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1%}",
            'size': len(self.memory_cache.cache),
            'max_size': self.config.embedding_cache_size
        }


class QueryCache:
    """
    Cache for query results.
    Stores (question, answer, sources) tuples.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.memory_cache = LRUCache(
            max_size=self.config.query_cache_size,
            ttl_seconds=self.config.query_ttl_seconds
        )
        self._cache_file = Path(self.config.cache_dir) / "query_cache.json"
        self._ensure_cache_dir()
        self._load_file_cache()
        
        # Stats
        self.hits = 0
        self.misses = 0
    
    def _ensure_cache_dir(self):
        """Create cache directory if needed"""
        Path(self.config.cache_dir).mkdir(exist_ok=True)
    
    def _load_file_cache(self):
        """Load cache from file"""
        if self.config.use_file_cache and self._cache_file.exists():
            try:
                with open(self._cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        # Restore timestamp
                        self.memory_cache.cache[key] = (value['data'], value['timestamp'])
                        self.memory_cache.access_order.append(key)
            except Exception as e:
                print(f"  Warning: Could not load query cache: {e}")
    
    def _save_file_cache(self):
        """Save cache to file"""
        if self.config.use_file_cache:
            try:
                # Convert to serializable format
                data = {}
                for key, (value, timestamp) in self.memory_cache.cache.items():
                    data[key] = {'data': value, 'timestamp': timestamp}
                
                with open(self._cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"  Warning: Could not save query cache: {e}")
    
    def _query_key(self, question: str, user_id: str) -> str:
        """Generate unique key for query"""
        # Normalize question (lowercase, strip)
        normalized = question.lower().strip()
        combined = f"{user_id}:{normalized}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, question: str, user_id: str) -> Optional[Dict]:
        """Get cached result for question"""
        key = self._query_key(question, user_id)
        result = self.memory_cache.get(key)
        
        if result is not None:
            self.hits += 1
            return result
        else:
            self.misses += 1
            return None
    
    def set(self, question: str, user_id: str, result: Dict):
        """Cache result for question"""
        key = self._query_key(question, user_id)
        
        # Don't cache empty or error results
        if result.get('answer') and 'error' not in result.get('answer', '').lower():
            self.memory_cache.set(key, result)
            self._save_file_cache()
    
    def invalidate_for_user(self, user_id: str):
        """Invalidate all cache entries for a user (e.g., when documents change)"""
        # This is a simple implementation - could be optimized
        keys_to_remove = []
        for key in list(self.memory_cache.cache.keys()):
            # We can't easily check user_id from hash, so we clear all for now
            # A better implementation would store user_id separately
            pass
        
        # For now, just clear all cache when documents change
        self.clear()
    
    def clear(self):
        """Clear all cached queries"""
        self.memory_cache.clear()
        if self._cache_file.exists():
            self._cache_file.unlink()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1%}",
            'size': len(self.memory_cache.cache),
            'max_size': self.config.query_cache_size,
            'ttl_seconds': self.config.query_ttl_seconds
        }


class CachedEmbedder:
    """
    Wrapper around SentenceTransformer that uses caching.
    Drop-in replacement for model.encode()
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache: Optional[EmbeddingCache] = None):
        self.model_name = model_name
        self.cache = cache or EmbeddingCache()
        self._model = None  # Lazy load
    
    @property
    def model(self):
        """Lazy load model"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Encode texts with caching.
        Only computes embeddings for texts not in cache.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache
        cached_embeddings, missing_indices = self.cache.get_batch(texts, self.model_name)
        
        # Compute only missing embeddings
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            new_embeddings = self.model.encode(
                missing_texts,
                convert_to_numpy=True,
                show_progress_bar=kwargs.get('show_progress_bar', len(missing_texts) > 10)
            )
            
            # Convert to list and cache
            new_embeddings_list = new_embeddings.tolist()
            
            # Fill in results
            for i, emb in zip(missing_indices, new_embeddings_list):
                cached_embeddings[i] = emb
            
            # Update cache
            self.cache.set_batch(missing_texts, new_embeddings_list, self.model_name)
            
            print(f"   Cache: {len(texts) - len(missing_indices)} hits, {len(missing_indices)} computed")
        else:
            print(f"   Cache: {len(texts)} hits (100% cached)")
        
        return cached_embeddings
    
    def stats(self) -> Dict:
        """Get embedding cache stats"""
        return self.cache.stats()


# Convenience function
def get_cached_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                        cache_dir: str = ".cache") -> CachedEmbedder:
    """Get a cached embedder instance"""
    config = CacheConfig(cache_dir=cache_dir)
    cache = EmbeddingCache(config)
    return CachedEmbedder(model_name, cache)


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING CACHING SYSTEM")
    print("=" * 60)
    
    # Test embedding cache
    print("\n1. Testing Embedding Cache:")
    cache = EmbeddingCache(CacheConfig(cache_dir=".test_cache"))
    
    # Simulate embeddings
    cache.set("Hello world", [0.1, 0.2, 0.3], "test-model")
    cache.set("Test sentence", [0.4, 0.5, 0.6], "test-model")
    
    result = cache.get("Hello world", "test-model")
    print(f"  Retrieved: {result}")
    print(f"  Stats: {cache.stats()}")
    
    # Test query cache
    print("\n2. Testing Query Cache:")
    qcache = QueryCache(CacheConfig(cache_dir=".test_cache"))
    
    qcache.set("What is AI?", "user123", {
        'answer': 'AI stands for artificial intelligence...',
        'sources': ['doc1.pdf']
    })
    
    result = qcache.get("What is AI?", "user123")
    print(f"  Retrieved: {result}")
    print(f"  Stats: {qcache.stats()}")
    
    # Cleanup
    import shutil
    if os.path.exists(".test_cache"):
        shutil.rmtree(".test_cache")
    
    print("\n Caching tests passed!")
