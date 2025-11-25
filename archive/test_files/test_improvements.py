"""
Interactive Demo: Test All System Improvements
Run this to see chunking, caching, batch processing, and streaming in action!
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_chunking():
    """Test different chunking strategies"""
    print("\n" + "=" * 60)
    print("TEST 1: CHUNKING STRATEGIES")
    print("=" * 60)
    
    from utils.chunking import chunk_text
    
    # Read sample text
    with open('data/sample.txt', 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()[:5000]  # First 5000 chars
    
    print(f"Input text: {len(text)} characters\n")
    
    for strategy in ['fixed', 'semantic', 'sliding']:
        chunks = chunk_text(text, chunk_size=800, chunk_overlap=100, strategy=strategy)
        print(f"{strategy.upper()}: {len(chunks)} chunks")
        for i, c in enumerate(chunks[:2]):
            preview = c['text'][:60].replace('\n', ' ')
            print(f"  Chunk {i}: {len(c['text'])} chars - {preview}...")
        print()
    
    print(" Chunking test complete!")
    return True


def test_caching():
    """Test embedding cache"""
    print("\n" + "=" * 60)
    print("TEST 2: EMBEDDING CACHE")
    print("=" * 60)
    
    from utils.caching import EmbeddingCache, CacheConfig
    import time
    
    config = CacheConfig(cache_dir=".test_cache")
    cache = EmbeddingCache(config)
    
    # Simulate caching
    texts = ["Hello world", "Machine learning is great", "RAPTOR uses hierarchical retrieval"]
    
    print("First pass (cache miss):")
    for text in texts:
        result = cache.get(text, "test-model")
        status = "HIT" if result else "MISS"
        print(f"  '{text[:30]}...' -> {status}")
    
    # Add to cache
    print("\nAdding to cache...")
    for i, text in enumerate(texts):
        cache.set(text, [0.1 * i, 0.2 * i, 0.3 * i], "test-model")
    
    print("\nSecond pass (cache hit):")
    for text in texts:
        result = cache.get(text, "test-model")
        status = "HIT" if result else "MISS"
        print(f"  '{text[:30]}...' -> {status}")
    
    print(f"\nCache stats: {cache.stats()}")
    
    # Cleanup
    import shutil
    if os.path.exists(".test_cache"):
        shutil.rmtree(".test_cache")
    
    print(" Caching test complete!")
    return True


def test_clustering():
    """Test clustering algorithms"""
    print("\n" + "=" * 60)
    print("TEST 3: CLUSTERING ALGORITHMS")
    print("=" * 60)
    
    import numpy as np
    from utils.clustering import cluster_embeddings
    
    # Generate sample embeddings (5 clusters)
    np.random.seed(42)
    n_samples = 50
    n_features = 384
    
    centers = np.random.randn(5, n_features)
    embeddings = []
    for i in range(n_samples):
        center = centers[i % 5]
        point = center + np.random.randn(n_features) * 0.3
        embeddings.append(point)
    embeddings = np.array(embeddings)
    
    print(f"Input: {n_samples} embeddings, {n_features} dimensions\n")
    
    for method in ['kmeans', 'hierarchical', 'adaptive']:
        result = cluster_embeddings(embeddings, method=method)
        sil = f"{result.silhouette_score:.3f}" if result.silhouette_score else "N/A"
        print(f"{method.upper()}: {result.n_clusters} clusters, silhouette={sil}")
    
    print("\n Clustering test complete!")
    return True


def test_add_document():
    """Test adding a document with new chunking"""
    print("\n" + "=" * 60)
    print("TEST 4: ADD DOCUMENT (with semantic chunking)")
    print("=" * 60)
    
    from auth import AuthManager
    
    auth = AuthManager()
    user_id = auth.load_session()
    
    if not user_id:
        print(" Not logged in! Run: python auth.py login <user> <pass>")
        return False
    
    print(f"Logged in as: {auth.current_user['username']}")
    
    # Use small test file
    test_file = "data/sample_small.txt"
    if not os.path.exists(test_file):
        # Create it if missing
        with open("data/sample.txt", 'r', encoding='utf-8', errors='ignore') as f:
            small_text = f.read()[:20000]  # ~20KB
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(small_text)
        print(f"Created small test file: {test_file}")
    
    file_size = os.path.getsize(test_file)
    print(f"Test file: {test_file} ({file_size:,} bytes)")
    
    response = input(f"\nAdd {test_file} to database? (y/n): ").strip().lower()
    if response != 'y':
        print("Skipped.")
        return True
    
    # Import and run
    from add_document import add_document
    add_document(test_file, user_id, chunking_strategy="semantic")
    
    print(" Document add test complete!")
    return True


def test_streaming():
    """Test streaming QA"""
    print("\n" + "=" * 60)
    print("TEST 5: STREAMING QA")
    print("=" * 60)
    
    from auth import AuthManager
    
    auth = AuthManager()
    user_id = auth.load_session()
    
    if not user_id:
        print(" Not logged in!")
        return False
    
    print(f"Logged in as: {auth.current_user['username']}")
    
    question = input("\nEnter a question (or press Enter to skip): ").strip()
    if not question:
        print("Skipped.")
        return True
    
    print("\n Streaming answer:\n")
    
    from streaming_qa import StreamingQA
    qa = StreamingQA(user_id)
    
    for chunk in qa.ask_streaming(question, verbose=True):
        print(chunk, end='', flush=True)
    
    print("\n\n Streaming test complete!")
    return True


def test_batch_processor():
    """Test batch processing (dry run)"""
    print("\n" + "=" * 60)
    print("TEST 6: BATCH PROCESSOR (dry run)")
    print("=" * 60)
    
    from auth import AuthManager
    from batch_processor import BatchProcessor, BatchConfig
    
    auth = AuthManager()
    user_id = auth.load_session()
    
    if not user_id:
        print(" Not logged in!")
        return False
    
    config = BatchConfig(
        chunk_size=1000,
        chunk_overlap=200,
        chunking_strategy="semantic",
        dry_run=True,  # Preview only
        verbose=True
    )
    
    processor = BatchProcessor(config, user_id)
    
    # Find files in data/
    print("\nSearching for files in data/...")
    result = processor.run(["data/"])
    
    print(f"\nDry run results:")
    print(f"  Files found: {result.total_files}")
    print(f"  Would create: {result.total_chunks} chunks")
    
    print(" Batch processor test complete!")
    return True


def main():
    print("=" * 60)
    print("RAPTOR SYSTEM IMPROVEMENTS - INTERACTIVE DEMO")
    print("=" * 60)
    print("\nThis will test all the new improvements:")
    print("  1. Chunking strategies")
    print("  2. Embedding cache")
    print("  3. Clustering algorithms")
    print("  4. Add document (optional)")
    print("  5. Streaming QA (optional)")
    print("  6. Batch processor (dry run)")
    print()
    
    tests = [
        ("Chunking", test_chunking),
        ("Caching", test_caching),
        ("Clustering", test_clustering),
        ("Add Document", test_add_document),
        ("Streaming QA", test_streaming),
        ("Batch Processor", test_batch_processor),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n Error in {name}: {e}")
            results.append((name, False))
        
        input("\nPress Enter to continue...")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success in results:
        status = "" if success else ""
        print(f"  {status} {name}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
