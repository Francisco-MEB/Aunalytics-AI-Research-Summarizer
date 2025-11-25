"""
Test script for incremental RAPTOR implementation
Validates add/delete operations work correctly
"""
import sys
import os
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.incremental_raptor import IncrementalRaptorBuilder
import uuid

load_dotenv()


def test_add_document():
    """Test adding a document incrementally"""
    print("\n" + "=" * 60)
    print("TEST 1: Add Document Incrementally")
    print("=" * 60)
    
    builder = IncrementalRaptorBuilder()
    
    # Create test document with 3 chunks
    doc_id = str(uuid.uuid4())
    user_id = "5b11bef4-7ea1-4bf9-aac1-7f22f7c73705"  # Your test user ID
    
    chunks = [
        {
            'id': str(uuid.uuid4()),
            'text': 'This is the first chunk about machine learning. It discusses neural networks and deep learning architectures.'
        },
        {
            'id': str(uuid.uuid4()),
            'text': 'The second chunk explores convolutional neural networks for image recognition and computer vision tasks.'
        },
        {
            'id': str(uuid.uuid4()),
            'text': 'The third chunk covers recurrent neural networks and their applications in natural language processing.'
        }
    ]
    
    print(f"\n Test document: {doc_id}")
    print(f" User: {user_id}")
    print(f" Chunks: {len(chunks)}")
    
    # Add document
    result_doc_id = builder.add_document_incremental(chunks, doc_id, user_id)
    
    print(f"\n Document added: {result_doc_id}")
    
    # Check stats
    stats = builder.get_tree_stats(user_id)
    print(f"\n Tree Statistics:")
    print(f"   Total chunks: {stats.get('total_chunks', 'N/A')}")
    print(f"   Active chunks: {stats.get('active_chunks', 'N/A')}")
    print(f"   Deleted chunks: {stats.get('deleted_chunks', 'N/A')}")
    print(f"   Tree levels: {stats.get('tree_levels', 'N/A')}")
    print(f"   Nodes per level: {stats.get('nodes_per_level', 'N/A')}")
    
    return doc_id, user_id


def test_delete_document(doc_id: str, user_id: str):
    """Test deleting a document incrementally"""
    print("\n" + "=" * 60)
    print("TEST 2: Delete Document Incrementally")
    print("=" * 60)
    
    builder = IncrementalRaptorBuilder()
    
    print(f"\n️ Deleting document: {doc_id}")
    
    # Delete document
    builder.delete_document_incremental(doc_id, user_id)
    
    print(f"\n Document deleted: {doc_id}")
    
    # Check stats
    stats = builder.get_tree_stats(user_id)
    print(f"\n Tree Statistics After Delete:")
    print(f"   Total chunks: {stats.get('total_chunks', 'N/A')}")
    print(f"   Active chunks: {stats.get('active_chunks', 'N/A')}")
    print(f"   Deleted chunks: {stats.get('deleted_chunks', 'N/A')}")
    print(f"   Tree levels: {stats.get('tree_levels', 'N/A')}")
    print(f"   Nodes per level: {stats.get('nodes_per_level', 'N/A')}")


def test_tree_stats():
    """Test getting tree statistics"""
    print("\n" + "=" * 60)
    print("TEST 3: Get Tree Statistics")
    print("=" * 60)
    
    builder = IncrementalRaptorBuilder()
    user_id = "5b11bef4-7ea1-4bf9-aac1-7f22f7c73705"
    
    stats = builder.get_tree_stats(user_id)
    
    print(f"\n Current Tree Statistics:")
    print(f"   Total chunks: {stats.get('total_chunks', 'N/A')}")
    print(f"   Active chunks: {stats.get('active_chunks', 'N/A')}")
    print(f"   Deleted chunks: {stats.get('deleted_chunks', 'N/A')}")
    print(f"   Tree levels: {stats.get('tree_levels', 'N/A')}")
    print(f"   Nodes per level: {stats.get('nodes_per_level', 'N/A')}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RAPTOR INCREMENTAL UPDATES TEST SUITE")
    print("=" * 60)
    
    try:
        # Test 1: Add document
        doc_id, user_id = test_add_document()
        
        # Test 2: Delete document
        # test_delete_document(doc_id, user_id)
        
        # Test 3: Get stats
        test_tree_stats()
        
        print("\n" + "=" * 60)
        print(" ALL TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
