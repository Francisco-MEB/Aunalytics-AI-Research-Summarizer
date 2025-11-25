"""
Delete a document from the incremental RAPTOR tree
Usage: python delete_document.py <doc_id> <user_id>
"""

import sys
import os
from dotenv import load_dotenv
from embeddings.incremental_raptor import IncrementalRAPTORBuilder
from auth import AuthManager

# Load environment variables
load_dotenv()

def main():
    if len(sys.argv) < 2:
        print("Usage: python delete_document.py <doc_id>")
        print("\nExample:")
        print("  python delete_document.py 04809eba-...")
        print("\nTo get doc_id, run: python raptor_qa_incremental.py")
        print("Then type 'list' to see all documents with their IDs")
        print("\nNOTE: You must be logged in first!")
        print("  Run: python auth.py login <username> <password>")
        sys.exit(1)
    
    doc_id = sys.argv[1]
    
    # Load user from session
    auth = AuthManager()
    user_id = auth.load_session()
    
    if not user_id:
        print(" Not logged in! Please login first:")
        print("  python auth.py login <username> <password>")
        print("\nOr register a new account:")
        print("  python auth.py register <username> <password>")
        sys.exit(1)
    
    # Confirm deletion
    print(f"\n️  About to delete document: {doc_id}")
    print("This will:")
    print("  - Mark all chunks as deleted")
    print("  - Remove affected leaf nodes")
    print("  - Rebuild parent summaries if needed")
    
    confirm = input("\nType 'yes' to confirm deletion: ")
    if confirm.lower() != 'yes':
        print(" Deletion cancelled")
        sys.exit(0)
    
    # Initialize builder
    print("\n Initializing builder...")
    builder = IncrementalRAPTORBuilder(
        model_name=os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
        max_length=100,
        reduction_dimension=10,
        threshold=0.1,
        max_tree_depth=3
    )
    
    # Delete document
    print(f"\n️  Deleting document {doc_id}...")
    builder.delete_document_incremental(doc_id, user_id)
    
    print("\n Document deleted successfully!")
    print("\nTree has been updated. You can verify by running:")
    print(f"  python raptor_qa_incremental.py {user_id}")
    print("  Then type 'list' to see remaining documents")

if __name__ == "__main__":
    main()
