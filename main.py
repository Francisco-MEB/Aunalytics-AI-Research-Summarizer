#!/usr/bin/env python3
"""
RAPTOR Research Summarizer - Main Entry Point
==============================================
A hierarchical document Q&A system using RAPTOR (Recursive Abstractive Processing 
for Tree-Organized Retrieval) with incremental updates.

Usage:
    python main.py qa          # Interactive Q&A
    python main.py add FILE    # Add document
    python main.py stream      # Streaming Q&A
    python main.py cleanup     # Database maintenance
    python main.py --help      # Show all options
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCommands:")
        print("  qa        - Interactive Q&A system")
        print("  stream    - Streaming Q&A (real-time responses)")
        print("  add       - Add a document")
        print("  batch     - Batch process documents")
        print("  cleanup   - Database maintenance")
        print("  auth      - User authentication")
        print()
        return
    
    command = sys.argv[1].lower()
    
    # Remove the command from argv so submodules see correct args
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == 'qa':
        from cli.qa import main as qa_main
        qa_main()
    
    elif command == 'stream':
        from cli.streaming import main as stream_main
        stream_main()
    
    elif command == 'add':
        from cli.add import add_document
        from cli.auth import AuthManager
        
        if len(sys.argv) < 2:
            print("Usage: python main.py add <file_path> [--strategy semantic|fixed|sliding|hierarchical]")
            return
        
        auth = AuthManager()
        user_id = auth.load_session()
        if not user_id:
            print("Please login first: python main.py auth login <username> <password>")
            return
        
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("file_path")
        parser.add_argument("--strategy", default="semantic", 
                           choices=["fixed", "semantic", "sliding", "hierarchical"])
        args = parser.parse_args()
        
        add_document(args.file_path, user_id, args.strategy)
    
    elif command == 'batch':
        # Run batch processor
        exec(open('cli/batch.py').read())
    
    elif command == 'cleanup':
        from cli.cleanup import interactive_cleanup
        interactive_cleanup()
    
    elif command == 'auth':
        from cli.auth import AuthManager
        auth = AuthManager()
        
        if len(sys.argv) < 2:
            print("Usage:")
            print("  python main.py auth login <username> <password>")
            print("  python main.py auth register <username> <password>")
            print("  python main.py auth logout")
            return
        
        action = sys.argv[1]
        if action == 'login' and len(sys.argv) >= 4:
            auth.login(sys.argv[2], sys.argv[3])
        elif action == 'register' and len(sys.argv) >= 4:
            auth.register(sys.argv[2], sys.argv[3])
        elif action == 'logout':
            auth.logout()
        else:
            print("Invalid auth command")
    
    else:
        print(f"Unknown command: {command}")
        print("Run 'python main.py' for help")


if __name__ == "__main__":
    main()
