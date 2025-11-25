#!/usr/bin/env python3
"""
End-to-end test of the PDF → QA workflow
Tests the complete pipeline: PDF → embeddings → database → QA system
"""
import os
import sys
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_prerequisites():
    """Check if environment variables are set"""
    print_header("Checking Prerequisites")
    
    issues = []
    
    # Check DATABASE_URL
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        print(f" DATABASE_URL is set")
        # Mask password in output
        masked_url = db_url.split('@')[0].split(':')[0] + ":****@" + db_url.split('@')[1] if '@' in db_url else "****"
        print(f"   {masked_url}")
    else:
        print(" DATABASE_URL is not set")
        issues.append("DATABASE_URL")
    
    # Check GEMINI_API_KEY
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f" GEMINI_API_KEY is set")
        print(f"   Key starts with: {api_key[:10]}...")
    else:
        print(" GEMINI_API_KEY is not set")
        issues.append("GEMINI_API_KEY")
    
    if issues:
        print(f"\n️  Missing environment variables: {', '.join(issues)}")
        print("\n Please create a .env file with:")
        print("   DATABASE_URL=your_database_url")
        print("   GEMINI_API_KEY=your_gemini_api_key")
        return False
    
    return True

def test_pdf_ingestion():
    """Test PDF ingestion (creating embeddings)"""
    print_header("Step 1: PDF Ingestion (Create Embeddings)")
    
    # Check if there's a document to process
    test_doc = "data/science.1203877.docx"  # Use the existing docx file
    if not os.path.exists(test_doc):
        print(f"️  Test document not found: {test_doc}")
        print(" Place a PDF or .docx file in the data/ directory")
        return False
    
    print(f" Processing document: {test_doc}")
    output_file = "data/test_embedded.jsonl"
    
    # Run ingest.py
    cmd = [
        sys.executable,
        "embeddings/ingest.py",
        "--in", test_doc,
        "--out", output_file,
        "--chunk", "800",
        "--overlap", "100"
    ]
    
    print(f" Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(" PDF ingestion successful!")
        print(result.stdout)
        return output_file
    else:
        print(" PDF ingestion failed!")
        print(result.stderr)
        return None

def test_database_upload(jsonl_file):
    """Test uploading embeddings to database"""
    print_header("Step 2: Upload to Database")
    
    if not jsonl_file or not os.path.exists(jsonl_file):
        print(f" JSONL file not found: {jsonl_file}")
        return False
    
    # Run upload_to_db.py
    cmd = [
        sys.executable,
        "upload_to_db.py",
        "--input", jsonl_file
    ]
    
    print(f" Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(" Database upload successful!")
        print(result.stdout)
        return True
    else:
        print(" Database upload failed!")
        print(result.stderr)
        return False

def test_qa_system():
    """Test the QA system with sample questions"""
    print_header("Step 3: Test QA System")
    
    try:
        from qa_system import QASystem
        
        print(" Initializing QA system...")
        qa_system = QASystem()
        print(" QA system initialized!")
        
        # Test questions
        test_questions = [
            "What is this research about?",
            "What are the main findings?",
            "What methods were used?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n Question {i}: {question}")
            result = qa_system.ask_question(question)
            print(f" Answer: {result['answer'][:300]}...")
            print(f" Sources used: {result['num_sources']}")
        
        return True
        
    except Exception as e:
        print(f" QA system test failed: {e}")
        return False

def main():
    """Run the complete test workflow"""
    print_header("PDF → QA System End-to-End Test")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n Prerequisites check failed. Please fix the issues above.")
        return 1
    
    # Step 1: PDF ingestion
    jsonl_file = test_pdf_ingestion()
    if not jsonl_file:
        print("\n Test failed at PDF ingestion step")
        return 1
    
    # Step 2: Database upload
    if not test_database_upload(jsonl_file):
        print("\n Test failed at database upload step")
        return 1
    
    # Step 3: QA system test
    if not test_qa_system():
        print("\n Test failed at QA system step")
        return 1
    
    # Success!
    print_header(" All Tests Passed!")
    print(" PDF ingestion works")
    print(" Database upload works")
    print(" QA system works")
    print("\n You can now use the QA system interactively:")
    print("   python qa_system.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
