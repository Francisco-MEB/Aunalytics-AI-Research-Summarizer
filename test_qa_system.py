#!/usr/bin/env python3
"""
Test script for QA System - Tests individual components
"""
import os
import sys
from typing import List
from langchain_core.documents import Document

# Add the current directory to Python path so we can import qa_system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all imports work"""
    print("🧪 Testing imports...")
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI imported successfully")
        
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers imported successfully")
        
        from langchain_core.documents import Document
        print("✅ LangChain Document imported successfully")
        
        import psycopg2
        print("✅ psycopg2 imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_embedding_model():
    """Test if the embedding model works"""
    print("\n🧪 Testing embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Test encoding
        test_text = "What is machine learning?"
        embedding = model.encode(test_text, normalize_embeddings=True)
        
        print(f"✅ Embedding model works! Vector dimension: {len(embedding)}")
        print(f"✅ Sample embedding values: {embedding[:5]}...")
        return True
    except Exception as e:
        print(f"❌ Embedding model error: {e}")
        return False

def test_gemini_api_key():
    """Test if Gemini API key is set"""
    print("\n🧪 Testing Gemini API key...")
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("✅ GEMINI_API_KEY environment variable is set")
        print(f"✅ Key starts with: {api_key[:10]}...")
        return True
    else:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("💡 Please set your Gemini API key:")
        print("   1. Go to https://aistudio.google.com/app/apikey")
        print("   2. Create an API key")
        print("   3. Set environment variable: set GEMINI_API_KEY=your_key_here")
        return False

def test_gemini_connection():
    """Test if we can connect to Gemini"""
    print("\n🧪 Testing Gemini connection...")
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ Cannot test Gemini - API key not set")
            return False
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Simple test prompt
        response = model.generate_content("Say 'Hello, I am working!' in one sentence.")
        
        if response.text:
            print("✅ Gemini connection successful!")
            print(f"✅ Response: {response.text.strip()}")
            return True
        else:
            print("❌ Gemini returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ Gemini connection error: {e}")
        return False

def test_qa_system_init():
    """Test if QA system can initialize (without database)"""
    print("\n🧪 Testing QA System initialization...")
    try:
        # Set a dummy DATABASE_URL for testing
        os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"
        
        from qa_system import QASystem
        qa_system = QASystem()
        
        print("✅ QA System initialized successfully!")
        print(f"✅ Model loaded: {type(qa_system.model).__name__}")
        print(f"✅ Gemini model: {qa_system.gemini_model.model_name}")
        return True
        
    except Exception as e:
        print(f"❌ QA System initialization error: {e}")
        return False

def test_generate_answer_without_db():
    """Test the generate_answer function with mock data"""
    print("\n🧪 Testing answer generation (mock data)...")
    try:
        # Set environment variables
        os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"
        
        if not os.getenv("GEMINI_API_KEY"):
            print("❌ Cannot test answer generation - GEMINI_API_KEY not set")
            return False
        
        from qa_system import QASystem
        qa_system = QASystem()
        
        # Create mock context documents
        mock_context = [
            Document(
                page_content="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
                metadata={"doc_id": "doc1", "score": 0.95}
            ),
            Document(
                page_content="Deep learning is a type of machine learning that uses neural networks with multiple layers to process and learn from large amounts of data.",
                metadata={"doc_id": "doc2", "score": 0.87}
            )
        ]
        
        # Test answer generation
        question = "What is machine learning?"
        answer = qa_system.generate_answer(question, mock_context)
        
        print("✅ Answer generation successful!")
        print(f"✅ Question: {question}")
        print(f"✅ Answer: {answer[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Answer generation error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 QA System Component Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_embedding_model,
        test_gemini_api_key,
        test_gemini_connection,
        test_qa_system_init,
        test_generate_answer_without_db
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print("-" * 30)
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Your QA system components are working!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)