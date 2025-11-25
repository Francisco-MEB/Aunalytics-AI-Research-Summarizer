"""
Test RAPTOR QA System with sample questions
"""
from raptor_qa import RaptorQASystem
import sys


def run_tests(user_id: str):
    """Run test questions against the test papers"""
    qa = RaptorQASystem(user_id)
    
    # Test questions covering different types
    test_questions = [
        # FACTUAL - Should retrieve Level 0 details
        {
            "question": "What are the main algorithms in supervised learning?",
            "expected_type": "factual",
            "expected_level": 0
        },
        {
            "question": "What is the purpose of batch normalization?",
            "expected_type": "factual",
            "expected_level": 0
        },
        
        # SUMMARY - Should retrieve Level 1-2 summaries
        {
            "question": "Give me an overview of the papers",
            "expected_type": "summary",
            "expected_level": 2
        },
        {
            "question": "What are the main topics covered?",
            "expected_type": "summary",
            "expected_level": [1, 2]
        },
        
        # COMPARISON - Should retrieve mixed levels
        {
            "question": "Compare CNNs and Transformers for computer vision",
            "expected_type": "comparison",
            "expected_level": [0, 1]
        },
        
        # ANALYTICAL - Should retrieve all levels
        {
            "question": "How does the attention mechanism improve NLP models?",
            "expected_type": "analytical",
            "expected_level": [0, 1, 2]
        },
    ]
    
    print("=" * 80)
    print("RAPTOR QA SYSTEM TEST")
    print("=" * 80)
    print(f"User ID: {user_id}\n")
    
    results = []
    
    for i, test in enumerate(test_questions, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}/{len(test_questions)}")
        print(f"{'=' * 80}")
        print(f"Question: {test['question']}")
        print(f"Expected Type: {test['expected_type']}")
        print(f"Expected Levels: {test['expected_level']}")
        print("-" * 80)
        
        result = qa.ask(test['question'], verbose=True)
        
        print("\nANSWER:")
        print("-" * 80)
        print(result['answer'])
        
        # Validation
        type_match = result['question_type'] == test['expected_type']
        
        # Check if retrieved levels match expected
        retrieved_levels = set(doc['hierarchy_level'] for doc in result['sources'])
        expected = test['expected_level']
        if isinstance(expected, list):
            level_match = any(lvl in retrieved_levels for lvl in expected)
        else:
            level_match = expected in retrieved_levels
        
        print("\n" + "-" * 80)
        print(f"Classification: {result['question_type']} {'' if type_match else ''}")
        print(f"Levels Retrieved: {sorted(retrieved_levels)} {'' if level_match else ''}")
        print(f"Documents Used: {result['num_used']}/{result['num_retrieved']}")
        
        results.append({
            'test': test,
            'result': result,
            'type_match': type_match,
            'level_match': level_match
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    type_correct = sum(1 for r in results if r['type_match'])
    level_correct = sum(1 for r in results if r['level_match'])
    
    print(f"Classification Accuracy: {type_correct}/{len(results)} ({type_correct/len(results)*100:.1f}%)")
    print(f"Level Routing Accuracy: {level_correct}/{len(results)} ({level_correct/len(results)*100:.1f}%)")
    
    print("\nDetailed Results:")
    for i, r in enumerate(results, 1):
        status = "PASS" if (r['type_match'] and r['level_match']) else "FAIL"
        print(f"{i}. {status}: {r['test']['question'][:50]}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_qa.py <user_id>")
        sys.exit(1)
    
    user_id = sys.argv[1]
    run_tests(user_id)
