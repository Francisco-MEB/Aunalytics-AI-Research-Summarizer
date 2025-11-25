"""
Quick demo of RAPTOR QA
"""
from raptor_qa import RaptorQASystem

user_id = "2a6ba9c7-c3fc-4772-a2af-4ac8acaaa0c4"
qa = RaptorQASystem(user_id)

# Sample questions
questions = [
    "What is machine learning?",
    "Give me a summary of all papers",
    "Compare supervised and unsupervised learning"
]

print("=" * 70)
print("RAPTOR QA DEMO")
print("=" * 70)

for q in questions:
    print(f"\nQ: {q}")
    result = qa.ask(q, verbose=False)
    print(f"A: {result['answer']}\n")
    print("-" * 70)
