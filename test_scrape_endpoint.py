"""
Simple test to check if the scrape endpoint is working
"""

import requests
import json

API_URL = "http://127.0.0.1:8000"

print("Testing scrape/analyze endpoint...")
print("="*70)

# Test with a simple, reliable website
test_url = "https://www.example.com"
test_user_id = "test-scrape-user"

try:
    response = requests.post(
        f"{API_URL}/scrape/analyze",
        json={"url": test_url, "user_id": test_user_id},
        timeout=60
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ SUCCESS!")
        print(f"Summary length: {len(data.get('summary', ''))}")
        print(f"Papers found: {len(data.get('papers', []))}")
        print(f"Chunks stored: {data.get('chunks_stored', {})}")
    else:
        print("\n❌ ERROR!")
        print(f"Error: {response.json()}")
        
except Exception as e:
    print(f"\n❌ EXCEPTION!")
    print(f"Error: {e}")

print("="*70)
