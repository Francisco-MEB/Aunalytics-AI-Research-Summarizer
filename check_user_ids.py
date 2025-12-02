"""
Check what user_ids exist in the database
"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))

print("="*70)
print("CHECKING USER_IDS IN DATABASE")
print("="*70)

# Get unique user_ids and count chunks for each
response = supabase.table("documents").select("user_id").execute()

user_counts = {}
for row in response.data:
    user_id = row.get('user_id')
    if user_id:
        user_counts[user_id] = user_counts.get(user_id, 0) + 1

print(f"\nTotal chunks in database: {len(response.data)}")
print(f"Number of unique user_ids: {len(user_counts)}")

print("\nUser IDs and their chunk counts:")
for user_id, count in sorted(user_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {user_id}: {count} chunks")

print("\n" + "="*70)
print("To fix the issue:")
print("1. Open your browser console (F12)")
print("2. Type: localStorage.getItem('user_id')")
print("3. Copy that UUID")
print("4. Use it when uploading files or scraping websites")
print("="*70)
