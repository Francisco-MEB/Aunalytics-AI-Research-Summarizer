"""
Simple connection test - debugging Supabase connection issue
"""
import os
from dotenv import load_dotenv
import psycopg

load_dotenv()
db_url = os.getenv("DATABASE_URL")

print("=" * 60)
print("CONNECTION STRING TEST")
print("=" * 60)

# Parse the connection string to show what we're trying
print(f"\n Connection Details:")
print(f"Full URL (redacted): {db_url[:30]}...{db_url[-30:]}")
print(f"URL length: {len(db_url)} characters")

# Check for common issues
issues = []
if " " in db_url:
    issues.append("️  URL contains spaces")
if db_url.startswith(" ") or db_url.endswith(" "):
    issues.append("️  URL has leading/trailing whitespace")
if "\n" in db_url or "\r" in db_url:
    issues.append("️  URL contains newline characters")
if not db_url.startswith("postgresql://"):
    issues.append("️  URL doesn't start with 'postgresql://'")

if issues:
    print("\n Potential Issues Found:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n URL format looks clean")

# Try to connect with detailed error info
print("\n Attempting connection...")
try:
    # Add connection timeout
    conn = psycopg.connect(
        db_url,
        connect_timeout=10,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5
    )
    print(" CONNECTION SUCCESSFUL!")
    
    # Try a simple query
    with conn.cursor() as cur:
        cur.execute("SELECT 1 as test;")
        result = cur.fetchone()
        print(f" Query test: {result}")
    
    conn.close()
    print("\n Everything works!")
    
except psycopg.OperationalError as e:
    print(f" Connection failed!")
    print(f"\n Error details:")
    print(f"   Type: {type(e).__name__}")
    print(f"   Message: {str(e)}")
    
    error_str = str(e).lower()
    print(f"\n Diagnosis:")
    
    if "password authentication failed" in error_str:
        print("    Wrong password")
        print("   → Reset password in Supabase and update .env")
    elif "could not connect" in error_str or "connection refused" in error_str:
        print("    Can't reach server")
        print("   → Check if project is paused or internet connection")
    elif "closed the connection" in error_str or "terminated abnormally" in error_str:
        print("    Server rejected connection")
        print("   → Likely wrong password OR project paused")
        print("   → Try SQL Editor in Supabase to verify project is active")
    elif "timeout" in error_str:
        print("    Connection timeout")
        print("   → Network/firewall issue")
    else:
        print("    Unknown error - see message above")
    
except Exception as e:
    print(f" Unexpected error: {type(e).__name__}")
    print(f"   {str(e)}")

print("\n" + "=" * 60)
