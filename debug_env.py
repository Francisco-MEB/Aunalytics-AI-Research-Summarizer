"""Debug .env file reading"""
import os
from dotenv import load_dotenv

print("Testing .env file reading...")
print("=" * 60)

load_dotenv()

db_url = os.getenv("DATABASE_URL")
print(f"DATABASE_URL exists: {db_url is not None}")
print(f"DATABASE_URL length: {len(db_url) if db_url else 0}")

if db_url:
    # Check for hidden characters
    print(f"\nFirst 50 chars: {repr(db_url[:50])}")
    print(f"Last 50 chars: {repr(db_url[-50:])}")
    
    # Check password section specifically
    if "@" in db_url and ":" in db_url:
        parts = db_url.split("@")[0]  # Get user:password part
        password_part = parts.split(":")[-1]  # Get just password
        print(f"\nPassword from URL: '{password_part}'")
        print(f"Password length: {len(password_part)}")
        print(f"Password (repr): {repr(password_part)}")
        
        # Compare with DATABASE_PASSWORD if it exists
        db_pass = os.getenv("DATABASE_PASSWORD")
        if db_pass:
            db_pass = db_pass.strip()  # Remove any whitespace
            print(f"\nDATABASE_PASSWORD env var: '{db_pass}'")
            print(f"Passwords match: {password_part == db_pass}")
