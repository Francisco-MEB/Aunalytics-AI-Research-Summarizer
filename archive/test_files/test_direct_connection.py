import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Fetch variables
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

print("=" * 60)
print("DIRECT CONNECTION TEST")
print("=" * 60)
print(f"\nConnection parameters:")
print(f"  User: {USER}")
print(f"  Password: {'*' * len(PASSWORD) if PASSWORD else 'NOT SET'}")
print(f"  Host: {HOST}")
print(f"  Port: {PORT}")
print(f"  Database: {DBNAME}")
print()

# Connect to the database
try:
    print(" Attempting connection...")
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    print(" Connection successful!")
    
    # Create a cursor to execute SQL queries
    cursor = connection.cursor()
    
    # Example query
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    print(f" Current Time: {result[0]}")
    
    # Test pgvector
    cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
    vector_result = cursor.fetchone()
    if vector_result:
        print(f" pgvector extension is enabled")
    else:
        print("️  pgvector extension NOT found")

    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("\n All tests passed! Connection closed.")

except Exception as e:
    print(f" Failed to connect: {e}")
    print("\nTroubleshooting:")
    print("1. Check .env file has all variables set")
    print("2. Verify password is correct")
    print("3. Wait a few more minutes if you just reset password")

print("=" * 60)
