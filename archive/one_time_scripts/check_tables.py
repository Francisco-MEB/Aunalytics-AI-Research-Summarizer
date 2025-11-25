"""Quick script to check database tables and counts"""
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    user=os.getenv('user'),
    password=os.getenv('password'),
    host=os.getenv('host'),
    port=os.getenv('port'),
    dbname=os.getenv('dbname')
)
cur = conn.cursor()

# List all tables
cur.execute("""
    SELECT table_name FROM information_schema.tables 
    WHERE table_schema = 'public' 
    ORDER BY table_name
""")
print('TABLES IN DATABASE:')
for row in cur.fetchall():
    print(f'  - {row[0]}')

# Check counts in each relevant table
print('\nROW COUNTS:')
tables = ['documents', 'chunks', 'tree_nodes', 'chunk_to_leaf', 'users']
for t in tables:
    try:
        cur.execute(f'SELECT COUNT(*) FROM {t}')
        count = cur.fetchone()[0]
        print(f'  {t}: {count} rows')
    except Exception as e:
        print(f'  {t}: does not exist')
        conn.rollback()

# Check for user's data specifically
print('\nYOUR DATA (from session):')
try:
    import json
    with open('.session', 'r') as f:
        session = json.load(f)
    user_id = session.get('user_id')
    
    cur.execute("SELECT COUNT(*) FROM documents WHERE user_id = %s", (user_id,))
    print(f'  documents: {cur.fetchone()[0]} rows')
    
    cur.execute("SELECT COUNT(*) FROM chunks WHERE user_id = %s AND deleted = FALSE", (user_id,))
    print(f'  chunks (active): {cur.fetchone()[0]} rows')
    
    cur.execute("SELECT COUNT(*) FROM chunks WHERE user_id = %s AND deleted = TRUE", (user_id,))
    print(f'  chunks (deleted): {cur.fetchone()[0]} rows')
    
    cur.execute("SELECT COUNT(*) FROM tree_nodes WHERE user_id = %s", (user_id,))
    print(f'  tree_nodes: {cur.fetchone()[0]} rows')
    
except Exception as e:
    print(f'  Error: {e}')

conn.close()
