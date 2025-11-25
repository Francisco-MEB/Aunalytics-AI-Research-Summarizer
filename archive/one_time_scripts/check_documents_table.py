"""Check if documents table is still used"""
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

conn = psycopg2.connect(os.getenv('SUPABASE_DB_URL'))
cur = conn.cursor()

# Check if documents table exists
cur.execute("""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_name = 'documents'
    );
""")
exists = cur.fetchone()[0]
print(f"Documents table exists: {exists}")

if exists:
    # Check if it has any data
    cur.execute("SELECT COUNT(*) FROM documents WHERE user_id = '5b11bef4-7ea1-4bf9-aac1-7f22f7c73705'")
    count = cur.fetchone()[0]
    print(f"Rows in documents table: {count}")
    
    # Check if anything references it
    cur.execute("""
        SELECT
            tc.table_name, 
            kcu.column_name,
            ccu.table_name AS foreign_table_name
        FROM information_schema.table_constraints AS tc 
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY' 
            AND ccu.table_name = 'documents';
    """)
    refs = cur.fetchall()
    if refs:
        print("\nTables referencing documents:")
        for ref in refs:
            print(f"  {ref[0]}.{ref[1]} -> {ref[2]}")
    else:
        print("\n No tables reference documents - safe to drop!")

conn.close()
