import psycopg2

conn= psycopg2.connect("your_supabase_connection_string")
cur= conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS documents(
        id SEIRAL PRIMARY KEY,
        content TEXT,git 
        metadata JSONB,
        embedding VECTOR(384)
);
""")

conn.commit()
cur.close()
conn.close()

