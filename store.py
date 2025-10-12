import psycopg2

conn= psycopg2.connect("postgresql://postgres:[DATACSE2025]@db.vrnkkwhqymobivddukqc.supabase.co:5432/postgres")
cur= conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS documents(
        id SERIAL PRIMARY KEY,
        content TEXT,git 
        metadata JSONB,
        embedding VECTOR(384)
);
""")

conn.commit()
cur.close()
conn.close()

