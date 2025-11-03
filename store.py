import psycopg2

DATABASE_URL = "hi"

conn= psycopg2.connect("DATABASE_URL")
cur= conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS documents(
        id SERIAL PRIMARY KEY,
        content TEXT,
        metadata JSONB,
        embedding VECTOR(384)
);
""")

conn.commit()
cur.close()
conn.close()

