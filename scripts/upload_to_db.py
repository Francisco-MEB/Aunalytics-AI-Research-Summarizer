#!/usr/bin/env python3
"""
Upload JSONL embeddings to PostgreSQL with pgvector
Reads a .jsonl file (output from ingest.py) and uploads to database
"""
import argparse
import json
import os
import sys
import psycopg2
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_table_if_not_exists(conn):
    """Create the documents table with pgvector extension if it doesn't exist"""
    with conn.cursor() as cur:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(384),  -- all-MiniLM-L6-v2 produces 384-dim vectors
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create index for vector similarity search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx 
            ON documents USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        conn.commit()
        print("✅ Table 'documents' created/verified with pgvector extension")

def load_jsonl(file_path):
    """Load records from JSONL file"""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                records.append(json.loads(line))  # Load JSON object
    return records

def upload_records(conn, records):
    """Upload records to database"""
    with conn.cursor() as cur:
        inserted = 0
        updated = 0

        for record in tqdm(records, desc="Uploading to database"):  # tqdm progress bar
            doc_id = record['id']
            content = record['text']
            embedding = record['embedding']
            metadata = json.dumps(record['metadata']) # dump metadata as JSON string
            
            # Upsert: insert or update if exists
            cur.execute("""
                INSERT INTO documents (doc_id, content, embedding, metadata)
                VALUES (%s, %s, %s::vector, %s::jsonb)
                ON CONFLICT (doc_id) 
                DO UPDATE SET 
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata;
            """, (doc_id, content, embedding, metadata))
            
            if cur.rowcount == 1:
                inserted += 1
            else:
                updated += 1
        
        conn.commit()
        print(f"✅ Uploaded {inserted} new documents, updated {updated} existing documents")

def main():
    parser = argparse.ArgumentParser(description="Upload JSONL embeddings to PostgreSQL/pgvector")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file path")
    args = parser.parse_args()
    
    # Get connection parameters from .env
    user = os.getenv("user")
    password = os.getenv("password")
    host = os.getenv("host")
    port = os.getenv("port")
    dbname = os.getenv("dbname")
    
    if not all([user, password, host, port, dbname]):
        print("❌ Error: Missing connection parameters in .env file")
        print("Required: user, password, host, port, dbname")
        return 1
    
    # Validate input file
    if not os.path.isfile(args.input):
        print(f"❌ Error: Input file '{args.input}' does not exist")
        return 2
    
    try:
        # Connect to database using individual parameters
        print(f"🔌 Connecting to database...")
        conn = psycopg2.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            dbname=dbname
        )
        print("✅ Connected successfully!")
        
        # Create table if needed
        create_table_if_not_exists(conn)
        
        # Load records from JSONL
        print(f"📖 Loading records from '{args.input}'...")
        records = load_jsonl(args.input)
        print(f"✅ Loaded {len(records)} records")
        
        if not records:
            print("⚠️  No records to upload")
            return 0
        
        # Upload to database
        upload_records(conn, records)
        
        # Show stats
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents;")
            total = cur.fetchone()[0]
            print(f"\n📊 Database now contains {total} total documents")
        
        conn.close()
        print("✅ Upload complete!")
        return 0
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        return 3
    except Exception as e:
        print(f"❌ Error: {e}")
        return 4

if __name__ == "__main__":
    sys.exit(main())
