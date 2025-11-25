"""
Backfill missing file_hash values in documents (for Level 0) using the local filesystem

Usage: python tools/backfill_file_hashes.py --user-id <user_uuid>

This script will iterate over Level 0 documents that lack metadata.file_hash and attempt to compute it
based on the file path in metadata->>source_file. If the file exists locally, it will compute SHA256 and update the DB.
"""
import argparse
import os
import json
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor

def compute_file_hash(path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--user-id', required=True)
    args = p.parse_args()

    db_config = {
        'user': os.getenv('user'),
        'password': os.getenv('password'),
        'host': os.getenv('host'),
        'port': int(os.getenv('port', '5432')),
        'dbname': os.getenv('dbname')
    }

    conn = psycopg2.connect(**db_config)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT doc_id, metadata->>'source_file' as source_file, metadata FROM documents WHERE user_id = %s AND hierarchy_level = 0 AND (metadata->>'file_hash' IS NULL OR metadata->>'file_hash' = '')", (args.user_id,))
            rows = cur.fetchall()
            print(f"Found {len(rows)} Level 0 rows missing file_hash")
            updated = 0
            for row in rows:
                src = row['source_file']
                if not src:
                    continue
                # If path is absolute or relative, try to find it
                if os.path.exists(src):
                    file_hash = compute_file_hash(src)
                    try:
                        metadata = row['metadata'] or {}
                        if isinstance(metadata, str):
                            metadata = json.loads(metadata)
                        metadata['file_hash'] = file_hash
                        cur.execute("UPDATE documents SET metadata = %s WHERE doc_id = %s", (json.dumps(metadata), row['doc_id']))
                        updated += 1
                    except Exception as e:
                        print(f"Error updating {row['doc_id']}: {e}")
                else:
                    print(f"File path not found locally: {src}")
            conn.commit()
            print(f"Updated {updated} rows")
    finally:
        conn.close()

if __name__ == '__main__':
    main()
