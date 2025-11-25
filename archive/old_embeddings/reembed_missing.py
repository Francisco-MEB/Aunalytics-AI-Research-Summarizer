"""
Re-embed Level 0 chunks that have missing embeddings and store them back in the database.

Usage: python embeddings/reembed_missing.py --user-id <user_uuid>
"""
import os
import argparse
import json
from typing import List
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values


def get_db_config():
    return {
        'user': os.getenv('user'),
        'password': os.getenv('password'),
        'host': os.getenv('host'),
        'port': int(os.getenv('port', '5432')),
        'dbname': os.getenv('dbname')
    }


def find_missing_level0(db_config, user_id):
    conn = psycopg2.connect(**db_config)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT doc_id, content FROM documents WHERE user_id = %s AND hierarchy_level = 0 AND (embedding IS NULL OR array_length(embedding, 1) = 0)", (user_id,))
            return cur.fetchall()
    finally:
        conn.close()


def store_embeddings(db_config, rows, vectors):
    conn = psycopg2.connect(**db_config)
    try:
        with conn.cursor() as cur:
            data_to_update = []
            for (row, vector) in zip(rows, vectors):
                data_to_update.append((vector.tolist(), row['doc_id']))
            # Build update query
            for vec, doc_id in data_to_update:
                cur.execute("UPDATE documents SET embedding = %s WHERE doc_id = %s", (vec, doc_id))
            conn.commit()
    finally:
        conn.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--user-id', required=True)
    args = p.parse_args()

    db_config = get_db_config()
    rows = find_missing_level0(db_config, args.user_id)
    if not rows:
        print('No missing embeddings found.')
        return
    print(f'Found {len(rows)} rows missing embeddings.')
    texts = [r['content'] for r in rows]
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    vectors = model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    store_embeddings(db_config, rows, vectors)
    print('Embeddings stored for missing rows.')


if __name__ == '__main__':
    main()
