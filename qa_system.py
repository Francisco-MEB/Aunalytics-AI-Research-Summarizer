#made a file for the qa system 
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from sentence_transformers import SentenceTransformer
#library to connect PostgresSQL database
import psycopg, os
from langchain_core.documents import Document


# keep track of the input question, retrieved context, and generated answer
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Load the same embedding model as ingestion
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Take the user's input question
query = input("Enter your question: ")

# Embed it
query_vector = model.encode(query, normalize_embeddings=True).tolist()

# Connect to PostgreSQL (assuming that pgvector already exists - store in .env file?) 
DB_URL = os.getenv("DATABASE_URL")

with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
    # Search for the most similar chunks to the query embedding
    cur.execute(
        """
        SELECT doc_id, content, 1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT 4;
        """,
        (query_vector, query_vector)
    )
    rows = cur.fetchall()

# Wrap retrieved rows into LangChain Documents
retrieved_docs = [
    Document(
        page_content=row[1],
        metadata={"doc_id": row[0], "score": row[2]}
    )
    for row in rows
]

# Create the State object
state = {
    "question": query,
    "context": retrieved_docs,
    "answer": ""   # will be filled in by the LLM generation step
}


