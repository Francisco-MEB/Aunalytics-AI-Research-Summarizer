#made a file for the qa system 
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from sentence_transformers import SentenceTransformer


# keep track of the input question, retrieved context, and generated answer
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str




