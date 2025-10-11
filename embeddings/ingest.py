import argparse, json, os, sys, uuid
from typing import List, Dict
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def read_text(path:str) -> str:
    # Read text file with utf-8 encoding and ignore errors
    # If the data becomes more critical, consider using chardet to detect encoding
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()
# chunk_size and chunk_overlap are character counts (not tokens).
# Token-aware splitting (e.g., by a tokenizer for the target embedding model) can be more precise to control model input sizes and cost.
# this function currently uses default lanngchain text splitter which is character based.
# If you want token-aware splitting, consider using tiktoken or similar libraries.
# For production, consider splitting by tokens using the target model's tokenizer to ensure embeddings are within the model's maximum input length (e.g., 512/1024/2048 tokens).
def chunk_text(text:str, chunk_size: int , chunk_overlap: int) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # I want to keep paragraphs together, so split on double newlines first, then single newlines, then spaces, then characters
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.create_documents([text], metadatas=[{}])
    return [{"text": d.page_content, "metadata": d.metadata} for d in docs]



def embed_chunks(chunks: List[Dict], model_name: str, batch_size: int = 64,) -> List[List[float]]:
    model = SentenceTransformer(model_name)
    texts = [c["texts"] for c in chunks]
    vectors = model.encode(texts, batch_size=batch_size, convert_to_numpy =True, show_progress_bar=True)
    return [v.toList() for v in vectors]


def write_json1(out_path: str, records: List[Dict]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main() -> int:
    p = argparse.ArgumentParser(description = "Ingest .txt- -> chunks -> embeddings -> JSONL")
    p.add_argument("--in", dest="inp", required=True, help= "Input .txt file (plain text)")
    p.add_argument("--out", dest="out", required=True, help= "Output .jsonl path")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", required=True, help= "Sentence transformer model name")
    p.add_argument("--chunk", dest="chunk", type=int, default=1000, help= "Approx chars per chunk")
    p.add_argument("--overlap", dest="overlap", type=int, default=100, help= "Overlap between chunks (chars))")
    p.add_argument("--batch", dest="batch", type=int, default=64, help= "Embedding batch size")
    args = p.parse_args()

    if not os.path.isfile(args.inp):
        print(f"Error: Input file {args.inp} does not exist.", file=sys.stderr)
        return 2
    
    text = read_text(args.inp)
    if not text.strip():
        print(f"Error: Input file {args.inp} is empty or contains only whitespace.", file=sys.stderr)
        return 3
    
    print(f"Chunking '{args.inp}' into ~{args.chunk} char chunks with {args.overlap} char overlap...")
    chunks = chunk_text(text, chunk_size=args.chunk, chunk_overlap=args.overlap)
    print(f"Created {len(chunks)} chunks.")
    if not chunks:
        print("Error: No chunks were created from the input text.", file=sys.stderr)
        return 4
    
 
    #add simple metadata now; you can enhance this later (page numbers, ect.)
    for i, c in enumerate(chunks):
        c["id"] = str(uuid.uuid4())
        c["metadata"].update({"source": os.path.normpath(args.inp), "chunk_index": i})

    print(f"Embedding chunks using model '{args.model}' in batches of {args.batch}...")
    vectors = embed_chunks(chunks, model_name=args.model, batch_size=args.batch)

    records =[]
    for c,v in tqdm(list(zip(chunks, vectors)), total=len(chunks), desc="Writing JSONL"):
        records.append({
            "id": c["id"],
            "text": c["text"],
            "metadata": c["metadata"],
            "embedding": v
        })

    write_json1(args.out, records)
    print(f"Wrote {len(records)} records to '{args.out}'")
    return 0

if __name__ == "__main__":
    sys.exit(main())
    
