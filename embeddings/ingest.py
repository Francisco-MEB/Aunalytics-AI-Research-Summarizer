import argparse, json, os, sys, uuid
from typing import List, Dict
#for the progress bars
from tqdm import tqdm
#langchain text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
#sentence transformers, uses huggingface models
from sentence_transformers import SentenceTransformer

def read_text(path:str) -> str:
    # Read text file with utf-8 encoding and ignore errors
    # If the data becomes more critical, consider using chardet to detect encoding
    #if the file is large, consider reading in chunks or using memory-mapped files
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()
    
    
# chunk_size and chunk_overlap are character counts (not tokens).
# Token-aware splitting (e.g., by a tokenizer for the target embedding model) can be more precise to control model input sizes and cost.
# this function currently uses default lanngchain text splitter which is character based.
# If you want token-aware splitting, consider using tiktoken or similar libraries.
# For production, consider splitting by tokens using the target model's tokenizer to ensure embeddings are within the model's maximum input length (e.g., 512/1024/2048 tokens).
#i also want to vary chunk sizes based on content type (e.g., code vs prose), or technical vs non-technical text.
def chunk_text(text:str, chunk_size: int , chunk_overlap: int) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        #if chunk overlap >= chunk_size:
        #    raise ValueError("chunk_overlap must be less than chunk_size")
        # I want to keep paragraphs together, so split on double newlines first, then single newlines, then spaces, then characters
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.create_documents([text], metadatas=[{}])
    return [{"text": d.page_content, "metadata": d.metadata} for d in docs]

#takes the chunks (list of dicts with "text" key) and returns list of embeddings (list of floats)

#Optimization ideas

#Move model loading outside and pass a model instance in contexts where multiple calls happen.
#Add device selection and detection (auto-detect GPU).
#Add retry/backoff for model downloads (transient network failures).
#If memory is tight, stream embeddings in smaller batches and flush to disk intermittently rather than producing a single vectors list.

def embed_chunks(chunks: List[Dict], model_name: str, batch_size: int = 64,) -> List[List[float]]:
    model = SentenceTransformer(model_name)
    # chunks are expected to be dicts with a "text" key (see chunk_text)
    texts = [c["text"] for c in chunks]
    # encode returns a numpy array when convert_to_numpy=True
    vectors = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    # Convert numpy arrays to Python lists for JSON serialization
    return [v.tolist() for v in vectors]

#Edge cases and improvements

#Atomic writes: currently it writes directly to out_path — on interruption you may produce a partial file. Consider writing to a temp file and then renaming to atomic commit.
##Concurrency: if multiple processes write to the same file, you'll get corrupted output — use unique output files or locks.

def write_jsonl(out_path: str, records: List[Dict]) -> None:
    # If out_path is just a filename in the current directory, dirname may be empty.
    dir_name = os.path.dirname(out_path) or "."
    os.makedirs(dir_name, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main() -> int:
    #parse command line arguments
    #Argparse: you used both default and required=True for --model. If you set a default, remove required=True; 
    # if you want required, remove default. Currently argparse ignores required if default is provided, but it's cleaner to choose one.
#Metadata: you add {"source": normalized input path, "chunk_index": i} and an id. Good minimal provenance. 
# Consider adding char offsets (start/end) so the chunk can be mapped back into the original file exactly.
#Embedding and memory: embed_chunks returns all vectors at once and you then create a records list with all content before writing. 
# For very many chunks, this doubles memory utilization (vectors + records). To reduce peak memory, consider streaming: encode in batches and write out each batch before encoding next (no need to store all vectors/records).
#Progress bars: you show tqdm around writing records. You also set show_progress_bar=True in model.encode which 
# shows SentenceTransformer's own progress. You may end up with overlapping progress output but it's fine.
    p = argparse.ArgumentParser(description = "Ingest .txt- -> chunks -> embeddings -> JSONL")
    # required args: input file, output file, model name
    p.add_argument("--in", dest="inp", required=True, help= "Input .txt file (plain text)")
    p.add_argument("--out", dest="out", required=True, help= "Output .jsonl path")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help= "Sentence transformer model name")
    # optional args: chunk size, chunk overlap, batch size
    p.add_argument("--chunk", dest="chunk", type=int, default=1000, help= "Approx chars per chunk")
    p.add_argument("--overlap", dest="overlap", type=int, default=100, help= "Overlap between chunks (chars))")
    p.add_argument("--batch", dest="batch", type=int, default=64, help= "Embedding batch size")
    args = p.parse_args()


    # validate args, to make sure it makes sense
    if not os.path.isfile(args.inp):
        print(f"Error: Input file {args.inp} does not exist.", file=sys.stderr)
        return 2
    #reads in the text file
    text = read_text(args.inp)

    #makes sure its not some text file with just spaces or newlines
    if not text.strip():
        print(f"Error: Input file {args.inp} is empty or contains only whitespace.", file=sys.stderr)
        return 3
    

    #creates the chunks from the text file and tells the user what its doing, and if it doesnt make any chunks, it will error out
    print(f"Chunking '{args.inp}' into ~{args.chunk} char chunks with {args.overlap} char overlap...")
    chunks = chunk_text(text, chunk_size=args.chunk, chunk_overlap=args.overlap)
    print(f"Created {len(chunks)} chunks.")
    if not chunks:
        print("Error: No chunks were created from the input text.", file=sys.stderr)
        return 4
    
 
    #add simple metadata now; can enhance this later (page numbers, ect.)
    for i, c in enumerate(chunks):
        #creates a unique id for each chunk
        c["id"] = str(uuid.uuid4())
        #adds the source file path and chunk index to the metadata, we can use this later for reference
        c["metadata"].update({"source": os.path.normpath(args.inp), "chunk_index": i})

    #EMBEDDING STEP
    print(f"Embedding chunks using model '{args.model}' in batches of {args.batch}...")
    vectors = embed_chunks(chunks, model_name=args.model, batch_size=args.batch)

    records =[]
    #creates the final records to write out, combining chunk text, metadata, id, and embedding vector
    #combines the chunks and vectors into a single record for each chunk, zip iterates over two lists in parallel, all with progress bar
    for c,v in tqdm(list(zip(chunks, vectors)), total=len(chunks), desc="Writing JSONL"):
        records.append({
            "id": c["id"],
            "text": c["text"],
            "metadata": c["metadata"],
            "embedding": v
        })
    #writes out the jsonl file
    write_jsonl(args.out, records)
    print(f"Wrote {len(records)} records to '{args.out}'")
    return 0



# just makes sure that main is being executed directly
if __name__ == "__main__":
    sys.exit(main())

