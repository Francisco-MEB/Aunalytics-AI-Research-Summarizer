from langchain_text_splitters import RecursiveCharacterTextSplitter

def text_to_chunks(text: str, chunk_size=1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.create_documents([text])
    chunks = []
    for i, d in enumerate(docs):
        chunks.append({
            "id": f"chunk-{i}",
            "text": d.page_content,
            "metadata": {"chunk_index": i}
        })
    return chunks
