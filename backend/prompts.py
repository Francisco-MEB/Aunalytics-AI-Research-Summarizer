from langchain.prompts import ChatPromptTemplate

SYSTEM_INSTRUCTION = (
    "You are an expert AI assistant specializing in summarizing academic research. "
    "Use ONLY the provided context. Do not fabricate information."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTION),
        (
            "human",
            "CONTEXT:\n{context}\n\n"
            "QUESTION:\n{question}\n\n"
            "Answer ONLY using the context above."
        ),
    ]
)
