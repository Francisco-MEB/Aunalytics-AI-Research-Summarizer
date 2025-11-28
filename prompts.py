from langchain.prompts import ChatPromptTemplate


SYSTEM_INSTRUCTION = (
    [
        ("system", "You are a helpful AI with expertise in the research areas and details  of numerous professors. Your job is to simplify and put information into Layman's terms for the user interaction."),
    ]
)

template = ChatPromptTemplate.from_messages(
    {
        ("system", SYSTEM_INSTRUCTION),
        ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}"),
    }
)