from langchain.prompts import ChatPromptTemplate

SYSTEM_INSTRUCTION = ChatPromptTemplate(
    ("system", "You are a helpful AI bot")
)