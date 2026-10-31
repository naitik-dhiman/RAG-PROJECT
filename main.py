
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

# Initialize chat model
llm = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq",
    api_key=api_key
)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load existing vector database
vector_store = Chroma(
    persist_directory="Chroma DB",
    embedding_function=embedding_model
)

# Create retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a helpful AI assistant.
    Use ONLY the provided context to answer the question.
    If the answer is not present in the context,
    say: I could not find the answer in the document.
    """),

    ("human", """
    Context: {context}
    Question: {question}
    """)
])

print("RAG SYSTEM CREATED")
print("Press 0 to exit")

while True:

    query = input("You: ")

    if query == "0":
        break

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    # Extract context from retrieved documents
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Create final prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Generate response
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")
