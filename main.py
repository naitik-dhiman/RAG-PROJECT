import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print("API key loaded:", bool(api_key))

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

# Load the text file
data = TextLoader("document loaders/notes.txt")
docs = data.load()

# Create the prompt template
template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI that summarizes the text."),
    ("human", "{text}")
])

# Initialize the Groq model
model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq",
    api_key=api_key
)

# Insert document content into the prompt
prompt = template.format_messages(
    text=docs[0].page_content
)

# Get the response
response = model.invoke(prompt)

print(response.content)
