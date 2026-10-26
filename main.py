#importing text file , loading text file 

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print("API key loaded:", bool(api_key))

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

# Load the text file
data = PyPDFLoader("document loaders/DLbook.pdf")
docs = data.load()  




splitter  = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,


)


chunks = splitter.split_documents(docs)
# after this we store chunks embedding in vector Database 
print(chunks[0].page_content)



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


# chunk also have meta data 
# when we are sending data to from pdf to actual LLM there comes a problem of context window because sometime pdf can be longer 

# creating chunks to store in database later and to compare then 