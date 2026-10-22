
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage , SystemMessage , HumanMessage


# Load .env from the GENAI root folder
load_dotenv()

# Check whether the API key is available
api_key = os.getenv("GROQ_API_KEY")
print("API key loaded:", bool(api_key))

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

# Initialize the Groq model
model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq",
    api_key=api_key
)


response = model.invoke("Who is naitik")

print(response.content)