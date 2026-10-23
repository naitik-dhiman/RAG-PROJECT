# for interacting AI with a web page we can easily do web loading 
from langchain_community.document_loaders import WebBaseLoader


url = "https://www.apple.com/in/mac/"

data = WebBaseLoader(url)

docs = data.load()

print(docs[0].page_content)

# if multiple page we have to load make a list and traverse on it 

