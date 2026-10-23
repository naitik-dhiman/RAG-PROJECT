from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document loaders/notes.pdf")

docs = data.load()

# 2 len pages 
print(len(docs))

