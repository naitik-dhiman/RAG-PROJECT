from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

data = PyPDFLoader("document loaders/notes.pdf")

docs = data.load()

splitter = TokenTextSplitter(
    chunk_size =  100, # 100 token at once 
    chunk_overlap = 10 ,
)

chunks  = splitter.split_documents(docs)

# 2 len pages 
# print(len(docs))
# print(docs[0].page_content) 


print(len(chunks))
print(chunks[0].page_content)

# meaning captures good here 
