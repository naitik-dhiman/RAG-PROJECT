from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",
    chunk_size = 10,
    chunk_overlap = 1 ,
)
 

# 1. Initialize the loader
data = TextLoader("document loaders/notes1.txt")

# 2. Call .load() to get the documents
docs = data.load()

chunks = splitter.split_documents(docs)


# print(len(chunks))
# print(chunks)   

for i in chunks :
    print(i.page_content)
    print()
    print()
    print()

     



# 3. Print the actual documents/data
# it is the list that contains document  and every document has two things meta data and page content 
# for a  text file only one document 
# print(docs[0].page_content)
# print(len(docs))   


# 10 characters in one chunk 