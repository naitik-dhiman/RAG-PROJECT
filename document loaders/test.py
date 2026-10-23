from langchain_community.document_loaders import TextLoader

# 1. Initialize the loader
data = TextLoader("document loaders/notes.txt")

# 2. Call .load() to get the documents
docs = data.load()

# 3. Print the actual documents/data
# it is the list that contains document  and every document has two things meta data and page content 
# for a  text file only one document 
print(docs[0].page_content)
print(len(docs))