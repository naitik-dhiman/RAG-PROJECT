# using chrome Db and qdrant  
from langchain_community.vectorstores  import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings


# a library to create your own documents 
from langchain_core.documents import Document

docs =[
    Document(page_content="Python is used for Artificial intelligence" , metadata ={"source" : "AI_book"}),
    Document(page_content="Pandas  is used for Data Analysis " , metadata ={"source" : "Pandas_book"}),
    Document(page_content="Numpy  is used for performing mathematical functions " , metadata ={"source" : "Numpy_book"}),
]



# creating embeddings 
# 1. Load the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Convert a sentence into an embedding
# text = "RAG combines retrieval with language generation."

vector_store  = Chroma.from_documents(
   documents= docs ,
   embedding= embedding_model,
    

)
 

# 3. Inspect the result
print("Embedding dimensions:", len(vector_store))
print("First 10 values:", vector_store[:10])
