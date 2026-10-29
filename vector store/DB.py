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

   documents = docs ,

   embedding = embedding_model,

   persist_directory="chroma-db"

)

# vector store is not responsible  , vector store will retrieve and return the similar thing from those embeddings 

result = vector_store.similarity_search("What is used for data analysis " , k = 2  )


for  r in result : 
    print (r.page_content)
    print ("----------------------------")
    print(r.metadata)

# jupyter notebook is prefered here because if run code once then it will create the database and if we re run it it will make the database again 



# retrieving the 

retriever = vector_store.as_retriever()

docs = retriever.invoke()










# to create a directory we have to do it for storing in database

# creating local storage



 



# 3. Inspect the result

