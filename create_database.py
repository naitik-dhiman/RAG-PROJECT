# load pdf 
# split into chunks 
# create embeddings 
# store into chroma DB 
# one tim doing it 
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader



load_dotenv()


pdf_path = Path(__file__).parent / "document loaders" / "DLbook.pdf"
reader = PdfReader(pdf_path)
docs = [
    Document(
        page_content=page.extract_text() or "",
        metadata={"source": "document loaders/DLbook.pdf", "page": page_number},
    )
    for page_number, page in enumerate(reader.pages)
]



embedding_model = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"

)




# creating a function to  create chunks 
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000 ,
    chunk_overlap = 200 ,

)


chunks = splitter.split_documents(docs)


vector_store = Chroma.from_documents(
    documents= chunks ,
    embedding= embedding_model,
    persist_directory= "Chroma DB",
)







 
