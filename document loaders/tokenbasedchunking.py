# in token based chunking we have token in chunks 
# for ex-:  who are you has 3 tokens if the token chunk size is 2 then who & are  becomes 1 chunk 
from langchain_text_splitters import TokenTextSplitter

text_splitter = 