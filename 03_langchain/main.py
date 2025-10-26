from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

pdf_path = Path("./file.pdf")

loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

print(chunks)
