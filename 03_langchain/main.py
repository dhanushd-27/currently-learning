# Create a vector store for a PDF file

from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core import documents
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path = Path("./file.pdf")

loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
  model="models/gemini-embedding-001",
  google_api_key=os.getenv("GEMINI_API_KEY")
)

vector_store = QdrantVectorStore.from_documents(
  documents=chunks,
  embedding=embeddings,
  url="http://localhost:6333",
  collection_name="my_collection"
)

print("Indexing of documents is done")
