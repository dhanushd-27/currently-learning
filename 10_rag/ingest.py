from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from pypdf import PdfReader
import os
from dotenv import load_dotenv

load_dotenv()

reader = PdfReader("Frontend.pdf")
# This code reads all the pages from the PDF using pypdf's PdfReader,
# extracts the text from each page, and combines everything into a single string,
# with each page's text separated by a newline.
text = "\n".join([page.extract_text() for page in reader.pages])

# Chunk resume
splitter = RecursiveCharacterTextSplitter(
  chunk_size=400,
  chunk_overlap=80
)
chunks = splitter.split_text(text)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
  model="models/gemini-embedding-001",
  google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Qdrant
# Connect to local Qdrant instance
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = os.getenv("QDRANT_PORT", "6333")

client = QdrantClient(host=qdrant_host, port=int(qdrant_port))

# Collection name (equivalent to Pinecone index)
collection_name = "resume-collection"

# Note: We'll let LangChain handle the collection creation with the correct dimensions
# Delete collection if it exists to ensure clean start
try:
    client.delete_collection(collection_name)
    print(f"Deleted existing collection '{collection_name}'")
except Exception:
    print(f"Collection '{collection_name}' doesn't exist yet")

# Use LangChain's Qdrant integration to add documents
vector_store = QdrantVectorStore.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name=collection_name,
    url=f"http://{qdrant_host}:{qdrant_port}",
)

print("✅ Resume indexed successfully")