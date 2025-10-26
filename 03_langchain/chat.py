# Chat with a vector store

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

embeddings = GoogleGenerativeAIEmbeddings(
  model="models/gemini-embedding-001",
  google_api_key=os.getenv("GEMINI_API_KEY")
)

vector_store = QdrantVectorStore.from_existing_collection(
  url="http://localhost:6333",
  collection_name="my_collection",
  embedding=embeddings
)

user_query = input("👉: ")

docs = vector_store.similarity_search(user_query) 

context = "\n\n\n".join([f"Page Content: {doc.page_content}\nPage Number: {doc.metadata['page']}" for doc in docs])

SYSTEM_PROMPT = f"""
  You are a helpful AI Assistant who answers user query based on the avialable context retrieved from a PDF file along with page_contents and page number.

  Context: {context}
"""

openai_response = client.chat.completions.create(
  model="gemini-2.5-flash",
  messages=[
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_query}
  ]
)

print(openai_response.choices[0].message.content)