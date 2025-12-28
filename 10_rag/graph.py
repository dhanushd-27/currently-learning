import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

# -------------------------
# LLM
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# -------------------------
# Embeddings (same as ingest)
# -------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# -------------------------
# Qdrant connection
# -------------------------
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = os.getenv("QDRANT_PORT", "6333")
collection_name = "resume-collection"

client = QdrantClient(
    host=qdrant_host,
    port=int(qdrant_port),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

# This turns the vector store into a retriever that will return the top 5 most similar documents
# based on the embedding similarity for a given query.
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# -------------------------
# Graph State
# -------------------------
class State(dict):
    query: str
    docs: list
    answer: str

# -------------------------
# Nodes
# -------------------------
def retrieve(state: State):
    docs = retriever.invoke(state["query"])
    return {"docs": docs}

def generate(state: State):
    context = "\n\n".join(doc.page_content for doc in state["docs"])

    prompt = f"""
Answer the question using ONLY the resume content below.
If the answer is not present, reply exactly with:
"Not mentioned in resume."

Resume:
{context}

Question:
{state['query']}
"""

    response = llm.invoke(prompt)
    return {"answer": response.content}

# -------------------------
# LangGraph Definition
# -------------------------
graph = StateGraph(State)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge("retrieve", "generate")
graph.set_entry_point("retrieve")

app = graph.compile()