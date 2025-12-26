from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class AgentState(TypedDict):
  result: SystemMessage
  input: HumanMessage

graph = StateGraph(AgentState)

graph.add_node()