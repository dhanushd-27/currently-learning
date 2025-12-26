from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
import os
import operator

# Load .env from root directory
load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# State of the workflow
class TweetState(TypedDict):
  topic: str
  tweet: str
  feedback: str
  evaluation: Literal["approved", "needs_improvement"]
  iterations: int
  max_iterations: int

  tweet_history: Annotated[list[str], operator.add]
  feedback_history: Annotated[list[str], operator.add]

# LLMs for the workflow

generative_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
optimizer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)

# structured output for the evaluator_llm
class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(..., description="Final evaluation result.")
    feedback: str = Field(..., description="feedback for the tweet.")

# Create a Pydantic output parser
output_parser = PydanticOutputParser(pydantic_object=TweetEvaluation)
graph = StateGraph(TweetState)

# Methods used in the workflow
def generate_tweet(state: TweetState) -> TweetState:
  print("\n" + "="*70)
  print("📝 STEP 1: GENERATING TWEET")
  print("="*70)
  print(f"Topic: {state['topic']}")
  print(f"Iteration: {state['iterations'] + 1}/{state['max_iterations']}")
  print("-"*70)
  
  # prompt
  messages = [
    SystemMessage(content="You are a funny and clever Twitter/ X influencer."),
    HumanMessage(content=f"""Generate a tweet about {state['topic']}
    
    Rules:
    - The tweet should be 280 characters or less.
    - The tweet should be in the language of the topic.
    - The tweet should be unique and not repetitive.
    - The tweet should be interesting and engaging.
    - The tweet should be relevant to the topic.
    """)
  ]
  # send generator_llm
  response = generative_llm.invoke(messages)
  print(f"🤖 Generated Tweet:\n{response.content}")
  print("-"*70)

  # return response
  return {"tweet": response.content}

def evaluate_tweet(state: TweetState) -> TweetState:
  print("\n" + "="*70)
  print("🔍 STEP 2: EVALUATING TWEET")
  print("="*70)
  print(f"Tweet to evaluate:\n{state['tweet']}")
  print("-"*70)
  
  # prompt
  messages = [
    SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
    HumanMessage(content=f"""
      Evaluate the following tweet:

      Tweet: "{state['tweet']}"

      Use the criteria below to evaluate the tweet:

      1. Originality – Is this fresh, or have you seen it a hundred times before?  
      2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
      3. Punchiness – Is it short, sharp, and scroll-stopping?  
      4. Virality Potential – Would people retweet or share it?  
      5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

      Auto-reject if:
      - It's written in question-answer format (e.g., "Why did..." or "What happens when...")
      - It exceeds 280 characters
      - It reads like a traditional setup-punchline joke
      - Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., "Masterpieces of the auntie-uncle universe" or vague summaries)

      ### Respond ONLY in structured format:
      - evaluation: "approved" or "needs_improvement"  
      - feedback: One paragraph explaining the strengths and weaknesses 
      
      {output_parser.get_format_instructions()}
      """)
      ]

  # evaluate the generated tweet and return the evaluation with feedback
  response = evaluator_llm.invoke(messages)
  parsed_response = output_parser.parse(response.content)
  
  print(f"📊 Evaluation Result: {parsed_response.evaluation.upper()}")
  print(f"💬 Feedback:\n{parsed_response.feedback}")
  print("-"*70)

  # return response
  return {
    'evaluation': parsed_response.evaluation,
    'feedback': parsed_response.feedback
  }

def optimize_tweet(state: TweetState) -> TweetState:
  print("\n" + "="*70)
  print("⚡ STEP 3: OPTIMIZING TWEET")
  print("="*70)
  print(f"Current iteration: {state['iterations'] + 1}/{state['max_iterations']}")
  print(f"Previous tweet:\n{state['tweet']}")
  print(f"\nFeedback received:\n{state['feedback']}")
  print("-"*70)
  
  # prompt
  messages = [
    SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
    HumanMessage(content=f"""
    Improve the tweet based on this feedback:
    "{state['feedback']}"

    Topic: "{state['topic']}"
    Original Tweet:
    {state['tweet']}

    Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
    """)
  ]

  response = optimizer_llm.invoke(messages).content
  iterations = state['iterations'] + 1
  
  print(f"✨ Optimized Tweet:\n{response}")
  print("-"*70)
  print("🔄 Looping back to evaluation...")

  return {'tweet': response, 'iterations': iterations, 'tweet_history': [response]}

# route evaluator 
def route_evaluator(state: TweetState) -> Literal["optimize", "end"]:
  print("\n" + "="*70)
  print("🚦 ROUTING DECISION")
  print("="*70)
  
  if state['evaluation'] == "approved":
    print("✅ Tweet APPROVED! Moving to final output...")
    print("="*70 + "\n")
    return "end"
  elif state['iterations'] >= state['max_iterations']:
    print(f"⏱️  Max iterations ({state['max_iterations']}) reached. Moving to final output...")
    print("="*70 + "\n")
    return "end"
  else:
    print(f"🔄 Tweet needs improvement. Iteration {state['iterations']}/{state['max_iterations']}")
    print("   → Routing to OPTIMIZE step...")
    print("="*70)
    return "optimize"

# Nodes for the workflow
graph.add_node("generate", generate_tweet)
graph.add_node("evaluate", evaluate_tweet)
graph.add_node("optimize", optimize_tweet)

# Edges for the workflow
graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")

# add conditional edge from evaluate (removed unconditional edge - can't have both)
graph.add_conditional_edges(
  "evaluate",
  route_evaluator,
  {
    "optimize": "optimize",
    "end": END
  }
)

# add loop edge from optimize back to evaluate (removed optimize->END edge - can't have multiple unconditional edges)
graph.add_edge("optimize", "evaluate")

# Add the graph to the registry
model = graph.compile()

# Run the workflow
print("\n" + "="*70)
print("🚀 STARTING TWEET GENERATION WORKFLOW")
print("="*70)
print("Topic: AI")
print("Max Iterations: 3")
print("="*70 + "\n")

result = model.invoke({"topic": "AI", 'iterations': 0,"max_iterations": 3})

print("\n" + "="*70)
print("🎉 WORKFLOW COMPLETED")
print("="*70)
print("📌 FINAL APPROVED TWEET:")
print("-"*70)
print(result.get("tweet"))
print("-"*70)
print(f"📊 Total Iterations: {result.get('iterations')}")
print(f"✅ Final Status: {result.get('evaluation')}")
print("="*70 + "\n")