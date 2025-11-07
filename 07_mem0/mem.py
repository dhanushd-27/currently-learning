# Mem0 is a vector database for storing and retrieving memories.

import os
from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI
import json

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
  api_key=GEMINI_API_KEY,
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

config = {
  "version": "v1.1",
  "embedder": {
    "provider": "gemini",
    "config": { "api_key": GEMINI_API_KEY, "model": "gemini-embedding-001"}

  },
  "llm": {
      "provider": "gemini",
      "config": {
          "model": "gemini-2.0-flash-001",
          "temperature": 0.2,
      }
  },
  "vector_store": {
    "provider": "qdrant",
    "config": {
      "host": "localhost",
      "port": 6333,
      "collection_name": "mem0_gemini_768_v3",
      "embedding_model_dims": 768
    },
  }
}

memory = Memory.from_config(config)

while True:
    user_query = input("👉: ")

    search_memory = memory.search(query=user_query, user_id="Dhanush")

    memories = [
      f"ID: {mem.get('id')}, Memory: {mem.get('memory')}" for mem in search_memory.get("results")
    ]

    SYSTEM_PROMPT = (
        "You are a helpful assistant.\n"
        f"Here are some previous memories:\n{json.dumps(memories)}"
    )

    print("\n\n\n")

    print("--------------------------------")
    print(memories)
    print("--------------------------------")

    print("\n\n\n")
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ]
        )
        ai_response = response.choices[0].message.content
    except Exception as e:
        print("AI response error:", e)
        continue

    print("AI:", ai_response)
    try:
        memory.add(
            user_id="Dhanush",
            messages=[
                {
                    "role": "user",
                    "content": user_query
                },
                {
                    "role": "assistant",
                    "content": ai_response
                }
            ]
        )
        print("Memory added")
    except Exception as e:
        print("Error adding memory:", e)