from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


SYSTEM_PROMPT= """
  You are an assistant who only answers AI questions. You say sorry in case the question is not related to AI.
"""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Explain to me how AI works"
        },
        {
          "role": "user",
          "content": "What is the capital of France?"
        }
    ]
)

print(response.choices[0].message.content)