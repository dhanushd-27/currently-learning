from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT=os.getenv("SYSTEM_PROMPT")
message_history = [
  {"role": "system", "content": SYSTEM_PROMPT},
]

print("\n\n")

USER_INPUT=input("👉: ")
message_history.append({"role": "user", "content": USER_INPUT})

while True:
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        response_format={"type": "json_object"},
        messages=message_history
    )

    content = response.choices[0].message.content
    try:
        json_data = json.loads(content)
    except Exception:
        print(f"Error parsing response to JSON: {content}")
        break

    step = json_data.get("step", "")

    if step == "OUTPUT":
      print(f"🤖: {json_data.get('content', '').strip()}")
      break
    if step == "PLAN":
      print(f"🧠: {json_data.get('content', '').strip()}")
      message_history.append({"role": "assistant", "content": content})
    time.sleep(0.5)

print("\n\n")