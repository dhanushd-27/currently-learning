from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time
import requests

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
print("\n\n🚦: START")
message_history.append({"role": "user", "content": USER_INPUT})

def get_weather_status(city_name):
  response = requests.get(f"https://wttr.in/{city_name}?format=%C+%t")
  return response

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

  step = json_data.get("steps", "")
  if step == "PLAN":
    print(f"🧠: {json_data.get('content', '').strip()}")
    message_history.append({"role": "assistant", "content": content})
    time.sleep(0.5)
  if step == "FETCH":
    city_name = json_data.get('city_name', '').strip()
    if city_name:
      weather_status = get_weather_status(city_name)
      if weather_status and weather_status.ok:
        temp_content = weather_status.text
      else:
        temp_content = "Unable to fetch weather status at this time."
      observe_json = {
        "steps": "OBSERVE",
        "content": f"The weather status of {city_name.title()} is {temp_content}"
      }
      observe_content = json.dumps(observe_json)
      message_history.append({"role": "assistant", "content": observe_content})
      print(f"👀: {observe_json['content']}")
    time.sleep(0.5)
  if step == "OUTPUT":
    print(f"🤖: {json_data.get('content', '').strip()}")
    break

print("\n\n")