from google import genai
import os
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
try:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='Hello, are you there?'
    )
    print(f"Success with gemini-2.5-flash: {response.text}")
except Exception as e:
    print(f"Failed with gemini-2.5-flash: {e}")
