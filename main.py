

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
from google import genai
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# Load environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found! Check your .env file or path.")

# Initialize the Gemini client
client = genai.Client(api_key=api_key)

def main():
    print("✅ Gemini client initialized successfully!")
    print("Hello from langchain-course!")
    
    
    print("Available models:")
    for m in client.models.list():
        print("-", m.name)

    # Use a valid current model
    
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",  # ✅ valid model
        contents="Write a short poem about learning LangChain."
    )

    print("\nGemini Response:\n", response.text)

if __name__ == "__main__":
    main()
