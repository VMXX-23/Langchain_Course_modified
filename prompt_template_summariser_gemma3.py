import sys

from httpx import Client
sys.stdout.reconfigure(encoding='utf-8')
#USING GEMMA 3 INSTEAD OF GEMINI 
import os
from google import genai
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langsmith import Client as LangsmithClient
#Ensures the tracing is enabled and langchain runs properly even if env variables arent set properly
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainCourse"

# Load environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found! Check your .env file or path.")

# Initialize the Gemini client
client = genai.Client(api_key=api_key)

def main():
    print("✅ Ollama client initialized successfully!")
    print("Hello from langchain-course!\n")
    
    langsmith_client = LangsmithClient()
    print(langsmith_client.list_projects())

    # Person information to summarize
    information = '''
Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2021; as of October 2025, Forbes estimates his net worth at US$500 billion. He has founded multiple companies, including SpaceX, Tesla, Neuralink, The Boring Company, and X.com (PayPal). Musk has also been involved in politics and AI initiatives.
'''

    # Prompt template with proper variable substitution
    summary_template = """
Given the following information about a person:

{information}

Please create:
1. A short summary
2. Two interesting facts about the person
3. A creative title for the person
4. A fun fictional interview question and answer with the person
"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )
    
    llm = ChatOllama(
    model="gemma3:270m",
    temperature=0,
    # other params...
)
    # Format the prompt with the actual information for gemini client
    #prompt_text = summary_prompt_template.format(information=information)
    
    chain = summary_prompt_template | llm
    # Call the free-tier Gemini model
    '''response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt_text
    )
    '''
    #ollama response
    response = chain.invoke({ "information": information })

    print("Ollama Response:\n")
    print(response.content)

if __name__ == "__main__":
    main()
##ERRROR TO DO WITH THE INSTALLATION OF OLLAMA AND GEMMA