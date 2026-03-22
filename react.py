from dotenv import load_dotenv
import os
from langchain_core.tools  import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num:float) -> float:  
    """Returns the triple of a number."""
    return num * 3

tools = [TavilySearch( max_results = 1), triple]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro", temperature=0).bind_tools(tools) #Calling LLM model with tools using bind_tools method



