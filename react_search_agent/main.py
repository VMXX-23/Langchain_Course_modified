from dotenv import load_dotenv
load_dotenv()
from langgraph.prebuilt import create_react_agent #Updated for new syntax
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient
from typing import  List
from pydantic import BaseModel, Field
#Field class -> include meta data, Base Model -> Base class to incl
tavily = TavilyClient()

class Source (BaseModel):
    """Schema for source used by agent"""
    url:str = Field(description="The url of the source")

class AgentResponse (BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to a string")
    sources:List[Source] = Field(default_factory=list, description="The List of sources used to generate an answer!")




@tool
def search(query: str) -> str :
    '''Tool that searches the web for user inputs'''
    print(f"Searching:  {query}")
    query = f"What are the job open in India for the role:  {query}"
    search_res = tavily.search(query)
    return search_res

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=1024,
    timeout=120,
    max_retries=2,
    top_p=0.9,
    top_k=20,
    #streaming=True,
  #model_kwargs={
   #     "thinking_config": {
    #        "thinking_budget": 256,
     #       "include_thoughts": True
      #  }
    #} """
)

tools = [search]
'''search the web'''
web_search_agent = create_react_agent(model = model, tools=tools , response_format=AgentResponse,
                                      prompt="""You are a job search assistant.
For every job-related query, ALWAYS call the search tool.Never answer from your own knowledge.
Summarize the search results and include the source URLs."""
                                      ) #React web search agent

#Function to get only final message
def get_text(result):
    final_message = result["messages"][-1]

    if isinstance(final_message.content, list):
        return "\n".join(
            part["text"]
            for part in final_message.content
            if part.get("type") == "text"
        )

    return final_message.content

def main():
    print("Welcome to Web search tool /n")
    user = input("Enter preferred Job: ")
    result = web_search_agent.invoke({"messages":[HumanMessage(content=user)]})

    print(get_text(result)) # Print the final message

if __name__ == "__main__":
    main()
