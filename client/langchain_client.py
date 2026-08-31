#Connect to multiple servers
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

async def main():
    print("Starting multi-server client...")
    
if __name__ == "__main__":
    load_dotenv()
    import asyncio
    asyncio.run(main())