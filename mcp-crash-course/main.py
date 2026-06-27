import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools #Loads mcp tools from the mcp-tools package
from langchain.agents import create_agent
from langchain.messages import HumanMessage as Human
from dotenv import load_dotenv
load_dotenv()
print("Loaded environment variables:")
for key in os.environ:
    print(f"{key}={os.environ[key]}")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

#How to run MCP servers
server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["D:\LangchainCourseUDEMY\mcpdoc\servers\math_server.py"],
)
async def main():
    print("Hello from mcp-crash-course!")
    async with stdio_client(server_params) as (reader, writer):
        async with ClientSession(read_stream = reader,write_stream =  writer) as session:
            #Initialize the connection
            await session.initialize()
            
            #Get the tools
            tools = await load_mcp_tools(session)
            print(f"Loaded {len(tools)} tools from the MCP server."
                  )
            #Create and run the agent
            agent = create_agent(llm, tools)
            agent_response = await agent.ainvoke({"messages": [ Human(content="What is 2^2 + 2?")]})
            print(f"Resulting Agent response: {agent_response}")
            print(agent_response["messages"][-1].content)
            
if __name__ == "__main__":
    asyncio.run(main())
