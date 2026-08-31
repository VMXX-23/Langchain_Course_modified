import asyncio
import re
from langchain_core.tools import tool
import os
from mcp import ClientSession, StdioServerParameters #Single tool client connection
from mcp.client.stdio import stdio_client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools #Loads mcp tools from the mcp-tools package
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.messages import HumanMessage as Human
from dotenv import load_dotenv
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent #For HITL, interrupt, and other advanced features
from tool_selector import select_tools #To select tools based on user input keywords/ prevent rate limiting by only loading necessary tools
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
load_dotenv()
print("Loaded environment variables:")
#Debug: Print all environment variables to verify they are loaded correctly
'''for key in os.environ:
    print(f"{key}={os.environ[key]}")
'''
#CONSTANTS
MATH_SERVER_PATH = r"D:\LangchainCourseUDEMY\mcpdoc\servers\math_server.py"
WEATHER_SERVER_URL = "http://127.0.0.1:8000/sse"

# MCP Multi-Client Server Configuration
MCP_SERVER_CONFIG = {
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": [MATH_SERVER_PATH],
    },
    "weather": {
        "transport": "sse",
        "url": WEATHER_SERVER_URL,
    },
}

#Human input tool :HITL
@tool()
def ask_user(query: str) -> str:
    """Use this tool whenever you need clarification, missing inputs, confirmation,
    or additional information from the user before proceeding."""
    # Interrupt pauses execution and receives the human response directly as human_response
    human_response = interrupt(query)
    return str(human_response)

#Function to track token usage
def print_token_usage(messages):
    """Extracts and sums token usage from AIMessages returned in the state."""
    prompt_tokens = 0
    completion_tokens = 0
    
    for msg in messages:
        # Check if message has usage_metadata attached
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            prompt_tokens += msg.usage_metadata.get("input_tokens", 0)
            completion_tokens += msg.usage_metadata.get("output_tokens", 0)
            
    total_tokens = prompt_tokens + completion_tokens
    if total_tokens > 0:
        print(f"📊 [Token Usage] Input: {prompt_tokens} | Output: {completion_tokens} | Total: {total_tokens}")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2
)

#Run MCP servers
"""
server_params = StdioServerParameters(
    command="python",
    args=["D:\\LangchainCourseUDEMY\\mcpdoc\\servers\\math_server.py"],
)
"""

async def main():
    print("Hello from mcp-crash-course!")
    #async with stdio_client(server_params) as (reader, writer): #Single Client run setup
    #    async with ClientSession(read_stream = reader,write_stream =  writer) as session:
            #Initialize the connection
            # Mutli client connect using the server config constant
    #async with MultiServerMCPClient(MCP_SERVER_CONFIG) as session:
     #   await session.initialize()
        
    session = MultiServerMCPClient(MCP_SERVER_CONFIG)
    
    #Get the tools
    #tools = await load_mcp_tools(session)
    tools = await session.get_tools()
    print(f"Loaded {len(tools)} tools from the MCP server."
            )
    #Combine MCP tools with local human input tool
    all_tools = [ask_user] + tools
    # MemorySaver checkpointer is REQUIRED to freeze and resume execution state
    checkpointer = MemorySaver()
    
    # Configuration thread ID allows LangGraph to maintain context across interrupts
    config = {"configurable": {"thread_id": "session-1"}}
    while True:
        user_input = input("Enter your query (or type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the client.")
            return
        
        #Dynamic tool selection based on user input keywords to prevent rate limiting and unnecessary tool loading
        selected_tools = select_tools(user_input, all_tools)
        
        #Create and run the agent:Legacy create_agent() method is used here. For advanced features like HITL, interrupt, and other advanced features, use create_react_agent() instead.
        agent = create_agent(llm, selected_tools, checkpointer=checkpointer)
        #agent = create_react_agent(llm, all_tools, checkpointer=checkpointer)
        
        # Send the user input to the agent for processing
        payload = {"messages": [Human(content=user_input)]}
        
        while True:
            MAX_RETRIES = 3
            agent_response = None
            
            for attempt in range(MAX_RETRIES):
                try:
                    agent_response = await agent.ainvoke(payload, config=config)
                    break  # Exit the retry loop if successful
                except ChatGoogleGenerativeAIError as e:
                    error_message = str(e)
                    if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                        match = re.search(r"retry in (\d+\.?\d*)s", error_message, re.IGNORECASE)
                        wait_time = float(match.group(1)) + 1.0 if match else 7.0
                        print(f"\n⏳ [Rate Limit 429] Gemini quota reached. Waiting {wait_time:.1f}s (Attempt {attempt + 1}/{MAX_RETRIES})...")
                        await asyncio.sleep(wait_time)
                    
                    elif ( "503" in error_message or "UNAVAILABLE" in error_message or "high demand" in error_message):
                        wait_time = 5.0 * (2**attempt)  # Exponential backoff: 5s, 10s, 20s...
                        print(
                            f"\n⏳ [Server Overload 503] Gemini is experiencing high demand. "
                            f"Retrying in {wait_time:.1f}s (Attempt {attempt + 1}/{MAX_RETRIES})..."
                        )
                        await asyncio.sleep(wait_time)
                    
                    else:
                        raise e  # Re-raise non-429 errors
                    
            if agent_response is None:
                print("\n❌ Exceeded maximum retries due to rate limits. Please try again in a minute.")
                break
            # Check if the execution hit a human-in-the-loop interrupt
            state = await agent.aget_state(config)
            #state = agent.get_state(config)
            
            if state.next:  # Graph is paused on interrupt
                # Get the query text emitted by the ask_user tool's interrupt()
                question_for_human = state.tasks[0].interrupts[0].value
                
                # Print the LLM's question and capture live human input from terminal
                print(f"\n Your Agent Needs Input >>: {question_for_human}")
                human_reply = input("👤 Your Response: ").strip()
                #user_input = human_reply  # Update user_input for the
                # Prepare Command payload to resume graph execution with human response
                payload = Command(resume=human_reply)
            
            else:
                # Execution complete! Print final answer
                print("\n Final Answer:")
                #print(agent_response["messages"][-1].content)
                last_message = agent_response["messages"][-1]
                    
                if isinstance(last_message.content, list):
                    clean_text = "".join(part["text"] for part in last_message.content if "text" in part)
                else:
                    clean_text = last_message.content

                print(clean_text)
                # Print usage stats for the entire turn
                print_token_usage(agent_response["messages"])
                
                break
            
    
    """ 
    #Legacy test code for agent response without user input loop
        agent_response = await agent.ainvoke({"messages": [ Human(content="What is 2^2 + 2?")]})
        #print(f"Resulting Agent response: {agent_response}")
        #print(agent_response["messages"][-1].content)"""

                
if __name__ == "__main__":
    asyncio.run(main())
