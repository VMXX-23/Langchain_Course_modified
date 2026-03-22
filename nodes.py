from dotenv import load_dotenv
import os
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSTEM_MESSAGE = "You are a helpful assistant that can use tools to answer questions."

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    # An LLM call with system role and content from the state messages
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_MESSAGE},
        *state["messages"]
    ])
    
    return {"messages": [response] }
    
tool_node = ToolNode(tools)
