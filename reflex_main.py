from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage

# List of messages logged
from langgraph.graph import END, START, StateGraph, MessagesState


from reflex_chains import revisor, first_responder
from relfex_tool_executor import execute_tools

MAX_ITERATIONS = 2


# Receives messages to draft response
def draft_node(state: MessagesState):
    """Draft Initial response"""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


"""Revise the answer based on result"""


def revise_node(state: MessagesState):
    response = first_responder.invoke({"messages": state["messages"]})


"""Decides to continue or exit based on iteration count"""


def event_loop(state: MessagesState):
    count_tool_visits = sum(isinstance(itm, ToolMessage) for itm in state["messages"])

    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute tools"


builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute tools", execute_tools)
builder.add_node("revise", revise_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
graph = builder.compile()

print(graph.get_graph().draw_mermaid())

res = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
            }
        ]
    }
)

last_message = res["messages"][-1]
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["Answer"])
print(res)
