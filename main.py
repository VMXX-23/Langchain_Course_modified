from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from chains import generate_chain, reflect_chain


class GRAPH_MESSAGE(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: GRAPH_MESSAGE):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: GRAPH_MESSAGE):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=GRAPH_MESSAGE)
# builder = MessageGraph()
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.set_entry_point("generate")


def should_continue(state: GRAPH_MESSAGE):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")
graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {"messages": [HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)]}
    response = graph.invoke(inputs)
    print(response)
