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
    # Gemini requires the final message in the prompt to be a HumanMessage (user turn).
    # Since state["messages"] ends with the generation_node's AIMessage,
    # append a prompt requesting a critique on that message.
    messages = list(state["messages"])
    messages.append(
        HumanMessage(
            content="Critique the above generated tweet draft harshly. Suggest specific improvements."
        )
    )
    res = reflect_chain.invoke({"messages": messages})
    content = res.content if res.content else str(res)
    critique_content = f"CRITIQUE AND FEEDBACK FOR PREVIOUS TWEET:\n{content}\n\nPlease revise the tweet to address all points above and the STRICT OUTPUT FORMAT."
    return {"messages": [HumanMessage(content=critique_content)]}


builder = StateGraph(state_schema=GRAPH_MESSAGE)
# builder = MessageGraph()
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.set_entry_point("generate")


def should_continue(state: GRAPH_MESSAGE):
    #Debug
    #print("NUMBER OF MESSAGES:", len(state["messages"]))
    #print("MESSAGES:", state["messages"])
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")
graph = builder.compile()
#Mermaid Diagram graph viz
#print(graph.get_graph().draw_mermaid())
#ASCII mermaid viz
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello Tweet Enhancer Here !!")
    inputs = {"messages": [HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)]}
    response = graph.invoke(inputs)
    #print(response)
    #Print only the output result, not the entire chains
    print(response["messages"][-1].content)
