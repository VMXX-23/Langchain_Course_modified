from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

from nodes import run_agent_reasoning, tool_node
 

load_dotenv()

AGENT_REASON="Agent_Reason"
ACT="Act"
LAST=-1

#Should continue function to determine if the agent should continue reasoning or act
def should_continue(state: MessagesState ) -> str:
    if not state["messages"][LAST].tool_calls:
        return END 
    return ACT


#Adding a state graph message
flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)


flow.add_conditional_edges(AGENT_REASON, should_continue,{END:END, ACT:ACT})
flow.add_edge(ACT, AGENT_REASON)

app=flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="agent_executor_flow.png")



def main():
    print("Hello from agentexecutor!")



if __name__ == "__main__":
    main()
    res = app.invoke({"messages":[HumanMessage(content="What is the wheather in Chennai now and list it till the end of triple of that value?")]})
    print(res["messages"][LAST].content)