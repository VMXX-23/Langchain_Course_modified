# LangGraph Agent Executor — Custom State & Tool-Calling Agent

> A hands-on implementation of an **agent executor architecture built from scratch using LangGraph**, demonstrating custom state-driven execution, LLM tool calling, conditional routing, Tavily search, and custom Python tools.

---

## 🎯 Purpose

This branch demonstrates how an **agent executor** can be constructed explicitly rather than relying on a high-level abstraction.

The implementation uses:

- **LangGraph** for stateful workflow orchestration
- **Gemini** as the reasoning LLM
- **Tavily Search** as an external web information-retrieval tool
- A **custom Python tool** for deterministic arithmetic computations (`triple`).
- **MessagesState** for maintaining conversation/tool state
- Conditional routing to determine whether the agent should **reason again** or **execute a tool**
- A compiled LangGraph workflow for repeated **Reason → Act → Reason** execution

## 🔄 Execution Loop & Architecture

Unlike linear chains or static RAG pipelines, the Agent Executor allows the LLM to dynamically determine whether execution requires external tools before generating an answer.

### State Machine Lifecycle
The goal is to understand what happens underneath an agent executor:
```text
                  ┌───────────────────────────────────────────────┐
                  │                 MessagesState                 │
                  │   messages: [ HumanMessage("Query...") ]      │
                  └──────────────────────┬────────────────────────┘
                                         │
                                         ▼
                             ┌──────────────────────┐
                             │ AGENT_REASON Node    │
                             │ (llm.invoke)         │
                             └───────────┬──────────┘
                                         │
                                         ▼
            ┌───────────────────────────────────────────────────────────┐
            │ AIMessage(content="", tool_calls=[{name, args, id}])      │
            └────────────────────────────┬──────────────────────────────┘
                                         │
                                 should_continue()
                                         │
                        ┌────────────────┴────────────────┐
                        │                                 │
                   (tool_calls?)                   (No tool_calls)
                        │                                 │
                        ▼                                 ▼
                 ┌─────────────┐                       ┌─────┐
                 │  ACT Node   │                       │ END │
                 │ (ToolNode)  │                       └─────┘
                 └──────┬──────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ ToolMessage(content="Result", tool_call_id="call_xxx")                       │
└──────────────────────────────────────────────┬───────────────────────────────┘
                                               │
                                               └──────► (Appended to State)


```
## What This Branch Demonstrates

This implementation focuses on the mechanics behind an agent executor.

> Instead of simply doing: Prompt → LLM → Answer

the system allows the LLM to decide whether it needs to use a tool.

> The execution loop becomes: Reason → Decide → Act → Observe → Reason → ...
This is the fundamental pattern behind many tool-using agents.

## 📂 Repository Structure
```text
agent-executor/
├── main.py                   # StateGraph definition, compilation, and entry point
├── nodes.py                  # Agent reasoning node logic and ToolNode instance
├── react.py                  # Model initialization, tool definitions, and tool binding
├── agent_executor_flow.png   # Auto-generated Mermaid graph visualization
├── .env                      # API keys and environment variables (git-ignored)
└── README.md                 # Branch documentation
```

### Custom Agent Executor

The executor is implemented using a MessagesState state graph:

Agent Reasoning Node — invokes the LLM with the current message state.
Tool Node — executes requested tools using LangGraph's ToolNode.
Conditional Routing — checks whether the latest message contains tool calls.
Loop — tool results are returned to the reasoning node for further processing.
```text
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.add_node(ACT, tool_node)


flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {END: END, ACT: ACT}
)


flow.add_edge(ACT, AGENT_REASON)
```
# 🛠️ Tools
Tavily Search : Provides the agent with web-search capability for real-time information.

> TavilySearch(max_results=1)

 Custom triple Tool : A custom LangChain tool demonstrating how application-specific functionality can be exposed to the agent.
```text
@tool
def triple(num: float) -> float:
    """Returns the triple of a number."""
    return num * 3
```
The agent can combine multiple tools to answer a single query.

🤖 Gemini + Tool Binding

The Gemini model is configured with the available tools using bind_tools():

```text
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0
).bind_tools(tools)
```
This allows the LLM to decide when a tool should be called based on the user's request.

## 🔬 Deep-Dive: Under-the-Hood Engineering Mechanics

1. **Per-Turn System Instruction vs. State Cleanliness**
   
Passing {"role": "system", "content": SYSTEM_MESSAGE} dynamically in run_agent_reasoning() ensures that system instructions are sent as an explicit System Prompt during model execution without persisting inside MessagesState. LangChain's internal convert_to_messages() coerces {"role": "system", ...} into a proper SystemMessage, which the Google GenAI provider extracts into Gemini's top-level system_instruction parameter.

2. **The Reducer Protocol in MessagesState**

In standard LangGraph, returning {"messages": [response]} from a node does not overwrite graph history. MessagesState utilizes an underlying add_messages reducer function. Returning a list appends new AIMessage or ToolMessage instances. If a returned message shares an existing id, add_messages updates the entry rather than appending.

3. **Strict Tool ID Protocol Matching**
   
When Gemini requests a tool execution, it generates an AIMessage containing a unique tool_calls array with an ID (tool_call_id="call_xxx"). When ToolNode executes, it must return a ToolMessage carrying that exact matching ID. Mismatched IDs cause orphaned tool calls and fatal 400 Invalid Prompt API errors on subsequent model turns.

4. **Tool Selection vs. Execution Boundaries**
   
llm.bind_tools(tools) purely translates Python functions into JSON/Pydantic schemas attached to Gemini's API payload. It provides zero execution capability to the LLM. The actual execution is handled in isolated Python runtime steps managed by ToolNode.

5. **Parallel Tool Call Handling**
    
If a user prompt requires multiple actions (e.g., searching for weather data AND multiplying numbers), Gemini generates multiple tool call requests in a single turn. ToolNode processes all requested tools in parallel or sequence, appending every output ToolMessage before passing control back to AGENT_REASON.

6. **Loop Prevention & Recursion Caps**
    
Because workflow.add_edge(ACT, AGENT_REASON) introduces a cycle, failing tools can cause infinite execution loops. In production, invocations should configure recursion limits:


```Python
app.invoke(
    {"messages": [HumanMessage(content="...")]},
    config={"recursion_limit": 10}
)
```
``
## 🛠️ Setup & Execution
1. Environment Configuration
Create a .env file in the root directory:

```text
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```
2. Installation
```Bash
pip install langchain langgraph langchain-google-genai langchain-tavily python-dotenv
```
3. Run the Application
```bash
python main.py
```
The app executes the workflow, logs message transitions, renders agent_executor_flow.png, and prints the multi-step reasoning steps to the console.

## 📊 Graph Visualization
Upon compilation, the graph exports its layout via Mermaid:

```Python
app.get_graph().draw_mermaid_png(output_file_path="agent_executor_flow.png")
```

## 🎯 Key Concepts Demonstrated
1. LangGraph MessagesState
2. Custom agent executor architecture
3. Conditional graph routing
4. oolNode
5. Gemini tool calling
6. bind_tools()
7. Tavily web search
8. Custom LangChain tools
9. Multi-step agent reasoning
10. Agent → Tool → Agent execution loops

## 📈 Future Experiments

Possible extensions include:

1. Persistent agent memory
2. More custom tools
3. Human-in-the-loop approval
4. Streaming responses
5. Tool error handling
6. More complex state management
7. Multi-agent workflows
