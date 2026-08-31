# 🤖 Autonomous Agentic Workflow with Hybrid Multi-Server MCP & LangGraph

An end-to-end, production-ready implementation of an autonomous agentic system built on top of **LangGraph**, **LangChain**, and the **Model Context Protocol (MCP)** powered by **Gemini 2.5 Flash**.

This system demonstrates hybrid multi-server protocol orchestration (**STDIO** for local compute + **SSE** for network microservices), multi-turn state preservation, dynamic keyword-based tool pruning to manage context/token consumption, and resilient API rate-limit handling.

---

## 📸 Execution Showcase

Below is the multi-server system running in real-time. The top window shows the **FastMCP Weather SSE Server** streaming over HTTP (`http://127.0.0.1:8000/sse`), while the bottom window shows the **LangGraph Client** dynamically filtering tools and orchestrating execution across both servers:

<img width="1270" height="650" alt="image" src="https://github.com/user-attachments/assets/e9d4f531-bfb1-474b-a4e5-185882441262" />

<img width="1496" height="530" alt="image" src="https://github.com/user-attachments/assets/4d69b37d-fb0b-4eae-ad4e-65ea4f99ecd9" />

> **Execution Breakdown:**
> 1. **Server Initialization:** The FastMCP Weather SSE Server starts up on `http://127.0.0.1:8000/sse`.
> 2. **Client Startup & Tool Discovery:** The client connects to `MultiServerMCPClient`, loading tools from both the local **STDIO Math Server** and the remote **SSE Weather Server**.
> 3. **Dynamic Tool Selection:** The query *"what is the weather in Los Angeles and multiply it by a factorial that is a factor of 4"* automatically filters 21 tools down to the required 4 (`ask_user`, `get_weather`, `factorial`, `multiply`).
> 4. **Stateful Execution:** The agent fetches live temperature data over SSE (21.2°C), evaluates $4! = 24$, and uses LangGraph's `MemorySaver` to persist context across multi-turn user follow-ups.

---

## 🌟 Architecture & Key Features

* **Hybrid Multi-Server MCP Client**: Orchestrates local **STDIO** child-process transports (`math_server.py`) for high-speed local arithmetic alongside **SSE (Server-Sent Events)** network transports (`weather_server.py` at `http://127.0.0.1:8000/sse`) via `MultiServerMCPClient`.
* **LangGraph State Engine**: Built using modern `create_agent` factories with `MemorySaver` checkpointers to maintain thread context (`session-1`) across multi-turn conversational follow-ups.
* **Human-in-the-Loop (HITL)**: Custom `@tool` `ask_user` using LangGraph's `interrupt()` and `Command(resume=...)` to request human inputs mid-execution cleanly.
* **Dynamic Keyword Tool Selector**: Prunes tool definitions per user turn (`select_tools`) based on query keywords to prevent prompt context bloat and minimize API token overhead.
* **Resilient Multi-Tier Error Recovery**: 
  * **429 Rate Limits**: Parses exact delay values from error strings via regex (`retry in X.Xs`) with automatic wait pauses.
  * **503 Server Overloads**: Handles server busy states with exponential backoff ($5\text{s} \times 2^{\text{attempt}}$).
* **Token Usage Telemetry**: `print_token_usage` inspects `usage_metadata` on messages to output real-time `input_tokens` and `output_tokens` per turn.

---

## 🏗️ System Architecture

```text
               +----------------------------------+
               |        Terminal Client UI        |
               +----------------------------------+
                                |
                   (Query Input / HITL Resume)
                                v
               +----------------------------------+
               |   select_tools() Keyword Filter  |
               +----------------------------------+
                                |
                        (Filtered Tools)
                                v
               +----------------------------------+
               |      LangGraph Agent Engine      |
               |     (ChatGoogleGenerativeAI)     |
               +----------------------------------+
                    /                        \
                   /                          \
       (HTTP / SSE Transport)              (STDIO Subprocess)
                 v                              v
    +------------------------+      +-----------------------+
    |   weather_server.py    |      |    math_server.py     |
    | (Open-Meteo REST API)  |      |  (FastMCP Math Engine)|
    +------------------------+      +-----------------------+
``` 
## 💻 Multi-Server Configuration Setup

The client configures local STDIO processes and remote SSE HTTP streams side-by-side using MultiServerMCPClient:
```
Python
MCP_SERVER_CONFIG = {
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": [MATH_SERVER_PATH],
    },
    "weather": {
        "transport": "sse",
        "url": "[http://127.0.0.1:8000/sse](http://127.0.0.1:8000/sse)",
    },
}

# Initialize multi-server client and fetch tools
session = MultiServerMCPClient(MCP_SERVER_CONFIG)
tools = await session.get_tools()
all_tools = [ask_user] + tools
```

## 📁 Repository Structure
├── client/
│   ├── main.py              # Main interactive client, HITL loop & backoff retry engine
│   └── tool_selector.py     # Keyword-based tool pruning module
-> langchain_client.py: Client/ trigger: asyncio.run(main())
├── servers/
│   ├── weather_server.py    # FastMCP server running over SSE (Port 8000)
│   └── math_server.py       # FastMCP server running over STDIO
├── .env.example             # Environment variables template
├── pyproject.toml           # Project dependencies & virtual environment configuration
└── README.md                # Documentation

## 🚀 Quick Start
1. Prerequisites & Installation

  * Python 3.10+

   * Google Gemini API Key (Obtain from Google AI Studio)

Clone the repository and install dependencies using uv or pip:

```Bash
git clone [https://github.com/your-username/mcp-agentic-langgraph.git](https://github.com/your-username/mcp-agentic-langgraph.git)
cd mcp-agentic-langgraph
```

# Using uv (Recommended)
uv sync

2. **Environment Setup**
Create a .env file in the project root:

```Code snippet
GEMINI_API_KEY=your_gemini_api_key_here
LANGCHAIN_API_KEY="ls-..."
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT="mcp.."
```

3. **Launching the System**

First, start the SSE Weather Server in a background terminal:

```Bash
python servers/weather_server.py```
Then run the main LangGraph Client (which auto-spawns the STDIO Math Server):

```Bash
# Target active virtual environment to avoid path warnings
uv run --active client/main.py
```

---

## 💡 Example Terminal Session
<img width="1496" height="530" alt="image" src="https://github.com/user-attachments/assets/4d69b37d-fb0b-4eae-ad4e-65ea4f99ecd9" />
<img width="1299" height="441" alt="image" src="https://github.com/user-attachments/assets/7a1f7e4f-c1a3-4a2f-9a2d-731069dff271" />

---

## 💡 System Design Notes & Critical Value Additions

### 1. Enterprise Value Streams & Operational Architecture
* **Token Cost & Latency Reduction**: By dynamically filtering tool schemas prior to agent graph invocation via `select_tools()`, the client trims baseline system prompt payload by **~60-80%** per turn. In high-volume enterprise deployments, this directly minimizes token consumption and slashes request-response latency.
* **Resilient Multi-Tier Fault Tolerance**: The inner execution loop catches provider exceptions and applies targeted recovery policies:
  * **429 (Resource Exhausted)**: Dynamically extracts provider wait times via regex (`retry in X.Xs`) to prevent aggressive polling and quota lockouts.
  * **503 (Server Overload)**: Implements jittered exponential backoff ($5\text{s} \times 2^{\text{attempt}}$) to handle transient infrastructure spikes seamlessly.
* **Deterministic Auditing & Observability**: Structured `print_token_usage` metrics attached to state transitions provide fine-grained telemetry into prompt vs. completion overhead, serving as a foundation for enterprise FinOps tracking and LangSmith tracing.

---

### 2. Multi-Turn State Preservation & Memory Mechanics
When users supply short conversational follow-ups (such as `"4"` or `"yes"`), keyword matching defaults to a standard core tool set (`['ask_user', 'add', 'subtract', 'multiply', 'get_weather']`). 

Because LangGraph checkpointers (`MemorySaver`) persist execution state across graph invocations, intermediate calculations (such as $4! = 24$) remain available in the thread context window without re-invoking tools.

> **Production Scaling Pathway:** For complex multi-turn sessions, replace keyword fallback with a **Semantic Vector Tool Retriever** (e.g., using `FAISS` or `Chroma`) that evaluates entire conversation history to maintain tool relevance across extended multi-turn tasks.

---

### 3. Additional High-Impact Value Streams

#### A. Multi-Tenant Enterprise MCP Gateway
* **Protocol Decoupling**: Mixing **STDIO** (local isolated execution) and **SSE** (remote microservices) allows organizations to decouple compute heavy tools (e.g., localized Python data processing) from distributed API infrastructure (e.g., weather services, ERP integrations).
* **Role-Based Tool Access (RBAC)**: The `select_tools()` pipeline can be extended to filter tools based on user JWT claims or IAM roles, ensuring end-users only discover and execute MCP tools authorized for their role.

#### B. 🦾 Physical AI, Edge Computing & Autonomous Hardware Extensibility
Because the Model Context Protocol (MCP) strictly separates tool declaration from LLM reasoning, this architecture extends directly into **Physical AI**, **Robotics**, and **Edge IoT Systems**.

* **Aeromodelling & Drone Telemetry Server (`drone_server.py`)**:
Model Interface with flight controllers over `MAVLink` / `pymavlink` via an SSE microservice endpoint to expose real-time drone telemetry:
  ```python
  @mcp.tool()
  async def get_flight_telemetry(drone_id: str) -> str:
      """Returns real-time GPS coordinates, altitude, battery percentage, and airspeed."""
      ...

  @mcp.tool()
  async def execute_waypoint_mission(waypoints: List[Dict[str, float]]) -> str:
      """Dispatches autonomous flight coordinates to drone flight controller."""
      ...
---
## 📜 License

This project is licensed under the MIT License - see the license file for details. 

*Disclaimer: This repository is an educational project created for learning and portfolio purposes based on the LangChain & MCP course curriculum.*
