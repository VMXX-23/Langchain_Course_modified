# 🛠️ Paradigms of Tool Execution: ReAct, Function Calling & LangChain Tool Calling

An educational and architectural demonstration comparing three distinct paradigms for building LLM agents with external tool execution: **Raw Prompt-Based ReAct Loop**, **Native Google GenAI SDK Function Calling**, and **LangChain Unified Tool Calling**.

Using a product pricing and discount calculation domain, this branch contrasts how reasoning loops, tool declarations, text parsing mechanics, and conversation message histories are managed across different abstraction tiers.

---

## 📸 Architectural Overview & Paradigm Comparison

```text
               +-------------------------------------------------------+
               |                  User Prompt Query                    |
               +-------------------------------------------------------+
                                           |
         +---------------------------------+---------------------------------+
         |                                 |                                 |
         v                                 v                                 v
+------------------+             +--------------------+            +-------------------+
|   raw_react.py   |             |function_calling.py |            |  tool_calling.py  |
|  (Raw ReAct Loop)|             | (Google GenAI SDK) |            |  (LangChain Abstr)|
+------------------+             +--------------------+            +-------------------+
         |                                 |                                 |
   Prompt String                     Structured Types                 @tool Decorator
   + Regex Parsers                   Function Declarations           + init_chat_model
         |                                 |                                 |
         v                                 v                                 v
  [Manual Text Parse]              [Native JSON Schema]             [Unified Tool Call]
 `Action:` / `Input:`             `part.function_call`              `AIMessage.tool_calls`
```
## 🔍 Deep-Dive: Message Extraction & Parsing Mechanics (raw_react.py)
A primary objective of this repository is demonstrating how message contents and tool arguments are selected and parsed when working without native function calling frameworks.

### 1. raw_react.py — How Output Text & Tool Calls are Extracted
In raw_react.py, the LLM has no concept of structured JSON functions—it returns a raw text string following the ReAct prompt template. Control flow relies on manual string matching and regex parsing:

A. Getting the Text Output
The response object returned by client.models.generate_content() holds candidate outputs. We pull the raw text directly:

```code output = response.text
# Equivalent underlying path: response.candidates[0].content.parts[0].text
```

B. Detecting the Final Answer
Before checking for tool calls, we inspect the text output for the termination phrase specified in the ReAct prompt (Final Answer:):

```Python
final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
if final_answer_match:
    final_answer = final_answer_match.group(1).strip()
    return final_answer  # Loop terminates
```

C. Extracting Tool Name & Arguments via Regex
If no final answer is found, we parse out the tool name from Action: and its JSON arguments from Action Input::

```Python
action_match = re.search(r"Action:\s*(.+)", output)
action_input_match = re.search(r"Action Input:\s*(.+)", output)
tool_name = action_match.group(1).strip()        # e.g. "get_prod_price"
args = json.loads(action_input_match.group(1).strip()) # e.g. {"product_id": "laptop"}
```

D. Appending History ("The Scratchpad")
Because there are no structured message roles (ToolMessage), the execution result ("Observation") is appended directly onto the running prompt string:

```Python
history += output
history += f"\nObservation: {observation}\nThought:"
```

2. Message Extraction in function_calling.py vs. tool_calling.py

| Component | `raw_react.py` (String ReAct) | `function_calling.py` (GenAI SDK) | `tool_calling.py` (LangChain) |
| :--- | :--- | :--- | :--- |
| **Response Part Target** | `response.text` | `response.candidates[0].content.parts` | `ai_message.tool_calls` |
| **Tool Call Detection** | `re.search(r"Action:\s*(.+)", output)` | `getattr(part, "function_call", None)` | `if ai_message.tool_calls:` |
| **Tool Name Access** | `action_match.group(1)` | `part.function_call.name` | `tool_call["name"]` |
| **Arguments Access** | `json.loads(action_input_match.group(1))` | `dict(part.function_call.args)` | `tool_call["args"]` |
| **Observation Handling** | String concatenation to `history` string | Appended via `types.Part.from_function_response()` | Appended via `ToolMessage(content=..., tool_call_id=...)` |

---
## 📊 Paradigm Comparison Matrix
| Feature / Aspect | `raw_react.py` (Raw ReAct) | `function_calling.py` (GenAI SDK) | `tool_calling.py` (LangChain) |
| :--- | :--- | :--- | :--- |
| **Tool Declaration** | Manual `inspect.signature` string conversion | Native `types.Tool` & JSON Schema dicts | `@tool` decorator abstractions |
| **Model Invocation** | Unstructured text generation (`client.models.generate_content`) | Schema-constrained SDK calls (`types.GenerateContentConfig`) | Abbreviated LLM binding (`chat_model.bind_tools()`) |
| **Parsing Mechanism** | Regex extraction (`re.search`) for Action/Input | Structured SDK attributes (`part.function_call`) | Normalized Python dict (`AIMessage.tool_calls`) |
| **History Tracking** | Plain string concatenation (`prompt + history`) | `types.Content` objects (`role="tool"`) | `BaseMessage` list (`System`, `Human`, `ToolMessage`) |
| **Observability** | LangSmith `@traceable` on custom functions | LangSmith `@traceable` on SDK wrappers | Native LangChain trace integration |

---
## 📁 Repository Structure
```Plaintext
├── raw_react.py          # Pure string-based ReAct loop with regex output parsing
├── function_calling.py   # Native Google GenAI SDK structured function calling implementation
├── tool_calling.py       # High-level LangChain unified tool calling with init_chat_model
├── .env.example          # Environment variables template
├── pyproject.toml        # Dependencies and environment configuration managed by uv
└── README.md             # Documentation
```

## 🚀 Quick Start (Using uv)
1. Installation & Environment Setup
This project uses uv for fast, deterministic Python environment management.

Clone the repository and sync dependencies:

```Bash
git clone [https://github.com/VMXX-23/Langchain_Course_modified.git](https://github.com/VMXX-23/Langchain_Course_modified.git)
cd Langchain_Course_modified

# Sync dependencies using uv
uv sync
```
Create a .env file in the root directory:

```Code snippet
GEMINI_API_KEY=your_gemini_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here  # Optional for tracing
```

2. Running Demonstrations with uv
Execute each script using uv run --active or uv run:

```Bash
# 1. Run String-Based ReAct Paradigm (Manual regex parsing)
uv run --active raw_react.py

# 2. Run Native Google GenAI SDK Function Calling
uv run --active function_calling.py

# 3. Run Unified LangChain Tool Calling
uv run --active tool_calling.py
```

## 💡 Example Terminal Output

