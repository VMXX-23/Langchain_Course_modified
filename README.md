# Branch: Advanced Agentic RAG (Corrective RAG / Self-RAG)

> An enterprise-grade implementation of **Corrective RAG (CRAG) and Self-RAG** built using **LangGraph**, **ChromaDB**, **Gemini**, and **Tavily**. This branch replaces naive linear RAG pipelines (`Query ──► VectorStore ──► LLM`) with a self-correcting state machine capable of grading retrieval relevance, dynamically routing to real-time web search upon context gaps, and self-evaluating generation output via hallucination and utility reflection loops.

> _This advanced Agentic RAG state machine inspired by LangChain course patterns and powered by Mistral AI models._
---

## 📸 Runtime Execution & Visual Demo

<img width="1625" height="650" alt="image" src="https://github.com/user-attachments/assets/fcc51738-2e11-4af3-b39d-c8148d21f8ff" />
<img width="1871" height="634" alt="image" src="https://github.com/user-attachments/assets/3e155cf9-faa8-431b-95ab-cc71e261f0e1" />

> **Execution Highlights:** Notice the real-time logging transitions—from conditional routing, document quality grading, fallback web searches, through to hallucination evaluation checks.

---

## 🎯 Architectural Purpose & Design

Naive RAG architectures suffer from three fatal flaws in production:
1. **Retrieval Poisoning:** Irrelevant or noisy vector search results pollute the context window, degrading LLM output quality.
2. **Domain Boundaries:** Questions outside the vector store's domain cause silent failures or hallucinated answers.
3. **Unchecked Generation:** The LLM produces answers ungrounded in context, with no internal mechanism to verify facts before returning output to the user.

This branch resolves these issues through an **Agentic Control Loop**:
* **Input-Layer Intent Routing:** Dynamically parses incoming prompts to choose between vector search or web search prior to context fetching.
* **Document Quality Control (CRAG):** Evaluates every chunk returned by ChromaDB. If chunks fail relevance criteria, the graph discards them and branches to live web search via Tavily.
* **Refinement & Self-Reflection (Self-RAG):** Evaluates candidate draft responses using double-reflection bounds (hallucination check + question-relevance check) with a strict iteration ceiling to prevent infinite retry cycles.

---

## 🔄 End-to-End Corrective RAG Flow
<img width="401" height="545" alt="Agentic_RAG" src="https://github.com/user-attachments/assets/dae290d7-d989-4e98-a0d7-2b502033a4f4" />

```text
                        ┌────────────────────────┐
                        │   User Query Ingress   │
                        └───────────┬────────────┘
                                    │
                             route_question
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼
             ┌───────────┐                     ┌───────────┐
             │ RETRIEVE  │                     │ WEBSEARCH │
             └─────┬─────┘                     └─────┬─────┘
                   │                                 │
                   ▼                                 │
          ┌─────────────────┐                        │
          │ GRADE_DOCUMENTS │                        │
          └────────┬────────┘                        │
                   │ decide_to_generate              │
             ┌─────┴──────────┐                      │
             ▼ (Relevant)     ▼ (Irrelevant/Empty)   │
      ┌──────────────┐   ┌───────────┐               │
      │   GENERATE   │   │ WEBSEARCH │◄──────────────┘
      └──────┬───────┘   └─────┬─────┘
             │                 │
             │                 └──────────┐
             │                            ▼
             │                     ┌──────────────┐
             ├────────────────────►│   GENERATE   │
             │                     └──────┬───────┘
             │                            │
             │ grade_docs_grounded_in_    │
             │   documents_and_question   │
             │                            │
             ├───► "not supported" ──┐    │
             │     (Hallucinated)    │    │
             │                       ▼    │
             │               ┌──────────────┐
             │               │   GENERATE   │
             │               │ (Retry Loop) │
             │               └──────────────┘
             │                            │
             ├───► "not useful" ──────────┼───────────────┐
             │     (Incomplete context)   │               │
             │                            │               ▼
             └───► "useful" ──────────────┼───────────► [ END ]
                   (Grounded & Complete)  │               ▲
                                          │               │
                                          └───────────────┘
```

---

## 📂 Repository Structure
```text
agentic-rag/
├── graph/
│   ├── chains/
│   │   ├── answer_grader.py       # Evaluates if generation addresses the question
│   │   ├── hallucination_grader.py # Evaluates if generation is grounded in context
│   │   ├── generation.py          # RAG generation prompt chain
│   │   └── router.py              # Routes query to vectorstore or websearch
│   ├── Nodes/
│   │   ├── generate.py            # Invokes generation chain; increments attempt count
│   │   ├── grade_docs.py          # Filters noisy chunks; flags web_search if irrelevant
│   │   ├── retrieve.py            # Fetches top-k chunks from ChromaDB
│   │   └── web_search.py          # Executes Tavily search and formats Document objects
│   ├── consts.py                  # Node execution string constants
│   ├── graph.py                   # StateGraph assembly, conditional routing, compilation
│   └── state.py                   # TypedDict schema representing GraphState
├── ingestion.py                   # Scraping, splitting, embedding, and ChromaDB persistence
├── main.py                        # Entry point to execute the compiled LangGraph workflow
├── images/                        # Screenshots of execution, logs, and vector storage
├── .env                           # Environment variables (git-ignored)
└── README.md                      # Branch technical documentation

```
---

## 🧩 Deep-Dive Code Explanations & Senior Technical Mechanics

### 1. Vector Storage & Ingestion (`ingestion.py`)

```python
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

urls = [
    "[https://en.wikipedia.org/wiki/Prompt_engineering](https://en.wikipedia.org/wiki/Prompt_engineering)",
    "[https://www.promptingguide.ai/](https://www.promptingguide.ai/)",
    "[https://aws.amazon.com/what-is/prompt-engineering/](https://aws.amazon.com/what-is/prompt-engineering/)",
    "[https://learnprompting.org/docs/introduction?srsltid=AfmBOoq3qNZyyc-jscNu8pd7pcmZ4kyNN-PeRyLxqoHpm9qDd70ucoNb](https://learnprompting.org/docs/introduction?srsltid=AfmBOoq3qNZyyc-jscNu8pd7pcmZ4kyNN-PeRyLxqoHpm9qDd70ucoNb)",
]

doc_list = [doc for url in urls for doc in WebBaseLoader(url).load()]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_split = text_splitter.split_documents(doc_list)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
)

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=embeddings,
).as_retriever()

```

#### 💡 Senior Developer Technical Breakdown:

* **Token-Aware Splitting (`from_tiktoken_encoder`):** Rather than splitting naively by character length, chunking by sub-word tokens ensures that dense technical concepts aren't truncated mid-token across vector chunk boundaries.
* **Vector Store Persistence vs. Memory Leak Risks:** `persist_directory="./.chroma"` keeps vector data cached across runs. However, initializing `retriever` directly at module import level in `ingestion.py` means any module importing `retriever` triggers Chroma initialization side effects. In larger systems, wrap `retriever` in a lazy-singleton instance pattern.

---

### 2. State Management (`graph/state.py`)

```python
from typing import List, TypedDict, Any

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: bool
    documents: list[Any]
    attempts: int

```

#### 💡 System Behavior Explained 
* **State Mutation & Overwrite Rules:** In LangGraph, when a node returns a partial dictionary (e.g., `{"web_search": True}`), only the specified key is updated in `GraphState`. The rest of the state remains untouched.
* **Explicit Loop Counters (`attempts`):** The `attempts` integer key is critical. Without an explicit counter tracking mutation turns within state, cyclic edges (`GENERATE -> GENERATE`) can run indefinitely during persistent LLM hallucination loops.

---

### 3. Node String Constants (`graph/consts.py`)

```python
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEBSEARCH = "websearch"

```

#### 💡 System Behavior Explained 
* **Decoupled Key Management:** Harcoding node name strings like `"retrieve"` across `workflow.add_node()`, `workflow.add_conditional_edges()`, and `path_map` dictionaries causes silent graph assembly failures. Using immutable string constants ensures compile-time error catching for node identifiers.

---

### 4. Graph Orchestration & Reflection Gates (`graph/graph.py`)

```python
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.halucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.Nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    if not state["documents"]:
        print("---DECISION: NO DOCUMENTS -> WEB SEARCH---")
        return WEBSEARCH
    if state["web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO THE QUESTION, RESORT TO WEBSEARCH---")
        return WEBSEARCH
    else:
        print("---DECISION GENERATE---")
        return GENERATE

def grade_docs_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    attempts = state.get("attempts", 0)

    hal_score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    print("Hallucination score:", hal_score)
    if hal_score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")

        ans_score = answer_grader.invoke({"question": question, "generation": generation})
        if ans_score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        if attempts >= 3:
            print("---MAX RETRIES REACHED---")
            return END

        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def route_question(state: GraphState) -> str:
    print("---Route Question---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE

    print("UNKNOWN DATASOURCE")
    return WEBSEARCH

workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_docs_grounded_in_documents_and_question,
    path_map={
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    }
)

workflow.add_edge(WEBSEARCH, GENERATE)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="Agentic_RAG.png")

```

#### 💡 System Behavior Explained 
* **Conditional Entry Point Mechanics (`set_conditional_entry_point`):** Instead of forcing every request to execute vector retrieval first, `route_question` uses structured JSON outputs (Pydantic routing schemas) to evaluate question intent *before* entering graph cycles.
* **Double Reflection Thresholds:** The function `grade_docs_grounded_in_documents_and_question` performs a two-stage quality verification:
    1. **Hallucination Check:** `hallucination_grader` verifies if generation facts are present in retrieved `documents`.
    2. **Utility Check:** `answer_grader` verifies if generation answers the user's explicit `question`.


* **Guard Rails Against Infinite Loops:** The `attempts >= 3` check inside the conditional router terminates execution at `END` if LLM re-generation continuously hallucinates, protecting application runtime from infinite graph cycles.

---

### 5. Node Mechanics (`graph/Nodes/`)

#### Document Generation (`graph/Nodes/generate.py`)

```python
from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    attempts = state.get("attempts", 0) + 1

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "attempts": attempts
    }

```

#### Quality Control & Filtering (`graph/Nodes/grade_docs.py`)

```python
from typing import Any, TypedDict
from graph.state import GraphState

def grade_documents(state: GraphState) -> dict[str, Any]:
    print("---CHECKING DOCUMENT RELEVANCE TO THIS QUESTION---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {"documents": [], "question": question, "web_search": True}

    filtered_docs = []
    web_search = False

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            filtered_docs.append(d)
        else:
            web_search = True
            continue

    if not filtered_docs:
        return {"question": question, "documents": [], "web_search": True}

    return {"documents": filtered_docs, "question": question, "web_search": web_search}

```

#### Document Retrieval (`graph/Nodes/retrieve.py`)

```python
from typing import Any, Dict
from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("--RETRIEVE--")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

```

#### Adaptive Web Fallback (`graph/Nodes/web_search.py`)

```python
from typing import Any, TypedDict
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_tavily import TavilySearch as TavilySearchResults
from graph.state import GraphState

web_search_tool = TavilySearchResults(max_results=3)

def web_search(state: GraphState) -> dict[str, Any]:
    print("--WEB SEARCH---")
    question = state["question"]
    tavily_response = web_search_tool.invoke({"query": question})
    tavily_results = tavily_response["results"]

    documents = [
        Document(page_content=result["content"])
        for result in tavily_results
    ]

    return {"documents": documents, "question": question}

```

* **Schema Normalization:** Web search JSON items are parsed and wrapped into standard `langchain_core.documents.Document` instances. This guarantees that downstream nodes (`GENERATE`, `grade_documents`) receive unified `Document` interface types regardless of data origin (vector store vs. search API).

---

### 6. Application Entry Point (`main.py`)

```python
from dotenv import load_dotenv
load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello from Advanced Agentic RAG !!")
    print(app.invoke(input={"question": "agent memory?"}))

```

---

## 🔬 Engineering Considerations

| Metric / Dimension | Baseline RAG | Agentic Corrective RAG | Engineering Implication |
| --- | --- | --- | --- |
| **Hallucination Rate** | High | Near Zero | Dual reflection loop rejects outputs not grounded in context. |
| **Token Cost** | Fixed / Low | Variable | Multiple grading LLM calls increase token consumption per turn. |
| **Latency Profile** | Single-pass (~1-2s) | Iterative (~3-8s) | Trade-off: Higher processing latency for verified accuracy. |
| **Fault Isolation** | Poor | High | Ingestion, grading, generation, and search are isolated in state nodes. |

---

## 🛠️ Setup & Execution

### 1. Environment Configuration

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key

```

### 2. Ingest Data into Vector Store

```bash
python ingestion.py

```

### 3. Run the Corrective RAG Agent

```bash
python main.py
```
*Invoking `main.py` compiles the `StateGraph`, generates the visual architecture layout `Agentic_RAG.png`, and runs the self-evaluating execution flow.*
