# LangChain Documentation Helper — Branch: `doc-assist`

> An **end-to-end application** for documentation intelligence built with **LangChain**, **Gemini**, **Pinecone**, **Tavily**, and **Streamlit**. This branch demonstrates how to move beyond static chains into full agentic RAG architectures with tool-artifact decoupling and quota-aware document ingestion pipelines.

---

## 🎯 Purpose & Scope

This project is a technical implementation demonstrating an **end-to-end documentation-aware question-answering application**.

Rather than treating LangChain as a black-box framework, the implementation explicitly wires together the major components involved in a production-style RAG workflow:

- **Ingestion ETL:** Web crawling, document creation, text chunking, and batched embedding.
- **Embeddings:** Gemini embedding model for converting document chunks into vectors.
- **Vector Storage:** Pinecone serverless vector database for persistent semantic retrieval.
- **Similarity Search:** Dense vector retrieval of documentation relevant to the user's query.
- **Agentic RAG:** LangChain agent using `create_agent` and `init_chat_model`.
- **Tool Calling:** A dedicated `retrieve_context` tool for retrieving documentation from Pinecone.
- **Artifact Propagation:** Separation of LLM-facing textual context from application-facing `Document` objects and metadata.
- **Source Attribution:** Retrieved document metadata is propagated back to the Streamlit UI.
- **Observability:** LangSmith integration for tracing LangChain executions.
- **User Interface:** Streamlit-based conversational interface with session history.

The project is intentionally structured to demonstrate not only **how to make RAG work**, but also how the different LangChain abstractions interact internally.

---

## 🔄 End-to-End Architecture

The application is divided into two major phases:

### Phase 1 — Documentation Ingestion

```text
             Documentation Website
                     │
                     ▼
              TavilyCrawl
                     │
                     ▼
              Raw Documents
                     │
                     ▼
        RecursiveCharacterTextSplitter
              (1000 / 200)
                     │
                     ▼
             Document Chunks
                     │
                     ▼
          Gemini Embeddings API
                     │
                     ▼
        Dense Vector Representations
                     │
                     ▼
        ┌─────────────────────────┐
        │   Pinecone Serverless   │
        │      Vector Index       │
        └─────────────────────────┘


```
 ## Phase 2 — Retrieval & Inference Phase
```text
 User Question
      │
      ▼
┌───────────────┐
│   Streamlit   │
│   Frontend    │
└───────┬───────┘
        │
        ▼
┌────────────────────┐
│ LangChain Agent    │
│    create_agent    │
└─────────┬──────────┘
          │
          │ Tool Call
          ▼
┌────────────────────┐
│ retrieve_context() │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Pinecone Retriever │
│ Similarity Search  │
└─────────┬──────────┘
          │
          ▼
    Relevant Documents
       │          │
       │          └─────────────────────┐
       ▼                                │
Serialized Context                      │
       │                                │
       ▼                                │
┌──────────────────────┐                │
│ Gemini 2.5 Flash LLM │◄───────────────┘
└──────────┬───────────┘
           │
           ▼
   Grounded Answer
           │
           ▼
   Source Attribution
           │
           ▼
      Streamlit UI

```

---

## 🛠️ Technology Stack & Environment

| Layer | Technology | Purpose |
| --- | --- | --- |
| **Frontend UI** | Streamlit | Conversational UI, session history, source citation rendering |
| **Agent Orchestration** | LangChain (`create_agent`) | Tool-calling reasoning loop and execution agent |
| **Chat LLM** | Gemini 2.5 Flash | Core LLM initialized via `init_chat_model` |
| **Embeddings** | `gemini-embedding-001` | Semantic vector generation for document chunks |
| **Vector DB** | Pinecone (AWS Serverless) |> Persistent vector storage, dense similarity search, and document metadata storage|
| **Web Ingestion** | Tavily (`TavilyCrawl`) | Web crawling and raw document extraction |
| **Observability** | LangSmith | Tracing agent loops, tool invocations, and LLM calls |
| **Environment** | `python-dotenv` | API key management and configuration |

---

## 📂 Repository Structure

```text
doc-assist/
├── ingestion.py     # Document crawling, recursive chunking, batch embeddings, Pinecone upload
├── core.py          # Gemini LLM initialization, Pinecone retriever tool, agent setup
├── main.py          # Streamlit frontend UI, chat session state, tool artifact parsing
├── logger.py        # Centralized terminal logger with color-coded status output
├── .env             # API Keys (git-ignored)
└── README.md        # Technical branch documentation

```

### Module Breakdown

#### `ingestion.py`

Executes the document processing pipeline. Crawls external documentation, splits raw text into semantic units, generates vector embeddings in controlled batches to prevent rate limits, and upserts payloads into Pinecone.

#### `core.py`

Houses core RAG logic. Configures the Pinecone retriever tool using the `@tool(response_format="content_and_artifact")` decorator and initializes the Gemini 2.5 Flash agent.

#### `main.py`

The **end-to-end application** entry point. Built with Streamlit, it maintains user session state, handles user queries, invokes the LangChain agent, and parses/extracts `ToolMessage` artifacts and document metadata to render retrieved source information in the Streamlit interface.

#### `logger.py`

Provides color-coded terminal outputs (`log_info`, `log_success`, `log_error`, `log_header`) for step-by-step ingestion and execution monitoring.

---

## 💡 Key Technical Insights & Niche Engineering Tips

### 1. Pinecone Vector Index Dimension Alignment

* **Critical Rule:** The vector dimension in your Pinecone index **must exactly match** the output dimension of your embedding model.
* **Important:** When changing the embedding model or embedding configuration, verify the resulting vector dimension before indexing documents.

A mismatch results in a Pinecone error such as:

```text
Vector dimension 3072 does not match the dimension of the index 1536
```
* **Serverless Setup:** Always verify your AWS Pinecone serverless index specs match your active embedding model prior to running `ingestion.py`.

### 2. Embedding API Quota & Batch Ingestion

Crawling large documentation sites produces thousands of text chunks. Sending all chunks simultaneously to an embedding API triggers `429 RESOURCE_EXHAUSTED` errors.

```text
5,077 Document Chunks
  ├── Batch 1 [Chunks 0-10]   ──► Success
  ├── Batch 2 [Chunks 11-20]  ──► 429 Rate Limit
  │     └── Wait / Retry ──► Success
  └── Batch N ...             ──► Success

```

* **Solution:** `ingestion.py` executes async batched ingestion (batch_size=10) combined with retry handling for 429 RESOURCE_EXHAUSTED responses.
  
> **Concurrency consideration:** The current implementation creates batches asynchronously. While batching reduces individual request sizes, concurrent batch execution can still generate multiple embedding requests within the same quota window. For strict per-minute quotas, sequential batch processing or an explicit rate limiter is preferable.

### 3. Tool Artifact Decoupling (`content_and_artifact`)

Standard LangChain tools return a single string to the LLM context window. This application uses `@tool(response_format="content_and_artifact")` to return a dual payload:

```python
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    docs = retriever.invoke(query)
    
    # 1. Content: Concatenated text passed to the LLM prompt window
    content = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Artifact: Raw Document objects retained for application metadata (URLs)
    artifact = docs
    
    return content, artifact

```

In `main.py`, the Streamlit frontend inspects the agent's `ToolMessage` execution sequence, extracts the `artifact` containing raw `Document` objects, and dynamically builds source citation links without bloating the LLM prompt window.
The distinction is important because the **LLM does not need the complete application-level representation of every retrieved document**. It primarily needs serialized textual context, while the application benefits from retaining the original `Document` objects and metadata.

This creates a useful separation:
```
Tool
 │
 ├── Content → LLM context
 │
 └── Artifact → Application / UI
                 │
                 └── Source metadata
```
---

## 🔐 Environment Setup & Config

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
TAVILY_API_KEY=your_tavily_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=LangChain-DocAssist

```

---

## 🚀 Running the Application

1. **Run Ingestion Pipeline:**
```bash
python ingestion.py

```


*Crawls target sites, splits content into chunks, generates embeddings, and uploads vectors to Pinecone.*
2. **Launch Streamlit Frontend:**
```bash
streamlit run main.py

```

*Opens the interactive web interface for documentation QA.*

---

## 🚀 Future Scope & Enhancements

* [ ] **Custom URL/Website Input:** Allow users to dynamically input any documentation URL directly within the Streamlit UI to trigger on-demand ingestion.
* [ ] **Secondary Web Scraper Integration:** Integrate alternative fallbacks (e.g., BeautifulSoup / Playwright) alongside Tavily for complex JavaScript-heavy sites.
* [ ] **Hybrid Search & Reranking:** Combine keyword BM25 search with Pinecone dense vector retrieval and apply Cohere reranking for improved result relevance.
* [ ] **Streaming Responses:** Implement token-by-token streaming from Gemini to the Streamlit UI for lower perceived latency.

