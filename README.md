# 🔍 Specialized AI Job Search Agent — Branch: `ReAct-JobSeeker`

> **Architectural Paradigm:** Autonomous ReAct State Machine with Schema-Enforced Tool Calling, Time-Bounded Web Search, and Defensive Data Validation  
> **Primary Runtime Stack:** Python 3.10+ | LangGraph (`create_react_agent`) | Google Gemini 2.5 Flash | Tavily Search API | Pydantic v2  

---

## 📌 Project Overview

This repository houses an enterprise-grade, autonomous job search assistant engineered using **LangGraph**, **LangChain**, and **Google Gemini 2.5 Flash**. Unlike naive LLM wrappers that rely on unconstrained text output or return stale search index entries, this system enforces a **deterministic execution pipeline**:

1. **Input Normalization:** Strips conversational noise and extracts core role intent via regular expressions.
2. **Boolean-Optimized Web Retrieval:** Dynamically constructs boolean search queries targeting premier recruitment portals (`LinkedIn`, `Naukri`, `Indeed`) while filtering out low-quality aggregators (`Quikr`, `OLX`, `YouTube`).
3. **Structured Pydantic Validation:** Guarantees strict output contracts (`job_title`, `job_description`, `posted_date`, `source`) while defending against LLM JSON truncation and trailing payload artifacts (`{}`).
4. **Recency Ranking:** Uses custom post-processing sorting algorithms (`parse_recency_rank`) to prioritize freshly posted roles (`"Today"`, `"1 day ago"`) at the top of the output.

---

## 📸 Execution Showcase

<img width="1111" height="654" alt="image" src="https://github.com/user-attachments/assets/2404d281-b5b7-43bc-9377-599730bdff8d" />

Model Example
```text
Welcome to Job search tool 

Enter preferred Job role: SAP ABAP Developer
:>> Optimized Search Query: "SAP ABAP Developer" job openings India hiring site:[linkedin.com/jobs](https://linkedin.com/jobs) OR site:naukri.com

Found 5 job postings (Filtered & Sorted by Latest Date):

--- Job #1 ---
📌 Job Title:   SAP ABAP Technical Developer
🕒 Posted Date:  1 day ago
📝 Description: PwC India is hiring for Hyderabad location. Key skills: SAP S/4HANA, OData, WRICEF, and Object-Oriented ABAP.
🔗 Source:      [https://in.linkedin.com/jobs/application-developer-sap-abap-jobs-hyderabad](https://in.linkedin.com/jobs/application-developer-sap-abap-jobs-hyderabad)

--- Job #2 ---
📌 Job Title:   SAP ABAP Consultant (m/f/d)
🕒 Posted Date:  2 days ago
📝 Description: BASF is hiring for full-time ABAP Developer roles in Hyderabad/Bengaluru.
🔗 Source:      [https://in.linkedin.com/jobs/immediate-hiring-for-sap-abap-jobs](https://in.linkedin.com/jobs/immediate-hiring-for-sap-abap-jobs)
....
```
---
## 🎯 Purpose & Scope
This project is a technical implementation demonstrating an end-to-end, time-aware job discovery CLI application.

Rather than relying on unconstrained LLM web summaries that risk hallucinating open roles or returning dormant, months-old listings, the implementation explicitly wires together the major components involved in a production-ready agentic search workflow:
* **Query Normalization**: Regex-based sanitization stripping conversational noise ("find jobs for", "openings for") to maximize search engine precision.
* **Time-Bounded Web Retrieval**: Domain-scoped search querying premier Indian job portals (LinkedIn, Naukri, Indeed, Glassdoor) restricted to active postings (time_range="w").
* **Schema Enforcement**: Pydantic models (JobPosting, AgentResponse) ensuring valid structured data outputs with defensive fallback defaults.
* **Artifact Defense**: Schema design engineered to prevent OutputParserException failures caused by LLM truncation or trailing empty objects ({}).
* **Date-Aware Recency Ranking**: Custom post-processing ranking algorithm pushing immediate postings ("Today", "1 day ago", "24 hours ago") to the top of the output.
* **Resilience & Retry Strategy**: Tailored backoff and retry handlers addressing HTTP 429 (Quota Exceeded) and HTTP 503 (Server Overload) API errors.
* **Agentic Orchestration**: LangGraph ReAct runner (create_react_agent) enforcing system prompt guardrails against non-tool execution.

The project is intentionally structured to demonstrate not only how to build an AI job agent, but also how to handle real-world API rate limits and structural LLM output edge cases.

---
## 🔄 End-to-End Architecture
The application operates as a real-time, tool-augmented reasoning loop:

```Plaintext
               User CLI Input ("SAP ABAP Developer")
                                │
                                ▼
                   ┌──────────────────────────┐
                   │  Regex Query Cleaner     │
                   └────────────┬─────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │  Tavily Search Engine    │
                   │  (Plain-Text Doc Blocks) │
                   └────────────┬─────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │  ReAct Agent Loop        │
                   │  (Gemini 2.5 Flash)      │
                   └────────────┬─────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │  Pydantic Validation     │
                   │  (AgentResponse Schema)  │
                   └────────────┬─────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │  Date-Recency Sorter     │
                   │  parse_recency_rank()    │
                   └────────────┬─────────────┘
                                │
                                ▼
               Terminal Output (Sorted Jobs List)
```

--- 

📂 Modular Repository Structure
```Plaintext
react_search_agent/
├── schemas.py      # Pydantic data models & date-aware output formatting logic
├── tools.py        # Web search tool, regex query cleaner & Tavily API wrapper
├── main.py         # Light CLI entry point & LangGraph agent configuration
├── .env            # Environment API keys (git-ignored)
└── README.md       # Technical branch documentation
```
**Module Breakdown**
1. schemas.py
   Defines JobPosting and AgentResponse Pydantic models with default fallbacks. Houses parse_recency_rank() and get_text() to sort and format structured outputs cleanly for terminal display.

2. tools.py
   Contains the @tool search wrapper. Cleans conversational fluff via re.sub, executes boolean-targeted web searches, excludes noisy domain aggregators, and converts search dictionary payloads into plain-text string blocks for Gemini.

3. main.py
   The CLI application entry point. Configures ChatGoogleGenerativeAI with a 4096 token ceiling, initializes the LangGraph create_react_agent, and executes the user search loop.

---

## 🛠️ Common Issues & How They Were Rectified
During development, several real-world agent failure modes were identified and systematically resolved:

### 1. 🚨 Empty structured_response (jobs []) Despite Valid Search Results
Root Cause:
* The tool was returning a **raw Python list[dict] directly from tavily.search()**. LangChain's schema serializer failed to pass structured context back to Gemini.
* A **system prompt instruction ("Summarize search results...") conflicted with structured JSON generation**, causing Gemini to write a plain text summary instead of filling the jobs array.

Rectification:
* Updated @tool search to serialize raw results into clear plain-text string blocks (Title, URL, Content).
* Simplified system prompt to strictly instruct: "Extract individual job listings and populate the jobs array in the structured output format."

### 2. 💥 OutputParserException on Trailing Empty Objects ({})
Root Cause:
* When nearing context completion bounds, **Gemini occasionally generated trailing empty objects ({}) or partial keys inside the jobs array (e.g., jobs.12.job_title: Field required).**

Rectification: Added default values to all fields inside the Pydantic JobPosting model:

```Python
class JobPosting(BaseModel):
    job_title: str = Field(default="N/A", description="Job title")
    job_description: str = Field(default="No description.", description="Summary")
    posted_date: str = Field(default="Recently", description="Posting date")
    source: str = Field(default="", description="Source link")
```
Trailing {} objects now parse safely as default entries and are silently filtered out during list iteration (j.job_title != "N/A").

### 3. ✂️ Mid-Sentence JSON Truncation
Root Cause: 
* The **default max_tokens limit (1024) was too small to accommodate large JSON arrays** containing 5+ job objects.

Rectification: Raised max_tokens ceiling to 4096 in ChatGoogleGenerativeAI model initialization.

### 4. 🔍 Irrelevant Search Fallbacks (Dictionary Definitions, YouTube Videos)
Root Cause: 
* Combining **strict quotes ("SAP ABAP developer jobs") with narrow time_range="w" parameters yielded zero matches on Tavily**, causing the search API to fall back to generic internet domain indexes (e.g., Merriam-Webster, YouTube podcasts).

Rectification:
* Relaxed rigid quotes to catch role variations (e.g., SAP ABAP Consultant, Senior ABAP Lead).
* Explicitly added youtube.com, merriam-webster.com, quikr.com, and olx.in to exclude_domains.

---

## 🔐 Environment Setup & Config

1. Create a .env file in the project root
   
```Code snippet
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

2. 🚀 Running the Application
Install Dependencies:

```Bash
uv sync
```
Launch CLI Agent:

```Bash
uv run main.py
```
---

## 💡 Future Scope & Roadmap

[ ] **Location & Work-Mode Filters**: Add optional city filtering (e.g., Hyderabad, Bengaluru), Add overseas location filtering for location specific data and remote-only options.

[ ] **Automatic Export Pipeline**: Save validated jobs directly to local JSON/CSV files upon completion.

[ ] **LangGraph MemorySaver Integration**: Attach a thread-based checkpointer to enable multi-turn conversational follow-ups (e.g., "Filter those down to remote-only").
