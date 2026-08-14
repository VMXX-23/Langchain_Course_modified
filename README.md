# LangChain Course – LLM & Prompt Template Experiments

This repository contains my implementations and experiments while learning **LangChain** and **LLM application development**.

The project is based on concepts covered during a LangChain course, but the implementations have been **independently modified and extended** to experiment with different models, prompt workflows, integrations, and LangChain concepts.

The repository is being developed incrementally, with additional **LangChain experiments, RAG implementations, and LLM applications** planned for future branches.

---

## Current Implementation – Prompt Template Summarizer

The current `main` branch contains a **prompt-template-based summarization application** implemented using LangChain.

The application takes structured information about a person, inserts it into a reusable prompt template, and sends the resulting prompt to an LLM.

The prompt asks the model to generate:

-  A short summary
-  Two interesting facts
-  A creative title
-  A fictional interview question and answer

### Prompt Flow

```text
Information
     │
     ▼
PromptTemplate
     │
     ▼
    LLM
     │
     ▼
Generated Response


```

---

## Implementations

The current implementation explores the same prompt-template workflow using different LLM approaches.

### 1. Gemini

The project includes an implementation using Google's **Gemini API**.

**Model**: Gemini: 2.5 flash

The Gemini client is initialized using an API key loaded from environment variables:

```python
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
```

The application demonstates both **direct Gemini invocation without LCEL** and a **LangChain-based LCEL implementation**.

#### Gemini Without LCEL

In the non-LCEL approach, the prompt is formatted first and then passed directly to the Gemini API.

```python
prompt_text = summary_prompt_template.format(
    information=information
)

response = client.models.generate_content(
    model="models/gemini-2.5-flash",
    contents=prompt_text
)
```

The flow is:

```text
Information
     │
     ▼
PromptTemplate.format()
     │
     ▼
Formatted Prompt
     │
     ▼
Gemini API
     │
     ▼
Response
```

This approach keeps **prompt creation and model invocation as separate steps**.


---

### 2. Ollama + Gemma

The project also includes a **local LLM implementation** using **Ollama** with Google's **Gemma** model.

The LangChain integration uses:

```python
from langchain_ollama import ChatOllama
```

The model is initialized with:

```python
llm = ChatOllama(
    model="gemma3:270m",
    temperature=0
)
```

The prompt template is connected to the local model using LCEL:

```python
chain = summary_prompt_template | llm

response = chain.invoke({
    "information": information
})
```

The flow is:

```text
Information
     │
     ▼
PromptTemplate
     │
     ▼
Gemma 3 via Ollama
     │
     ▼
Generated Response
```

This allows the same prompt-template workflow to be tested with a **locally running LLM**, without sending the inference request to Gemini.

---

## LCEL vs. Without LCEL

One of the goals of this implementation is to understand the difference between **direct invocation** and **LCEL-based composition**.

### With LCEL

The components are composed into a reusable chain:

```python
chain = summary_prompt_template | llm
```

and invoked using:

```python
response = chain.invoke({
    "information": information
})
```

The resulting pipeline is:

```text
Information
     │
     ▼
PromptTemplate
     │
     ▼
LLM
     │
     ▼
Response
```

## Takeaway
LCEL provides a more **declarative, composable, and reusable** way of connecting LangChain components.

---

## Technologies Used

- **Python**
- **LangChain**
- **Google Gemini API**
- **Ollama**
- **Gemma**
- **LangChain Core**
- **LangChain Ollama**
- **LCEL**
- **python-dotenv**

---

## Gemini vs. Ollama + Gemma

| Implementation | Model | Execution | API Required |
|---|---|---|---|
| Gemini without LCEL | Gemini | Direct API invocation | Gemini API |
| Gemini with LCEL | Gemini | LangChain + LCEL | Gemini API |
| Ollama + Gemma | Gemma | Local inference + LCEL | Local Ollama |

The implementations allow the same **prompt-template workflow** to be experimented with using both **cloud-based and locally hosted LLMs**.

---

## Output/Working

### Gemini : without LCEL
> Gemini Output
<img width="854" height="424" alt="image" src="https://github.com/user-attachments/assets/bdc40c23-c422-413a-91e3-dd2ed1ed12ff" />

### Ollama + Gemma : with LCEL
> Ollama instance
<img width="691" height="173" alt="image" src="https://github.com/user-attachments/assets/a98d7b72-7e43-465e-bcd5-9e7014cfaa3a" />

> Ollama GUI
<img width="985" height="785" alt="image" src="https://github.com/user-attachments/assets/efcf94a6-e4e9-497a-a74d-96d4422f8313" />

> Ollama + Gemma Output
<img width="847" height="254" alt="image" src="https://github.com/user-attachments/assets/7dd23285-33d3-4277-a800-5bbdccb8c46f" />

---

## Environment Variables

The Gemini API key is loaded from a local `.env` file:

```env
GEMINI_API_KEY=your_api_key_here :{
```

The `.env` file should **not** be committed to the repository.

Recommended `.gitignore` entries:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

---

## Future Work

This repository will be expanded with additional experiments and implementations.

Planned areas include:

- **Retrieval-Augmented Generation (RAG)**
- **Embeddings**
- **Vector stores**
- **Retrievers**
- **Pinecone**
- **LCEL-based RAG pipelines**
- **Tool calling**
- **Agents**
- **Additional LLM providers**
- **Local and cloud-based models**

Future implementations will be added through **additional branches** as the project evolves.
   
