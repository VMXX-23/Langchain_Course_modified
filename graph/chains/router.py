from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

class RouteQuery(BaseModel):
    """Route a user query to relevant database"""

    datasource: Literal["VectorStore", "Websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

structured_router_grader = model.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_router_grader