from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


#Grade documents
class Grade_docs(BaseModel):
    '''Binary score on relevant docs'''

    binary_score : str = Field(
        description="Documents are relevant to the question Yes or No?"
    )


structured_llm_grader = model.with_structured_output(Grade_docs)
'''
system = """
You are a document relevance classifier for a Retrieval-Augmented Generation (RAG) system.

Your task is to determine whether a retrieved document contains information that helps answer the user's question.

Guidelines:
- Focus on semantic relevance, not just exact keyword matches.
- A document is relevant if it:
  - directly answers the question,
  - contains supporting facts,
  - discusses closely related concepts,
  - or provides useful contextual information.
- A document is not relevant if it is off-topic, unrelated, or too vague to help answer the question.

Respond with only:
- 'yes' if relevant
- 'no' if not relevant

Do not output anything else.
"""
'''
system = """
Grade document relevance.

'yes' = contains information useful for answering the question.
'no' = does not contain useful information.

Answer only: yes or no.
"""
'''Prompt to find relevance of docs passed'''
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {docs} \n \n User Question : {question}")
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader