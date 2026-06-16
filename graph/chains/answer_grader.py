from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Answer the question, 'yes' or 'no'"
    )


from dotenv import load_dotenv
load_dotenv()
#client = Client()


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=1,
    # other params...
)

structured_model_grader = model.with_structured_output(GradeAnswer)


system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_model_grader
