from typing import List as list

from pydantic import BaseModel, Field


class Reflection(BaseModel()):
    missing: str = Field(description="Critique of what is missing")
    superflous: str = Field(description="Critique of what is superflous")


class AnswerQuestion(BaseModel()):
    answer: str = Field(description="250 word detailed answer to your question")
    reflection: Reflection = Field("Your reflection on the initial answer")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: list[str] = Field(
        description="Citations motivating your updated answer."
    )
