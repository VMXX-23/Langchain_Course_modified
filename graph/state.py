from typing import List, TypedDict, Any


class GraphState(TypedDict):
    '''
    Represents the state of graph

    Attributes:
        Question: question
        generation: LLM generation
        web_search : weather to add search
        docs: list of docs
    '''
    question: str
    generation: str
    web_search: bool
    documents: list[Any]
    attempts: int
