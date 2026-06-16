from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("--RETRIEVE--")
    question = state["question"]

    documents = retriever.invoke(question)

    #debug line
    print("Retrieved:", len(documents), "documents")
    return {"documents": documents,
            "question": question,
            }