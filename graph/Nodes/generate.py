from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState
"""READS THE DOCUMENTS AND STARTS THE PROCESS"""

def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    #debug
    print("DOCUMENT COUNT:", len(documents))

    for i, doc in enumerate(documents):
        print(f"DOC {i + 1}:")
        print(doc.page_content[:200])
    print("QUESTION:", question)
    attempts = state.get("attempts", 0) + 1

    generation = generation_chain.invoke({"context": documents, "question": question})
    print(f"Generation : {generation} Attempt: :{attempts}")
    return {"documents": documents,
            "question": question,
            "generation": generation,
            "attempts":attempts
            }