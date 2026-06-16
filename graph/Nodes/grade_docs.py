from typing import Any,TypedDict
from graph.state import GraphState

def grade_documents(state: GraphState) -> dict[str, Any]:
    '''Decides if a document is relevant or not, and sets the flag'''
    print("---CHECKING DOCUMENT RELEVANCE TO THIS QUESTION---")

    #debug line
   # print("STATE RECEIVED:", state)
    #print("STATE KEYS:", list(state.keys()))

    question = state["question"]
    documents = state["documents"]

    # No documents retrieved -> go directly to web search
    if not documents:
        print("---NO DOCUMENTS RETRIEVED, USE WEB SEARCH---")
        return {
            "documents": [],
            "question": question,
            "web_search": True,
        }

    filtered_docs = []
   # web_search = False

    for d in documents:
        score = retrieval_grader.invoke(
            {"question":question, "document":d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
        # If everything was filtered out
    if not filtered_docs:
        print("---NO RELEVANT DOCUMENTS FOUND, USE WEB SEARCH---")
        return {
            "question": question,
            "documents": [],
            "web_search": True,
        }

    print(f"---{len(filtered_docs)} RELEVANT DOCUMENTS FOUND---")

    return {"documents":filtered_docs, "question":question, "web_search":web_search}



