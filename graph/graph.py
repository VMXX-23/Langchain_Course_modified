from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.halucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router, RouteQuery
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.Nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

'''FUNCTION TO CHOOSE WHICH NODE TO TAKE'''
def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if not state["documents"]:
        print("---DECISION: NO DOCUMENTS -> WEB SEARCH---")
        return WEBSEARCH

    if state["web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO THE QUESTION, RESORT TO WEBSEARCH---"
              )

        return WEBSEARCH

    else:
        print("---DECISION GENERATE---")
        return GENERATE


def grade_docs_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    attempts = state.get("attempts", 0)

    """Hallucination score"""
    hal_score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    print("Hallucination score:", hal_score)
    if hal_score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")

        ans_score = answer_grader.invoke({"question": question, "generation": generation})
        if ans_score.binary_score: #score is checked for answer grader also for different values of yes
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        if attempts >= 3: #Guard against infinite loop
            print("---MAX RETRIES REACHED---")
            return END

        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def route_question(state: GraphState) -> str:
    print("---Route Question---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE

    #Safety check against unknown source
    print("UNKNOWN DATASOURCE")
    return WEBSEARCH


workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)

workflow.add_node(GRADE_DOCUMENTS, grade_documents)

workflow.add_node(GENERATE, generate)

workflow.add_node(WEBSEARCH, web_search)
workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)
#1st function to run
#workflow.set_entry_point(RETRIEVE)
#Move to grade retrieved docs
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
#Check to either do web search or generate docs
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={
    WEBSEARCH: WEBSEARCH,
    GENERATE: GENERATE},
)

workflow.add_conditional_edges(
    GENERATE,
    grade_docs_grounded_in_documents_and_question,
    path_map={
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    }
)

#Generate response
workflow.add_edge(WEBSEARCH, GENERATE)
#Complete after web search
#workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="Agentic_RAG.png")




