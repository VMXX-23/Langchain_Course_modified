from dotenv import load_dotenv
from pprint import pprint
load_dotenv()

from graph.chains.retrieval_grader import Grade_docs, retrieval_grader
from graph.chains.tests import Grade_Hallucinations, hallucination_grader
from graph.chains.router import route_prompt, question_router
from graph.chains.generation import generation_chain
from ingestion import retriever

def test_retriever_grader_answer_yes() -> None:
    question = "Agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: Grade_docs = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retriever_grader_answer_no() -> None:
    question = "Agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: Grade_docs = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain()-> None:
    question = 'agent memory'
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context":docs, "question":question})
    pprint(generation)


def test_hallucination_grader_answer_yes()-> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: Grade_Hallucinations  = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: Grade_Hallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score

def test_router_to_vectorstore() -> None:
    question = "agent memory"

    res: Routequery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


"""In case of no inputs from vectorstore"""
def test_router_to_websearch() -> None:
    question = "how to make pizza"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"

