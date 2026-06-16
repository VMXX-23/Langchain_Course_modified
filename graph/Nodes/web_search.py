from typing import Any, TypedDict
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_tavily import TavilySearch as TavilySearchResults
#from langchain_community.tools.tavily_search import TavilySearchResults

from graph.state import GraphState
'''Do web search from tavily '''
#Use TavilySearchResults to get structured data (list of dicts)
web_search_tool = TavilySearchResults(max_results=3)

def web_search(state: GraphState)->dict[str, Any]:
    print("--WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents")
    '''Earlier state["documents"]'''

    #Use ofr TavilySearchResults
    #tavily_results = web_search_tool.invoke({"query": question})


    tavily_response = web_search_tool.invoke({"query": question})
    tavily_results = tavily_response["results"]
    #print(type(tavily_results))
    #print(tavily_results)
    ''' #This part is used to merge into 1 single text 
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )

    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    '''
    # Modif: Combines with individual tavily data
    documents = [
        Document(page_content=result["content"])
        for result in tavily_results
    ]

    #debug check
    print(f"Retrieved {len(documents)} web docs")
    for i, doc in enumerate(documents):
        print(f"\nDOC {i + 1}")
        print(doc.page_content[:300])


    return {"documents": documents,
            "question": question}

if __name__=="__main__":
    web_search(state={"question": "agent memory", "documents":None})