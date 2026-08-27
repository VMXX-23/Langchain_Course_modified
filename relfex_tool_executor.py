# For executing 2 tools based on Tavily search
from dotenv import load_dotenv
load_dotenv()
from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from reflex_schemas import AnswerQuestion, ReviseAnswer
tavily_search = TavilySearch(max_results=5)


def run_queries(search_queries: list[str], **kwargs):
    # Run the generated queries
    return tavily_search.batch([{"query": query} for query in search_queries])


execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
