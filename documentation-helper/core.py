import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools import Tool
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

#Initialze embeddings (Gemini)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",google_api_key=os.environ["GEMINI_API_KEY"], show_progress_bar=False, chunk_size=5, retry_min_seconds=10)


#Initialize vector store
index = "langchain-doc-assist7"
vectorstore = PineconeVectorStore(index_name=index, embedding=embeddings)

#Initialize the model
llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
        google_api_key=os.environ["GEMINI_API_KEY"],

    max_retries=6,
    temperature=0
)


@tool(response_format="content_and_artifact")
def retrieve_context(query:str):
    """Retrieve Relevant docs and help user answer queries"""
    #Retrieves 4 distinct docs
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=2)

    #Serialize docs for the model
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def run_llm(query: str) -> Dict[str, Any]:
    """
    Runs the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """

    #Creating agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(llm, tools=[retrieve_context], system_prompt=system_prompt)

    #Build messages list
    messages = [{"role": "user", "content": query}]

    #invoke the agent
    response = agent.invoke({"messages":messages})

    #Extract the messages from last AI message
    raw_content = response["messages"][-1].content

    # This prevents the {'type': 'text', 'text': '...'} error in the UI
    if isinstance(raw_content, list) and len(raw_content) > 0:
        answer = raw_content[0].get("text", str(raw_content))
    elif isinstance(raw_content, dict):
        answer = raw_content.get("text", str(raw_content))
    else:
        answer = str(raw_content)


    #Extract context message from tool message artifacts
    context_docs = []

    for message in response["messages"]:
        #Check if this tool message is an artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }


if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)





