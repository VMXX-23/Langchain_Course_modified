from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

"""URLS LIST TO SCRAPE"""
urls = [
    "https://en.wikipedia.org/wiki/Prompt_engineering",
    "https://www.promptingguide.ai/",
    "https://aws.amazon.com/what-is/prompt-engineering/",
    "https://learnprompting.org/docs/introduction?srsltid=AfmBOoq3qNZyyc-jscNu8pd7pcmZ4kyNN-PeRyLxqoHpm9qDd70ucoNb",
]

"""Loading docs"""
doc_list = [doc for url in urls for doc in WebBaseLoader(url).load()]


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_split = text_splitter.split_documents(doc_list)
"""Vector store with persistence"""
"""vectorstore = Chroma.from_documents(
    documents=doc_split,
    collection_name="rag_chroma",
    embedding=GoogleGenerativeAIEmbeddings,
    persist_directory="./.chroma",
)"""

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
   # output_dimensionality=768,  # Suggested: 768, 1536, or 3072 (default)
)
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=embeddings,
).as_retriever()
