import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
#from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tavily import TavilyClient

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=os.environ["GEMINI_API_KEY"], show_progress_bar=False, chunk_size=50, retry_min_seconds=10)

#Retry again after sometime incase of quota exceeded
import time
import asyncio

async def retry_add_documents(batch, batch_num, retries=5, wait_time=60):
    """Retry Gemini embedding when rate limit (429) occurs."""

    for attempt in range(retries):
        try:
            await vectorstore.add_documents(batch)
            return True

        except Exception as e:
            error_msg = str(e)

            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                print(f"\n⚠️ Gemini quota exceeded for batch {batch_num}")
                print(f"⏳ Waiting {wait_time} seconds before retrying...")

                for remaining in range(wait_time, 0, -1):
                    print(f"Retrying in {remaining}s...", end="\r")
                    await asyncio.sleep(1)

                print("\n🔁 Retrying batch...\n")

            else:
                print(f"❌ Non-retryable error: {e}")
                return False

    print(f"❌ Batch {batch_num} failed after {retries} retries.")
    return False
#vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(
     index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings
 )


tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()



async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    # revert this to old one
    async def add_batch(batch: List[Document], batch_num: int):
        success = retry_add_documents(batch, batch_num)

        if success:
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
            return True
        else:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num}")
            return False

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    # Crawl the documentation site

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )


    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=10) #was 500 init, resource exhausted: Quota 100 req per min

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")



if __name__ == "__main__":
    asyncio.run(main())
