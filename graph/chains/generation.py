#import langchainhub as pull
#from langsmith import Client
'''Resorting to local ChatPromptTemplate due to security issues from hub.pull'''
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()
#client = Client()
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=1,
    # other params...
)


#Chat prompt template
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:"""

#prompt = client.pull_prompt("rlm/rag-prompt")
prompt = ChatPromptTemplate.from_template(template)

generation_chain = prompt | model | StrOutputParser()



