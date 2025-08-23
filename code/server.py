from contextlib import asynccontextmanager
import redis
import hashlib
import json

from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
# from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

load_dotenv(verbose=True)
# global variables (to be reused across requests)
qa_chain = None
DATA_PATH = "/Users/shreenidhi/PycharmProjects/RAG-Powered Technical Documentation Assistant/data/adult-combined-schedule.pdf"
DB_PATH = "chroma_db"
redis_client = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)

#create the data class for the question
class Question(BaseModel):
    question: str

def initialize_rag_pipeline_variables():
    global qa_chain
    #connect to redis client

    # 1. Load data & split
    chunks = load_data_file(DATA_PATH)
    # 2. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

 # 3. Vector store (persisted)
    #only re-embed if the Chroma db is not present
    if os.path.exists(DB_PATH):
        vector_store = Chroma(
            collection_name="Adult_Immunization_Schedule_Collection",
            embedding_function=embeddings,
            persist_directory=DB_PATH,
        )
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            collection_name="Adult_Immunization_Schedule_Collection",
            embedding=embeddings,
            persist_directory=DB_PATH,
        )

    # 4. Retriever using single correct answer
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 5. LLM
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

    # 6. Build QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

def load_data_file(file_path):
    # Use TextLoader for .PDF files
    loader = PyPDFLoader(file_path)
    #this line creates document object
    docs = loader.load()

    # Add metadata to each document (page number + source file)
    for index, doc in enumerate(docs):
        doc.metadata["page_number"] = index + 1
        doc.metadata["source"] = file_path

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Number of documents loaded: {len(docs)}")
    print(f"Number of chunks created: {len(chunks)}")
    return chunks


#utility functions for redis access
#get the key for the query
def get_key_for_query(query):
    return "qa:" + hashlib.sha256(query.encode()).hexdigest()

#get the value from redis
def get_answer_for_query(query):
    global redis_client
    key = get_key_for_query(query)
    return redis_client.get(key)

#store the query and its answer in redis
def store_answer_for_query(query, answer):
    global redis_client
    key = get_key_for_query(query)
    redis_client.set(key, answer, ex=3600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_rag_pipeline_variables()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message":"Welcome to the Recommended Adult Immunization Schedule!"}

@app.post("/user_query")
def get_user_query(question: Question):
    global qa_chain, redis_client
    user_query = question.question
    cached_answer = get_answer_for_query(user_query)
    if cached_answer:
        return {"User Query": cached_answer}

    response = qa_chain.invoke({"query": user_query})
    print(response)
    store_answer_for_query(user_query, response["result"])
    return {"User Query": response["result"]}