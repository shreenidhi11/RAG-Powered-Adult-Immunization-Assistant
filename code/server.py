import csv
from contextlib import asynccontextmanager
from datetime import datetime
from tokenize import endpats

import numpy as np
import redis
#Implementing Semantic Cache with Redis Vector Search - Feature-1
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.exceptions import ResponseError
import hashlib
import json
# Feature 2 - Monitoring & logging → Track user queries, cache hits/misses, and latency.
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
import time
import logging
#Feature 3 - Adding follow up questions to the answer given by the RAG ---to be done
from langchain.prompts import PromptTemplate

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
embeddings = None
metrics_app = None
Request_counter = None
cache_hit_counter, cache_miss_counter, llm_call_counter = None, None, None
# Define the path to your feedback CSV file
FEEDBACK_CSV_FILE = "feedback.csv"
#create the data class for the question
class Question(BaseModel):
    question: str

class Feedback(BaseModel):
    query: str
    answer: str
    feedback_type: int | None


def initialize_rag_pipeline_variables():
    global qa_chain, embeddings, Request_counter, cache_hit_counter, cache_miss_counter, llm_call_counter
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

    # Feature 2 - Configure logging
    logging.basicConfig(
        filename="rag_app.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # defining the prometheus configuration
    Request_counter = Counter(
        "app_request_total",  # name of the metric
        "total number of request to the application",  # metric description
        ["endpoint"],  # labels(e.g. endpoint name)
    )

    cache_hit_counter = Counter(
        "app_cache_hit_total", "total number of request to the application", ["endpoint"],
    )

    cache_miss_counter = Counter(
        "app_cache_miss_total", "total number of request to the application", ["endpoint"],
    )

    llm_call_counter = Counter(
        "app_llm_call_total", "total number of request to the application", ["endpoint"],
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

#Implementing Semantic Cache with Redis Vector Search - Feature-1
#sample schema is below
# {
#   "query": "What is COVID vaccine?",
#   "embedding": [0.123, -0.455, ...],  # numerical vector
#   "answer": "The CDC recommends..."
# }

def define_redis_index():
    index_name = "query_index"
    try:
        redis_client.ft(index_name).info()  # will fail if index doesn't exist
    except ResponseError:
        # Index doesn't exist → create it
        redis_client.ft(index_name).create_index([
            TextField("query"),
            TextField("answer"),
            VectorField("embedding", "FLAT", {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"})
        ], definition=IndexDefinition(prefix=["q:"], index_type=IndexType.HASH))
    #schema creation
    # redis_schema = (TextField("query"), TextField("answer"),VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"}))
    # #insert this schema in redis
    # redis_client.ft(index_name).create_index(redis_schema, definition=IndexDefinition(prefix=["q:"], index_type=IndexType.HASH))

def store_in_redis(user_query, model_answer):
    global embeddings
    query_vec = embeddings.embed_query(user_query)
    # pack as bytes for Redis
    query_vec_bytes = np.array(query_vec, dtype=np.float32).tobytes()
    redis_client.hset(
        f"q:{user_query}",
        mapping={
            "query": user_query,
            "answer": model_answer,
            "embedding": query_vec_bytes
        }
    )


def retrieve_from_redis(user_query, top=1, threshold=0.85):
    global embeddings, redis_client

    # 1. Embed query
    query_vec = embeddings.embed_query(user_query)
    query_vec_bytes = np.array(query_vec, dtype=np.float32).tobytes()

    # 2. Perform vector similarity search
    results = redis_client.execute_command(
        "FT.SEARCH", "query_index",
        f"*=>[KNN {top} @embedding $vec_param AS score]",
        "SORTBY", "score",
        "RETURN", 2, "query", "answer",
        "PARAMS", 2, "vec_param", query_vec_bytes,
        "DIALECT", 2
    )
    # 3. Parse results
    # results looks like: [num_results, doc_id1, [field, value, ...], doc_id2, [field, value, ...], ...]
    if len(results) > 1:
        num_results = results[0]
        docs = []
        for i in range(1, len(results), 2):
            doc_id = results[i]
            fields = dict(zip(results[i + 1][::2], results[i + 1][1::2]))
            score = float(fields.get("score", 1.0))
            if score >= threshold:  # Lower score = closer match
                docs.append((doc_id, fields))
        if docs:
            return docs[0][1].get("answer")  # return top answer

    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_rag_pipeline_variables()
    define_redis_index()
    yield

app = FastAPI(lifespan=lifespan)
#code changes for adding the prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

#feature 4
def save_feedback_to_csv(feedback_data: dict):
    """Appends feedback data to a CSV file."""

    # Check if the file exists to determine if we need to write headers
    file_exists = os.path.isfile(FEEDBACK_CSV_FILE)

    with open(FEEDBACK_CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        fieldnames = ["timestamp", "question", "answer", "feedback_type", "user_text"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Write the header row if the file is new

        # Prepare the row with a timestamp
        row = {
            "timestamp": datetime.now().isoformat(),
            "question": feedback_data.get("query"),
            "answer": feedback_data.get("answer"),
            "feedback_type": feedback_data.get("feedback_type"),
        }

        writer.writerow(row)


@app.get("/")
def read_root():
    Request_counter.labels(endpoint="/").inc()
    return {"message":"Welcome to the Recommended Adult Immunization Schedule!"}

#Implementing Semantic Cache with Redis Vector Search - Feature-1
@app.post("/user_query")
def get_user_query(question: Question):
    global qa_chain, redis_client, cache_hit_counter
    #Feature 2 - Configure logging
    start_time = time.time()
    user_query = question.question
    cached_answer = retrieve_from_redis(user_query)

    if cached_answer:
        cache_hit_counter.labels(endpoint="/user_query").inc()
        latency = time.time() - start_time
        logging.info(f"Query='{user_query}' | Cache=HIT | Latency={latency:.3f}s")
        return {"User Query": cached_answer}

    response = qa_chain.invoke({"query": user_query})
    llm_call_counter.labels(endpoint="/user_query").inc()
    latency = time.time() - start_time
    logging.info(f"Query='{user_query}' | Cache=MISS | Latency={latency:.3f}s")
    store_in_redis(user_query, response["result"])
    return {"User Query": response["result"]}


#Feature 3 - Feeback URL
@app.post("/submit_feedback")
def submit_feedback(feedback: Feedback):
    logging.info(f"Feedback Question='{feedback.query}'")
    logging.info(f"Feedback Answer='{feedback.answer}'")
    logging.info(f"Feedback Type='{feedback.feedback_type}'")
    #adding the feedback to csv file
    feedback_dict = feedback.model_dump()
    save_feedback_to_csv(feedback_dict)
