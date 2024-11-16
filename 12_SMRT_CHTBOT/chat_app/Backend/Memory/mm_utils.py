# setup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import TextLoader
#from langchain_community.vectorstores import FAISS
#from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from datetime import datetime, timedelta
from typing import List
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from Memory.myagent import MYGenerativeAgent
from Memory.mymemory import MYGenerativeAgentMemory
import math
import faiss
from langchain_openai import ChatOpenAI
import os


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    return   score                           #1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    #embeddings_model = OpenAIEmbeddings()
    #embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # from sentence_transformers import SentenceTransformer
    # embeddings_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4')
    embeddings_model = HuggingFaceEmbeddings(model_name="msmarco-distilbert-base-v4")
    # Initialize the vectorstore as empty
    embedding_size = 768     #384    #1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=10
    )


def instruction_to_agent_to_remember(agentmemory:MYGenerativeAgentMemory, observation:str):
    instructions = [observation]
    for instruction in instructions:
        agentmemory.add_memory(instruction)
    print(f"instructions added successfully")
    return agentmemory.memory_retriever.memory_stream


def db_storage_of_agent_mmry(agentmemory:MYGenerativeAgentMemory,path_to_db:str):
    try:
        agentmemory.memory_retriever.save_on_local_vector_db(path_to_db)
        print(f"success")
    except:
        agentmemory.memory_retriever.vectorstore.save_local(path_to_db)
        print(f"success")



def fetch_from_db(agentmemory:MYGenerativeAgentMemory,path_to_db:str):
    embeddings_model = HuggingFaceEmbeddings(model_name="msmarco-distilbert-base-v4")
    new_db = agentmemory.memory_retriever.vectorstore.load_local(path_to_db, embeddings_model,allow_dangerous_deserialization = True)
    lst = []
    for i in new_db.index_to_docstore_id:
        #print(new_db.index_to_docstore_id[i])
        lst.append(new_db.docstore.search(new_db.index_to_docstore_id[i]))

    return lst


def interview_agent(agent: MYGenerativeAgent, message: str,t_response:str) -> str:
    """Help the notebook user interact with the agent."""
    USER_NAME = "Sam" 
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message,t_response)