from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from Memory.myagent import MYGenerativeAgent
from Memory.mymemory import MYGenerativeAgentMemory
from langchain_openai import ChatOpenAI
import os
from Memory.mm_utils import create_new_memory_retriever



class Memory_Agent:

    def __init__(self, api_key, path_to_db):
        self.api_key = api_key
        self.model = "gpt-3.5-turbo-0125"
        self.path_to_db = path_to_db
        self.embeddings_model = HuggingFaceEmbeddings(model_name="msmarco-distilbert-base-v4")

        LLM = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

        self.siri_memory = MYGenerativeAgentMemory(
            llm=LLM,
            memory_retriever=create_new_memory_retriever(),
            verbose=False,
            reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
            )

        self.siri = MYGenerativeAgent(
            name="Siri",
            age=25,
            traits="attentive , supportive, helpful",  # You can add more persistent traits here
            status=" a helpful personal assistant to Sam ",  # When connected to a virtual world, we can have the characters update their status
            memory_retriever=create_new_memory_retriever(),
            llm=LLM,
            memory=self.siri_memory,
            )

        
    def store_db(self):
        #db_storage_of_agent_mmry(self.siri_memory, path_to_db)
        try:
            self.siri_memory.memory_retriever.save_on_local_vector_db(self.path_to_db)
            print(f"success")
        except:
            self.siri_memory.memory_retriever.vectorstore.save_local(self.path_to_db)
            print(f"success")



    def fetch_db(self):
        
        new_db = self.siri_memory.memory_retriever.vectorstore.load_local(self.path_to_db, self.embeddings_model,allow_dangerous_deserialization = True)
        lst = []
        for i in new_db.index_to_docstore_id:
            #print(new_db.index_to_docstore_id[i])
            lst.append(new_db.docstore.search(new_db.index_to_docstore_id[i]))

        return lst
    
    def add_instruction(self, observation:str):
        instructions = [observation]
        for instruction in instructions:
            self.siri_memory.add_memory(instruction)
        print(f"instructions added successfully")
        return self.siri_memory.memory_retriever.memory_stream
    

    def ask_agent(self, message,t_response):
        USER_NAME = "Sam" 
        new_message = f"{USER_NAME} says {message}"
        return self.siri.generate_dialogue_response(new_message,t_response)


    def initalize_agent_memory(self, stream):
        self.siri_memory.memory_retriever.memory_stream = stream
        print(f"agent memory initalized with predefined memory stream")




