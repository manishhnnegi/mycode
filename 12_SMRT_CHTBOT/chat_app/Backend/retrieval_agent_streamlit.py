from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
from LLM.openai_server import Openai_LLM
from LLM.gemini_server import Google_LLM
import os
#from memory_agent import Memory_Agent
OPENAI = False
GOOGLE = True


# Define models
class Query(BaseModel):
    query: str
    num_tools: int
    from_openai: bool

# Initialize FastAPI instances
app_agent = FastAPI()


# Endpoint in Agent Server to receive queries and forward them to Tool Server
@app_agent.post("/send-query/")
async def send_query(query: Query):
    # Forward the query to Tool Server
    print("in agent--------------------------",query)

    print(type(query))
    # Access query and num_tools
    user_query = query.query
    num_tools = query.num_tools
    from_openai = query.from_openai
    
    # Process the data (You can replace this with your logic)
    processed_data = {
        "user_query": user_query,
        "num_tools": num_tools
    }


    response = requests.post("http://localhost:8001/process-query/", json=query.dict())

    print("------------------------------------", response)
    print("------------------------------------", response.json())
    
    tool_response = str(response.json()["response"])
    print("------------------------------------", tool_response)

    OPENAI_KEY = os.getenv("MY_KEY")
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = GEMINI_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY


    if from_openai:
        if OPENAI:
            cls =  Openai_LLM(OPENAI_KEY,tool_response, str(query.query))
        elif GOOGLE:
            cls =  Google_LLM(GEMINI_KEY,tool_response, str(query.query) )
        thought, final_answer = cls.run_agent()
        print("------------------------------------", final_answer)
       

        # Example JSON string
        json_string = {"tool_response": tool_response, "final_answer": final_answer}
        
    
    else:
        
        #pa = Memory_Agent(api_key=OPENAI_KEY, path_to_db= "Agent_db")
        pa = None
        try:
            lst = pa.fetch_db()
            if lst:
                pa.siri_memory.memory_retriever.memory_stream = lst
        except:
            print("no db yet")

        status_flag, final_answer  = pa.ask_agent(message=str(query.query),t_response=tool_response )

        pa.store_db()

        
        print("------------------------------------", final_answer)
     

        # Example JSON string
        json_string = {"tool_response": tool_response, "final_answer": final_answer}



    # Convert JSON string to Python dictionary
    json_object = json.dumps(json_string)

    # Print the JSON object
    print(json_object)

    
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error in Tool Server")
    

    return json_object


# Run the servers
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app_agent, host="0.0.0.0", port=8000)
