from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import json
from LLM.openai_server import Openai_LLM
from LLM.gemini_server import Google_LLM

# Set flags
OPENAI = False
GOOGLE = True

# Initialize FastAPI instance
app_agent = FastAPI()

# Enable CORS
app_agent.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domain in production, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, including POST, GET, OPTIONS, etc.
    allow_headers=["*"],  # Allows all headers
)

# Define models
class Query(BaseModel):
    query: str
    num_tools: int
    from_openai: bool

@app_agent.post("/send-query/")
async def send_query(query: Query):
    # Access query and num_tools
    user_query = query.query
    num_tools = query.num_tools
    from_openai = query.from_openai
    
    # Forward the query to Tool Server
    try:
        response = requests.post("http://localhost:8001/process-query/", json=query.dict())
        response.raise_for_status()  # Raises an HTTPError if the status is 4xx, 5xx
        tool_response = response.json().get("response", "No response from tool server")
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error contacting Tool Server: {e}")
    
    # Set up OpenAI or Google LLM
    OPENAI_KEY = os.getenv("MY_KEY")
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = GEMINI_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

    if not from_openai:
        if OPENAI:
            cls = Openai_LLM(OPENAI_KEY, tool_response, str(query.query))
        elif GOOGLE:
            cls = Google_LLM(GEMINI_KEY, tool_response, str(query.query))
        thought, final_answer = cls.run_agent()
    else:
        # Handle the case when from_openai is True
        pa = None  # Replace with actual Memory_Agent if used
        try:
            if pa:
                pa.store_db()
            status_flag, final_answer = pa.ask_agent(message=str(query.query), t_response=tool_response)
        except Exception as e:
            final_answer = "Failed to process with memory agent"

    # Return as JSON response (not as string)
    return {
        "tool_response": tool_response,
        "final_answer": final_answer
    }

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app_agent, host="0.0.0.0", port=8000)
