from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from uuid import UUID, uuid4
import requests
import os
from LLM.openai_server import Openai_LLM
from LLM.gemini_server import Google_LLM

app = FastAPI()

# Allow CORS for specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this according to your React app's origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Allow necessary HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# In-memory storage for message history per session
sessions_db: Dict[UUID, List[Dict[str, str]]] = {}
# Store the LLM instances by session ID
llms: Dict[UUID, object] = {}

# Message model
class Message(BaseModel):
    sender: str
    content: str

# Session-based Message Request model
class MessageRequest(BaseModel):
    session_id: UUID
    message: Message

# Generate a new session ID
@app.get("/session", response_model=UUID)
def create_session() -> UUID:
    session_id = uuid4()
    sessions_db[session_id] = []  # Initialize a new session history

    # Set up the Google LLM
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = GEMINI_KEY
    llms[session_id] = Google_LLM(GEMINI_KEY)

    return session_id

# Endpoint to get messages for a specific session
@app.get("/messages/{session_id}", response_model=List[Message])
def get_messages(session_id: UUID):
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions_db[session_id]

# Endpoint to send a message and get a response
@app.post("/messages", response_model=Message)
def send_message(request: MessageRequest):
    session_id = request.session_id
    user_message = request.message.content

    # Check if session exists
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")

    # Add the user's message to the session history
    sessions_db[session_id].append({"sender": request.message.sender, "content": user_message})

    inp_dic = {
        "query": user_message,
        "num_tools": 2,
        "from_openai": False,
    }

    # Forward the query to the Tool Server
    try:
        response = requests.post("http://localhost:8001/process-query/", json=inp_dic)
        response.raise_for_status()  # Raises an HTTPError if the status is 4xx, 5xx
        tool_response = response.json().get("response", "No response from tool server")
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error contacting Tool Server: {e}")

    # Retrieve the LLM instance for this session
    llm = llms[session_id]

    # Generate the response from the LLM
    thought, response_content = llm.run_agent(tool_response, user_message)

    print('------------------------', response_content)

    # Create a message response for the LLM
    response_message = Message(sender="GPT", content=response_content)

    # Add the bot's response to the session history
    sessions_db[session_id].append({"sender": response_message.sender, "content": response_content})

    # Return both the user's message and the bot's response for real-time updates
    return response_message

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
