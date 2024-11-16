# backend/main.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from uuid import UUID, uuid4
from llmx import LLM  # Importing the LLM class we created

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
llms: Dict[UUID, LLM] = {}

# Message model
class Message(BaseModel):
    sender: str
    content: str

# Session-based Message Request model
class MessageRequest(BaseModel):
    session_id: UUID
    message: Message

# Generate a new session ID
@app.get("/session")
def create_session() -> UUID:
    session_id = uuid4()
    sessions_db[session_id] = []  # Initialize a new session history
    llms[session_id] = LLM()  # Initialize a new LLM instance for this session
    return session_id

# Endpoint to get messages for a specific session
@app.get("/messages/{session_id}", response_model=List[Message])
def get_messages(session_id: UUID):
    return sessions_db.get(session_id, [])

# Endpoint to send a message and get a response
@app.post("/messages", response_model=Message)
def send_message(request: MessageRequest):
    session_id = request.session_id
    user_message = request.message.content

    # Add the user's message to the session history
    sessions_db[session_id].append({"sender": request.message.sender, "content": user_message})

    # Retrieve the LLM instance for this session
    llm = llms[session_id]

    # Generate a response from the LLM
    response_content = llm.get_response(user_message)
    response_message = Message(sender="GPT", content=response_content)
    
    # Add the bot's response to the session history
    sessions_db[session_id].append({"sender": response_message.sender, "content": response_content})

    return response_message

# uvicorn main:app_agent --host 0.0.0.0 --port 8000 --reload


#uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#uvicorn backend.main:app --reload