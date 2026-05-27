from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langchain_core.messages import HumanMessage, AIMessage

# Import our LangGraph app
from src.graph import build_graph

app = FastAPI(title="Smart Support Triage API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
import os
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Build the LangGraph once
graph_app = build_graph()

# In-memory session storage (for demo purposes)
sessions = {}

import random

class ChatRequest(BaseModel):
    session_id: str
    message: str
    triage_model: str = None  # Optional override

class ChatResponse(BaseModel):
    response: str
    triage_model: str = None
    triage_route: str = None

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id
    user_msg = req.message

    if session_id not in sessions:
        # Assign model on session creation: either use provided override or random split
        assigned_model = req.triage_model if req.triage_model else random.choice(["llama", "qwen"])
        sessions[session_id] = {
            "messages": [],
            "triage_model": assigned_model
        }
    
    # We append the new human message to the session's history
    sessions[session_id]["messages"].append(HumanMessage(content=user_msg))

    initial_state = {
        "messages": sessions[session_id]["messages"],
        "triage_route": "",
        "triage_model": sessions[session_id]["triage_model"],
        "retrieved_context": "",
        "final_response": ""
    }
    
    # Run the graph
    result = graph_app.invoke(initial_state)
    
    # Update session history
    sessions[session_id]["messages"] = result["messages"]

    return ChatResponse(
        response=result["final_response"],
        triage_model=result.get("triage_model", ""),
        triage_route=result.get("triage_route", "")
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
