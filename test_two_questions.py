from langchain_core.messages import HumanMessage
from src.graph import build_graph

app = build_graph()

sessions = {}
session_id = "123"

def ask(user_msg):
    if session_id not in sessions:
        sessions[session_id] = []
    
    sessions[session_id].append(HumanMessage(content=user_msg))
    
    initial_state = {
        "messages": sessions[session_id],
        "triage_route": "",
        "retrieved_context": "",
        "final_response": ""
    }
    
    print(f"\n--- Asking: {user_msg} ---")
    try:
        result = app.invoke(initial_state)
        sessions[session_id] = result["messages"]
        print("Success! Final Response:", result["final_response"])
    except Exception as e:
        print("CRASHED:", e)

ask("What is end to end analytics?")
ask("How do I set up email tracking?")
