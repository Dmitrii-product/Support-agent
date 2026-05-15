import os
from dotenv import load_dotenv

# Load environment variables before any other imports that might depend on them
load_dotenv()

from src.graph import build_graph
from langchain_core.messages import HumanMessage

def main():
    app = build_graph()
    
    print("Welcome to the Smart Support Triage Agent! (Type 'exit' to quit)")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "triage_route": "",
            "retrieved_context": "",
            "final_response": ""
        }
        
        # Run the graph
        result = app.invoke(initial_state)
        
        print(f"Agent ({result.get('triage_route')} route): {result.get('final_response')}")

if __name__ == "__main__":
    main()
