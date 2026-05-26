import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
from src.graph import build_graph

app = build_graph()

state = {
    "messages": [
        HumanMessage(content="I want to buy a product"),
        AIMessage(content="Please provide your name, email, and explicit consent to store your information in our database."),
        HumanMessage(content="My name is John, email john@example.com, and I consent.")
    ],
    "triage_route": "",
    "retrieved_context": "",
    "final_response": ""
}

print("Invoking turn 1 (PII)...")
result1 = app.invoke(state)
print("Result 1:", result1.get("final_response"))

# Turn 2
state2 = {
    "messages": result1["messages"] + [HumanMessage(content="What is end to end analytics?")],
    "triage_route": "",
    "retrieved_context": "",
    "final_response": ""
}

print("\nInvoking turn 2 (New question)...")
result2 = app.invoke(state2)
print("Result 2:", result2.get("final_response"))

