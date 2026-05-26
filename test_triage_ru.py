import os
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
from src.nodes.triage import triage_node

state = {
    "messages": [
        HumanMessage(content="что такое сквозная аналитика?")
    ]
}

print("Running triage_node...")
try:
    result = triage_node(state)
    print("Result:", result)
except Exception as e:
    print("Crash:", e)

