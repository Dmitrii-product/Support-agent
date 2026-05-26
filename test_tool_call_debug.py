import os
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
from src.nodes.responder import responder_node
import src.nodes.responder

# Monkey-patch responder_node to print the exception
original_invoke = src.nodes.responder.llm_with_tools.invoke
def mock_invoke(*args, **kwargs):
    try:
        return original_invoke(*args, **kwargs)
    except Exception as e:
        print(f"DEBUG: Exception caught during invoke: {e}")
        if hasattr(e, "body"):
            print(f"DEBUG: e.body = {e.body}")
        raise e

src.nodes.responder.llm_with_tools.invoke = mock_invoke

state = {
    "messages": [
        HumanMessage(content="I want to buy a product. My name is Dima, email dima@example.com, and I consent.")
    ],
    "retrieved_context": "",
    "triage_route": "responder"
}

print("Running responder_node...")
try:
    result = responder_node(state)
    print("\n--- Final Messages ---")
    for msg in result["messages"]:
        print(f"{type(msg).__name__}: {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"  Tool Calls: {msg.tool_calls}")
except Exception as e:
    print("Crash:", e)

