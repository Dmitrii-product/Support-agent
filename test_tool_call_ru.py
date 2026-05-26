import os
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
from src.nodes.responder import responder_node

state = {
    "messages": [
        HumanMessage(content="Я хочу купить. Меня зовут Дима, почта dima@example.com, я согласен.")
    ],
    "retrieved_context": "",
    "triage_route": "responder"
}

print("Running responder_node in Russian...")
try:
    result = responder_node(state)
    print("\n--- Final Messages ---")
    for msg in result["messages"]:
        print(f"{type(msg).__name__}: {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"  Tool Calls: {msg.tool_calls}")
    print("\n--- Final Response ---")
    print(result.get("final_response"))
except Exception as e:
    print("Crash:", e)

