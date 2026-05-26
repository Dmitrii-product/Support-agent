import os
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
from src.nodes.responder import responder_node

state = {
    "messages": [
        HumanMessage(content="Я хочу купить этот тариф."),
        AIMessage(content="Отлично! Для оформления покупки мне нужно сохранить ваши данные. Пожалуйста, предоставьте ваше имя, email и явное согласие на сохранение этих данных в нашей базе."),
        HumanMessage(content="Дима, dima@example.com, согласен.")
    ],
    "retrieved_context": "",
    "triage_route": "responder"
}

print("Running responder_node in multi-turn...")
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

