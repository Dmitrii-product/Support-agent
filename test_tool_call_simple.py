import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def save_lead_tool(name: str, email: str, consent: bool) -> str:
    """Saves the customer's contact details to the database. Call this immediately when the customer provides their name, email, and consent."""
    return "Saved!"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools([save_lead_tool])

sys_msg = SystemMessage(content="You are a support agent. If the user provides name, email, and consent, call save_lead_tool immediately.")

msg = HumanMessage(content="Я хочу купить. Меня зовут Дима, почта dima@example.com, я согласен.")

print("Invoking...")
res = llm_with_tools.invoke([sys_msg, msg])
print("Content:", res.content)
print("Tool calls:", res.tool_calls)
