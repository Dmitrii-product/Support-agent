import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def save_lead_tool(name: str, email: str, consent: bool) -> str:
    """Saves the customer's contact details (lead) to the database. Call this immediately when the customer shows intent to buy and provides their name, email, and consent."""
    return "Saved!"

@tool
def escalate_to_human_tool(name: str, email: str, consent: bool, issue: str) -> str:
    """Escalates a customer's unresolved issue to a human support agent by creating a ticket. Call this immediately when the customer provides their name, email, consent, and you have their issue description."""
    return "Ticket created"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools([save_lead_tool, escalate_to_human_tool], tool_choice="auto")

system_msg = (
    "You are a helpful customer support agent.\n"
    "1. LANGUAGE: You MUST respond in the exact same language that the customer used in their most recent message. Explicitly check the language of the user's query.\n"
    "2. LEAD CAPTURE: If the user shows intent to buy, ask for their name, email, and explicit consent to store their info. Once they provide ALL THREE, you MUST immediately call the `save_lead_tool`.\n"
    "3. ESCALATION: If you cannot resolve the issue or the user wants a human, ask for their name, email, and explicit consent. Once they provide ALL THREE, you MUST immediately call the `escalate_to_human_tool`.\n"
    "4. TOOL USAGE RULES:\n"
    "   - DO NOT call any tool until you have gathered ALL THREE pieces of info (name, email, consent).\n"
    "   - If data is missing, politely insist up to 2 times, then drop it.\n"
    "   - When you have the data, you MUST call the tool IMMEDIATELY. DO NOT generate any conversational text (like 'Thank you' or 'I will save this now') in the same message as the tool call. Output ONLY the tool call.\n"
    "   - You are FORBIDDEN from telling the user 'your data has been saved' or 'I created a ticket' UNLESS you see a ToolMessage in the conversation history confirming the database action."
)

sys_msg = SystemMessage(content=system_msg)

messages = [
    sys_msg,
    HumanMessage(content="Я хочу купить этот тариф."),
    AIMessage(content="Отлично! Для оформления покупки мне нужно сохранить ваши данные. Пожалуйста, предоставьте ваше имя, email и явное согласие на сохранение этих данных в нашей базе."),
    HumanMessage(content="Дима, dima@example.com, согласен.")
]

print("Invoking...")
res = llm_with_tools.invoke(messages)
print("Content:", res.content)
print("Tool calls:", res.tool_calls)
