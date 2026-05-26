import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def save_lead_tool(name: str, email: str, consent: bool) -> str:
    """Use this tool to save a lead when a customer shows intent to buy.
    You MUST ask for their name, email, and explicit consent to store their data BEFORE using this tool.
    """
    return "Saved!"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools([save_lead_tool])

sys_msg = SystemMessage(content="""You are a customer support agent.
If the user wants to buy and provides their name, email, and consent, you MUST call the `save_lead_tool`.
DO NOT reply conversationally without calling the tool first. Call the tool IMMEDIATELY.""")

msg = HumanMessage(content="I want to buy. My name is Dima, email dima@example.com, and I consent.")

print("Invoking...")
res = llm_with_tools.invoke([sys_msg, msg])
print("Content:", res.content)
print("Tool calls:", res.tool_calls)
