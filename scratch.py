import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

load_dotenv()

@tool
def dummy_tool(text: str) -> str:
    """Dummy tool"""
    return text

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
llm_with_tools = llm.bind_tools([dummy_tool])

try:
    response = llm_with_tools.invoke([SystemMessage(content="You must only use the tool if the user says 'USE TOOL'. Otherwise, just answer normally. Explain end-to-end analytics in Russian."), HumanMessage(content="What is end to end analytics? do not use tools.")])
    print("Success:", response)
except Exception as e:
    print("Caught exception:", type(e))
    print("Dir:", dir(e))
    if hasattr(e, 'body'):
        print("Body:", e.body)
    if hasattr(e, 'response'):
        print("Response:", e.response)
