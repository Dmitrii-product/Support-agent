import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage
from src.state import SupportState
from dotenv import load_dotenv

load_dotenv()

def responder_node(state: SupportState):
    """
    Generates the final response using the user's query and retrieved context.
    """
    messages = state.get("messages", [])
    context = state.get("retrieved_context", "")
    
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "qwen-2.5-32b"),
        temperature=0.7
    )
    
    system_msg = "You are a helpful customer support agent."
    if context:
        system_msg += f"\n\nUse the following retrieved context to help answer the user's question:\n{context}"
        
    response = llm.invoke([SystemMessage(content=system_msg)] + messages)
    
    return {"final_response": response.content, "messages": [response]}
