import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from src.state import SupportState
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class RouteDecision(BaseModel):
    route: str = Field(description="The routing decision, either 'researcher' or 'responder'")

def triage_node(state: SupportState):
    """
    Analyzes the user's query and routes it to the appropriate node.
    """
    messages = state.get("messages", [])
    
    # Filter messages to prevent Groq API errors when tool calls from responder
    # are sent to the triage LLM which doesn't have those tools bound.
    safe_messages = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            safe_messages.append(AIMessage(content=msg.content))
        elif not isinstance(msg, ToolMessage):
            safe_messages.append(msg)
    
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "qwen-2.5-32b"),
        temperature=0
    )
    
    # Use structured output for routing
    router = llm.with_structured_output(RouteDecision)
    
    system_msg = """You are an expert support triage agent. 
Analyze the user's query and decide the next step.
Choose 'researcher' IF:
- The user asks ANY question about products, services, features, analytics, or technical details (even if you know the answer). We must retrieve company-specific docs.
- The user asks "how to" or "what is" regarding a feature.

Choose 'responder' IF:
- The user is making a simple greeting or thanking you.
- The user is expressing intent to buy or purchase something.
- The user asks about price, pricing, cost, or tariffs (e.g. "цена", "стоимость", "тариф").
- The user is providing their contact information (name, email) or consent.
- The user is asking to speak to a human.

CRITICAL: You are an internal router. DO NOT converse with the user. DO NOT generate conversational text. Output ONLY the valid JSON routing decision."""

    system_prompt = SystemMessage(content=system_msg)

    try:
        decision = router.invoke([system_prompt] + safe_messages)
        route = decision.route
    except Exception as e:
        print(f"Triage routing failed: {e}. Defaulting to 'responder'.")
        route = "responder"
    
    return {"triage_route": route}
