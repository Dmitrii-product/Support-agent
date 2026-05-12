import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
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
    
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "qwen-2.5-32b"),
        temperature=0
    )
    
    # Use structured output for routing
    router = llm.with_structured_output(RouteDecision)
    
    system_prompt = SystemMessage(content="""You are an expert support triage agent. 
Analyze the user's query and decide the next step.
If the query requires looking up external documentation or facts, choose 'researcher'.
If it is a simple greeting, thank you, or general question that doesn't need external docs, choose 'responder'.""")

    decision = router.invoke([system_prompt] + messages)
    
    return {"triage_route": decision.route}
