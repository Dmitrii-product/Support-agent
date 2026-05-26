import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from src.state import SupportState
from src.database import save_lead, create_support_ticket
from dotenv import load_dotenv

load_dotenv()

@tool
def save_lead_tool(name: str, email: str, consent: bool) -> str:
    """Saves the customer's contact details (lead) to the database. Call this immediately when the customer shows intent to buy and provides their name, email, and consent."""
    return save_lead(name, email, consent)

@tool
def escalate_to_human_tool(name: str, email: str, consent: bool, issue: str) -> str:
    """Escalates a customer's unresolved issue to a human support agent by creating a ticket. Call this immediately when the customer provides their name, email, consent, and you have their issue description."""
    return create_support_ticket(name, email, consent, issue)

def responder_node(state: SupportState):
    """
    Generates the final response using the user's query and retrieved context.
    Handles lead capture via tool calling.
    """
    messages = state.get("messages", [])
    context = state.get("retrieved_context", "")
    
    # Using llama-3.3-70b-versatile for robust tool calling support
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL_RESPONDER", "llama-3.3-70b-versatile"),
        temperature=0.7
    )
    llm_with_tools = llm.bind_tools([save_lead_tool, escalate_to_human_tool])
    
    system_msg = (
        "You are a helpful customer support agent.\n"
        "1. LANGUAGE: You MUST respond in the exact same language that the customer used in their most recent message. Explicitly check the language of the user's query.\n"
        "2. LEAD CAPTURE: If the user shows intent to buy, you should eventually capture them as a lead. However, see the TIMING RULE below.\n"
        "3. ESCALATION: If you cannot resolve the issue or the user wants a human, ask for their name, email, and explicit consent. Once they provide ALL THREE, you MUST immediately call the `escalate_to_human_tool`.\n"
        "4. TIMING RULE: NEVER ask for the user's personal data (name, email, consent) in your very first conversational turn with them, UNLESS they explicitly request to contact a human. Always answer their initial question first.\n"
        "5. TOOL USAGE RULES:\n"
        "   - DO NOT call any tool until you have gathered ALL THREE pieces of info (name, email, consent).\n"
        "   - If data is missing, politely insist up to 2 times, then drop it.\n"
        "   - When you have the data, you MUST call the tool IMMEDIATELY. DO NOT generate any conversational text (like 'Thank you' or 'I will save this now') in the same message as the tool call. Output ONLY the tool call.\n"
        "   - You are FORBIDDEN from telling the user 'your data has been saved' or 'I created a ticket' UNLESS you see a ToolMessage in the conversation history confirming the database action.\n"
        "   - Once you see a successful ToolMessage confirming that either `escalate_to_human_tool` or `save_lead_tool` executed, you MUST tell the user that their request was successfully stored and that a manager will contact them within 1 business day.\n"
        "6. PRICING: If the user asks about price, pricing, cost, or tariffs, ALWAYS refer them to https://www.mango-office.ru/products/calltracking/price/ and highlight that the exact price depends on the tariff. Per the TIMING RULE, do not ask for their data in the first turn.\n"
        "7. NO HALLUCINATION: NEVER invent answers about MANGO OFFICE products. If the user asks a question and the information is NOT found in the retrieved context provided below, you MUST simply say that you do not know. Then, suggest creating a support ticket or lead so a human can help, and ask for their name, email, and explicit consent (keeping the TIMING RULE in mind if it's the first turn)."
    )
    if context:
        system_msg += f"\n\nUse the following retrieved context to help answer the user's question:\n{context}"
        
    current_messages = [SystemMessage(content=system_msg)] + messages
    
    try:
        response = llm_with_tools.invoke(current_messages)
    except Exception as e:
        # Handle Groq's strict tool calling error when it generates conversational text instead
        failed_gen = None
        if hasattr(e, "body") and isinstance(e.body, dict):
            failed_gen = e.body.get("error", {}).get("failed_generation")
        elif hasattr(e, "response"):
            try:
                failed_gen = e.response.json().get("error", {}).get("failed_generation")
            except Exception:
                pass
        
        if failed_gen:
            print(f"DEBUG: Groq API Error caught. failed_gen: {failed_gen}")
            from langchain_core.messages import AIMessage
            response = AIMessage(content=failed_gen)
        else:
            print(f"DEBUG: Unhandled exception: {e}")
            raise e
    
    # Handle tool calls internally in the node
    if hasattr(response, "tool_calls") and response.tool_calls:
        current_messages.append(response)
        for tool_call in response.tool_calls:
            if tool_call["name"] == "save_lead_tool":
                args = tool_call["args"]
                try:
                    result = save_lead_tool.invoke(args)
                except Exception as e:
                    # Catch pydantic validation errors (missing fields)
                    result = f"Error: Failed to save lead because some data is missing or invalid. Please ask the user for the missing data (name, email, or consent). System error: {str(e)}"
            elif tool_call["name"] == "escalate_to_human_tool":
                args = tool_call["args"]
                try:
                    result = escalate_to_human_tool.invoke(args)
                except Exception as e:
                    result = f"Error: Failed to create ticket because some data is missing or invalid. Please ask the user for the missing data (name, email, consent). System error: {str(e)}"
            else:
                result = f"Error: Tool '{tool_call['name']}' not found."
            
            tool_msg = ToolMessage(content=result, tool_call_id=tool_call["id"])
            current_messages.append(tool_msg)
        
        # Invoke again to get the final text response after tool execution
        try:
            response = llm_with_tools.invoke(current_messages)
        except Exception as e:
            failed_gen = None
            if hasattr(e, "body") and isinstance(e.body, dict):
                failed_gen = e.body.get("error", {}).get("failed_generation")
            elif hasattr(e, "response"):
                try:
                    failed_gen = e.response.json().get("error", {}).get("failed_generation")
                except Exception:
                    pass
            
            if failed_gen:
                from langchain_core.messages import AIMessage
                response = AIMessage(content=failed_gen)
            else:
                raise e
    
    return {"final_response": response.content, "messages": [response]}
