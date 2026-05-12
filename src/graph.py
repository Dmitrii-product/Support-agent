from langgraph.graph import StateGraph, END
from src.state import SupportState
from src.nodes.triage import triage_node
from src.nodes.researcher import researcher_node
from src.nodes.responder import responder_node

def route_request(state: SupportState):
    """
    Routing function based on the triage_route in state.
    """
    route = state.get("triage_route", "responder")
    if route == "researcher":
        return "researcher"
    return "responder"

def build_graph():
    # Initialize the graph
    workflow = StateGraph(SupportState)

    # Add nodes
    workflow.add_node("triage", triage_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("responder", responder_node)

    # Set entry point
    workflow.set_entry_point("triage")

    # Add conditional edges from triage
    workflow.add_conditional_edges(
        "triage",
        route_request,
        {
            "researcher": "researcher",
            "responder": "responder"
        }
    )

    # Add normal edges
    workflow.add_edge("researcher", "responder")
    workflow.add_edge("responder", END)

    # Compile
    app = workflow.compile()
    return app
