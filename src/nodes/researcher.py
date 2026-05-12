from src.state import SupportState
from src.vectorstore import get_retriever

def researcher_node(state: SupportState):
    """
    Retrieves relevant context from ChromaDB based on the user's query.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"retrieved_context": ""}
        
    last_query = messages[-1].content
    
    retriever = get_retriever()
    docs = retriever.invoke(last_query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    return {"retrieved_context": context}
