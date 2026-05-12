from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage

class SupportState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    triage_route: str
    retrieved_context: str
    final_response: str
