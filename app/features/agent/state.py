from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # just a pile of messages
    messages: Annotated[Sequence[BaseMessage], operator.add]
