from typing import Literal
from app.core.config import get_settings
from app.features.agent.state import AgentState
from app.features.agent.tools import agent_tools
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import Tool


settings = get_settings()

# spin up the brain
model = ChatOpenAI(
    model=settings.OPENAI_MODEL_GPT4,
    api_key=settings.OPENAI_API_KEY,
    temperature=0
)

# give it the tools
model_with_tools = model.bind_tools(agent_tools)

# the thinking part
def agent(state: AgentState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# the doing part (manual cuz version mismatch sucks)
async def tools(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # lookup table for tools
    tool_map = {tool.name: tool for tool in agent_tools}
    
    responses = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name in tool_map:
            tool = tool_map[tool_name]
            # execute or die trying
            try:
                # async power
                tool_output = await tool.ainvoke(tool_args)
            except Exception as e:
                tool_output = f"Error executing tool {tool_name}: {str(e)}"
            
            responses.append(
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=str(tool_output),
                    name=tool_name
                )
            )
    
    return {"messages": responses}

# decider: keep going or shut up?
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    # tool calls? go to tools. otherwise bail.
    if last_message.tool_calls:
        return "tools"
    return "__end__"

# build the maze
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent)
workflow.add_node("tools", tools)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
)

workflow.add_edge("tools", "agent")

# compile it
app = workflow.compile()
