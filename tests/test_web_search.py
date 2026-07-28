import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from nodes import _get_llm, TOOLS
from tools import db_search

def test():
    llm = _get_llm().bind_tools(TOOLS)
    messages = [
        SystemMessage(content="You are a debater. Use web_search if you can't find what you need locally."),
        AIMessage(content="", tool_calls=[{'name': 'db_search', 'args': {'query': 'unrelated topic'}, 'id': 'call_1'}]),
        ToolMessage(content="No local documents matched 'unrelated topic'.", name="db_search", tool_call_id="call_1"),
        HumanMessage(content="It is your turn. You can use tools to research, or provide your response if you don't need tools.")
    ]
    
    print("Invoking LLM after a failed db_search...")
    response = llm.invoke(messages)
    
    print("Response tool calls:", getattr(response, "tool_calls", None))
    print("Response content:", response.content)

if __name__ == "__main__":
    test()
