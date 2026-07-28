from tools import web_search, db_search
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolCall

def test():
    node = ToolNode([web_search, db_search])
    msg = AIMessage(
        content="", 
        tool_calls=[ToolCall(name="db_search", args={"query": "test"}, id="call_1")]
    )
    result = node.invoke({"messages": [msg]})
    print("ToolNode result:")
    print(result)

if __name__ == "__main__":
    test()
