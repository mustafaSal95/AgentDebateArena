import os
from dotenv import load_dotenv
load_dotenv()
from config import config
from nodes import _get_llm, TOOLS
from langchain_core.messages import SystemMessage, HumanMessage

def test():
    print(f"Offline? {config.is_offline}")
    llm = _get_llm().bind_tools(TOOLS)
    messages = [
        SystemMessage(content="You are a helpful assistant. Please search the web for the latest news on AI.")
    ]
    print("Invoking LLM...")
    response = llm.invoke(messages)
    print("Response tool calls:")
    print(getattr(response, "tool_calls", None))
    print("Response content:")
    print(response.content)

if __name__ == "__main__":
    test()
