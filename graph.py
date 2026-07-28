"""
Graph assembly for DebateArena.

Adds one small thing not in nodes.py: a terminal input node that actually
captures moderator interrupts between turns (nodes.py's handle_interrupt
only *parses* moderator_message — something has to set it first). Kept here
rather than in nodes.py since it's UI/IO, not debate logic.
"""
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state import DebateState
from tools import web_search, db_search
from nodes import (
    select_agent,
    agent_llm_action,
    route_after_agent_action,
    score_turn,
    display,
    handle_interrupt,
    route_after_display,
)


def check_human_interrupt(state: DebateState) -> dict:
    """
    Prompts for optional moderator input after each turn's display.
    Blank/Enter -> no interrupt, debate continues.
    Anything else -> stored as moderator_message for handle_interrupt to parse.
    """
    raw = input(
        "\n[moderator] Press Enter to continue, address an agent, "
        "or type 'exit' to stop: "
    ).strip()

    if not raw:
        return {"moderator_interrupted": False, "moderator_message": None}

    return {"moderator_interrupted": True, "moderator_message": raw}


def route_after_interrupt_check(state: DebateState) -> str:
    if state.get("moderator_interrupted"):
        return "interrupt"
    if all(n <= 0 for n in state["turns_left_per_agent"].values()):
        return "end"
    return "continue"


def build_graph():
    graph = StateGraph(DebateState)

    graph.add_node("select_agent", select_agent)
    graph.add_node("agent_llm_action", agent_llm_action)
    graph.add_node("tools", ToolNode([web_search, db_search]))
    graph.add_node("score_turn", score_turn)
    graph.add_node("display", display)
    graph.add_node("check_human_interrupt", check_human_interrupt)
    graph.add_node("handle_interrupt", handle_interrupt)

    graph.set_entry_point("select_agent")
    graph.add_edge("select_agent", "agent_llm_action")

    # agent_llm_action now produces the final argument itself once it stops
    # calling tools, so a no-tool-calls response goes straight to scoring —
    # no separate formulate_argument LLM pass.
    graph.add_conditional_edges(
        "agent_llm_action",
        route_after_agent_action,
        {"tools": "tools", END: "score_turn"},
    )
    graph.add_edge("tools", "agent_llm_action")

    graph.add_edge("score_turn", "display")
    graph.add_edge("display", "check_human_interrupt")

    graph.add_conditional_edges(
        "check_human_interrupt",
        route_after_interrupt_check,
        {"interrupt": "handle_interrupt", "continue": "select_agent", "end": END},
    )

    # Fix: exit ("exit"/"quit"/"stop") stops immediately instead of running
    # one more full turn before the exit check is seen again.
    graph.add_conditional_edges(
        "handle_interrupt",
        lambda state: "end" if state.get("exit_interrupt") else "continue",
        {"end": END, "continue": "select_agent"},
    )

    return graph.compile()