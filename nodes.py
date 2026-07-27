"""
Node functions for the DebateArena graph.

Depends on modules not shown here — adjust imports to match your actual layout:
  - config.py        -> `config` object (config.model_name, config.temperature, config.groq_api_key)
  - tools.py          -> `web_search`, `db_search` tool functions
  - personas.py        -> `get_persona_prompt(agent_id)`, `PERSONAS`/agent list

Each node takes the full DebateState and returns only the keys it's updating —
LangGraph merges that partial dict into state using each field's reducer.
"""
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END
from langgraph.prebuilt import tools_condition

from state import DebateState
from tools import web_search, db_search
from personas import get_persona_prompt

TOOLS = [web_search, db_search]


def _get_llm(temperature: float = 0.7):
    """Fresh ChatGroq client. Swap model_name/temperature via config if needed."""
    from config import config
    return ChatGroq(
        model=config.model_name,
        temperature=temperature,
        api_key=config.groq_api_key,
    )


def select_agent(state: DebateState) -> dict:
    """
    Picks the next agent to speak. Simple round-robin over agent_ids,
    skipping anyone with 0 turns left. If moderator_target is set, honor it
    for this turn instead (then clear it — it's a one-shot override).
    """
    agent_ids = state["agent_ids"]
    turns_left = state["turns_left_per_agent"]

    if state.get("moderator_target"):
        next_agent = state["moderator_target"]
        return {
            "previous_turn": state.get("current_turn"),
            "current_turn": next_agent,
            "moderator_target": None,
        }

    prev = state.get("current_turn")
    if prev is None:
        candidates = [a for a in agent_ids if turns_left.get(a, 0) > 0]
        next_agent = candidates[0] if candidates else agent_ids[0]
    else:
        start = agent_ids.index(prev) if prev in agent_ids else -1
        ordered = agent_ids[start + 1:] + agent_ids[:start + 1]
        candidates = [a for a in ordered if turns_left.get(a, 0) > 0]
        next_agent = candidates[0] if candidates else prev

    return {
        "previous_turn": prev,
        "current_turn": next_agent,
    }


def agent_llm_action(state: DebateState) -> dict:
    """
    LLM call with tools bound. May emit tool_calls (routed to ToolNode by
    tools_condition) or a direct response with no tool calls.
    IMPORTANT: response.name is set to agent_id so messages stay attributable.
    """
    agent_id = state["current_turn"]
    llm = _get_llm().bind_tools(TOOLS)

    system = SystemMessage(content=get_persona_prompt(agent_id, state))
    response = llm.invoke([system, *state["messages"]])
    response.name = agent_id

    return {"messages": [response]}


def route_after_agent_action(state: DebateState) -> str:
    """Wrapper around tools_condition so it reads cleanly in add_conditional_edges."""
    return tools_condition(state)


def formulate_argument(state: DebateState) -> dict:
    """
    Second LLM pass: takes whatever came back from agent_llm_action / tool
    results and writes the actual persuasive argument for this turn.
    """
    agent_id = state["current_turn"]
    llm = _get_llm(temperature=0.8)

    prompt = (
        f"{get_persona_prompt(agent_id, state)}\n\n"
        "Using the research above (if any), write your argument for this turn. "
        "Be persuasive, concise, and directly address the opponent's last point if there is one."
    )
    response = llm.invoke([*state["messages"], HumanMessage(content=prompt)])
    response.name = agent_id

    turns_left = dict(state["turns_left_per_agent"])
    if agent_id in turns_left:
        turns_left[agent_id] = max(0, turns_left[agent_id] - 1)

    return {
        "messages": [response],
        "turns_left_per_agent": turns_left,
    }


def score_turn(state: DebateState) -> dict:
    """
    Rule-based scoring, not another LLM call. +1 tool used, +1 has numbers,
    +1 rebuts opponent, -1 if rambling. Updates cumulative_scores + scores_per_turn.
    """
    agent_id = state["current_turn"]
    last_msg = state["messages"][-1]
    content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

    delta = 0.0
    reasons = []

    used_tool = any(
        getattr(m, "name", None) == agent_id and getattr(m, "tool_calls", None)
        for m in state["messages"][-5:]
    )
    if used_tool:
        delta += 1
        reasons.append("used a tool")

    if any(ch.isdigit() for ch in content):
        delta += 1
        reasons.append("cited concrete numbers")

    prior_opponent_msgs = [
        m for m in state["messages"][:-1]
        if getattr(m, "name", None) not in (agent_id, None)
    ]
    if prior_opponent_msgs:
        last_opp = prior_opponent_msgs[-1].content
        opp_words = set(str(last_opp).lower().split())
        overlap = opp_words & set(content.lower().split())
        if len(overlap) >= 3:
            delta += 1
            reasons.append("directly engaged opponent's last point")

    if len(content.split()) > 180:
        delta -= 1
        reasons.append("too long/rambling")

    cumulative = dict(state["cumulative_scores"])
    cumulative[agent_id] = cumulative.get(agent_id, 0) + delta

    turn_number = len(state["scores_per_turn"]) + 1
    score_entry = {
        "turn": turn_number,
        "scores": {agent_id: delta},
        "feedback": "; ".join(reasons) if reasons else "no notable signals",
    }

    return {
        "cumulative_scores": cumulative,
        "scores_per_turn": [score_entry],
    }


def display(state: DebateState) -> dict:
    """
    Render this turn: last message, tool calls used, live scoreboard.
    Pure side effect (printing) — returns no state changes.
    Swap the print calls for your `rich`-based UI later.
    """
    agent_id = state["current_turn"]
    last_msg = state["messages"][-1]
    last_score = state["scores_per_turn"][-1]

    print(f"\n=== Turn {last_score['turn']} — {agent_id} ===")
    print(last_msg.content)
    print(f"Score delta: {last_score['scores'][agent_id]:+.0f} ({last_score['feedback']})")
    print(f"Cumulative: {state['cumulative_scores']}")

    return {}


def handle_interrupt(state: DebateState) -> dict:
    """
    Parses moderator_message (set elsewhere, e.g. by a UI/input loop) into a
    moderator_target agent name if the message addresses someone by name.
    Clears moderator_interrupted once handled.
    """
    message = state.get("moderator_message") or ""
    target: Optional[str] = None

    lowered = message.lower()
    if lowered.strip() in ("exit", "quit", "stop"):
        return {
            "exit_interrupt": True,
            "moderator_interrupted": False,
            "moderator_message": None,
        }

    for agent_id in state["agent_ids"]:
        if agent_id.lower() in lowered:
            target = agent_id
            break

    return {
        "moderator_interrupted": False,
        "moderator_target": target,
        "messages": [HumanMessage(content=message, name="moderator")] if message else [],
    }


def route_after_display(state: DebateState) -> str:
    """
    Decides what happens after a turn is displayed:
    exit requested -> end; moderator interrupted -> handle it; else keep going
    until every agent is out of turns.
    """
    if state.get("exit_interrupt"):
        return "end"
    if state.get("moderator_interrupted"):
        return "interrupt"
    if all(n <= 0 for n in state["turns_left_per_agent"].values()):
        return "end"
    return "continue"