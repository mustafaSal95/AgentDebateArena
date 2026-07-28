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

# Clients are cached per temperature so repeated turns reuse the same
# ChatGroq/Gemini instance instead of paying client-construction overhead
# on every single node call. Small win on its own, but it adds up over a
# multi-turn debate.
_llm_cache: dict[float, "ChatGroq"] = {}
_gemini_cache: dict[float, object] = {}


def _get_llm(temperature: float = 0.7):
    """
    Cached ChatGroq client for this temperature, or the deterministic
    offline stand-in if config.is_offline is True (no GROQ_API_KEY set).
    Bounded by config.agent_timeout_seconds so a hung call fails fast
    instead of blocking the whole debate.
    """
    from config import config
    if config.is_offline:
        from offline_model import OfflineChatModel
        return OfflineChatModel()
    if temperature not in _llm_cache:
        _llm_cache[temperature] = ChatGroq(
            model=config.model_name,
            temperature=temperature,
            api_key=config.groq_api_key,
            timeout=config.agent_timeout_seconds,
        )
    return _llm_cache[temperature]


def _get_gemini_llm(temperature: float = 0.7):
    """
    Cached Gemini LLM client if a GEMINI_API_KEY is configured, else None.
    Same timeout bound as the Groq client.
    """
    from config import config
    if not config.has_gemini:
        return None
    if temperature not in _gemini_cache:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _gemini_cache[temperature] = ChatGoogleGenerativeAI(
            model=config.gemini_model,
            temperature=temperature,
            google_api_key=config.gemini_api_key,
            timeout=config.agent_timeout_seconds,
        )
    return _gemini_cache[temperature]


def _fallback_invoke(messages, agent_id: str, bind_tools: bool = False):
    """
    Fallback chain: Gemini → Offline (with db_search facts).
    Called when Groq rate limits are hit.
    """
    gemini = _get_gemini_llm()
    if gemini:
        try:
            if bind_tools:
                llm = gemini.bind_tools(TOOLS)
            else:
                llm = gemini
            print(f"\n[info] Falling back to Gemini for {agent_id}.")
            return llm.invoke(messages)
        except Exception as e:
            print(f"[warning] Gemini fallback also failed ({e}); using offline model.")

    print(f"\n[info] Using offline model with local knowledge base for {agent_id}.")
    from offline_model import OfflineChatModel
    return OfflineChatModel().invoke(messages)


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
    Single LLM call per pass, with tools bound. Two possible outcomes:
      - Model emits tool_calls -> routed to ToolNode by tools_condition, then
        this node is called again with the tool results in state["messages"].
      - Model emits a direct response with no tool_calls -> that response IS
        the final argument for this turn (no separate formulate_argument
        pass). turns_left_per_agent is decremented here in that case.
    This collapses what used to be two sequential LLM calls per turn
    (agent_llm_action then formulate_argument) into one in the common case
    where the agent doesn't need to research.
    IMPORTANT: response.name is set to agent_id so messages stay attributable.
    """
    agent_id = state["current_turn"]
    system = SystemMessage(content=get_persona_prompt(agent_id, state))
    full_messages = [system, *state["messages"]]

    # Prompt the LLM explicitly so it operates in assistant mode rather than
    # continuation mode, and so a no-tool-call response is already the final,
    # ready-to-score argument rather than a rough draft needing a second pass.
    full_messages.append(HumanMessage(content=(
        "It is your turn. If you need evidence, call a tool. Otherwise, skip "
        "tools entirely and respond now with your final, plain-text persuasive "
        "argument for this turn — concise, in character, under 100 words. "
        "Do not output tool calls or `<function=...>` tags unless you are "
        "actually invoking a tool."
    )))

    try:
        llm = _get_llm().bind_tools(TOOLS)
        response = llm.invoke(full_messages)
    except Exception as e:
        if "rate_limit_exceeded" in str(e) or "Rate limit reached" in str(e):
            response = _fallback_invoke(full_messages, agent_id, bind_tools=True)
        else:
            import re, json
            from langchain_core.messages import AIMessage
            error_str = str(e)
            if "tool_use_failed" in error_str and "failed_generation" in error_str:
                match = re.search(r"'failed_generation': '(.*?)'", error_str)
                if match:
                    failed_gen = match.group(1)
                    # parse <function=name>args</function>
                    fn_match = re.search(r"<function=([a-zA-Z0-9_]+)[=>]*(.*?)</function>", failed_gen)
                    if fn_match:
                        name = fn_match.group(1)
                        args_str = fn_match.group(2)
                        if args_str.startswith(">"):
                            args_str = args_str[1:]
                        try:
                            args = json.loads(args_str)
                            print(f"\n[info] Recovered malformed tool call for {name}")
                            tool_call_id = f"call_{name}_{len(full_messages)}"
                            response = AIMessage(
                                content="",
                                tool_calls=[{
                                    "name": name,
                                    "args": args,
                                    "id": tool_call_id,
                                    "type": "tool_call"
                                }]
                            )
                            response.name = agent_id
                            return {"messages": [response]}
                        except Exception:
                            pass

            print(f"[warning] tool-calling failed for {agent_id} ({e}); retrying without tools")
            try:
                llm = _get_llm()
                response = llm.invoke(full_messages)
            except Exception as inner_e:
                if "rate_limit_exceeded" in str(inner_e) or "Rate limit reached" in str(inner_e):
                    response = _fallback_invoke(full_messages, agent_id)
                else:
                    raise inner_e

    response.name = agent_id

    if getattr(response, "tool_calls", None):
        # Still researching — route to ToolNode, come back here after.
        return {"messages": [response]}

    # No tool call: this response is the final argument for the turn, so
    # this is the one place turn-count bookkeeping happens (formerly done
    # in a separate formulate_argument node after a second LLM call).
    turns_left = dict(state["turns_left_per_agent"])
    if agent_id in turns_left:
        turns_left[agent_id] = max(0, turns_left[agent_id] - 1)

    return {
        "messages": [response],
        "turns_left_per_agent": turns_left,
    }


def route_after_agent_action(state: DebateState) -> str:
    """Wrapper around tools_condition so it reads cleanly in add_conditional_edges."""
    return tools_condition(state)


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
    Renders this turn via ui.render_turn: message panel, tool-call table,
    live scoreboard. Pure side effect — returns no state changes.
    """
    from ui import render_turn
    render_turn(state)
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