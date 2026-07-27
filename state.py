import operator
from typing import Annotated, Dict, List, Optional, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_dicts(left: Optional[Dict], right: Optional[Dict]) -> Dict:
    """Safely merges dictionary updates to prevent agents from overwriting each other."""
    return {**(left or {}), **(right or {})}


class TurnScore(TypedDict):
    """Scores assigned to agents for a specific turn."""

    turn: int
    scores: Dict[str, float]
    feedback: Optional[str]


class DebateState(TypedDict):
    """LangGraph state schema for managing an AI multi-agent debate session."""

    # Built-in message list: Replaces custom `arguments` and `tools_used`.
    # IMPORTANT: every AIMessage/ToolMessage appended here MUST set name=<agent_id>.
    # Since per-agent Argument/ToolUsage records are gone, `name` is the only way
    # to tell which agent produced a given message or triggered a given tool call.
    messages: Annotated[List[AnyMessage], add_messages]

    # Original prompt / debate topic
    original_prompt: str

    # Moderator interruption control
    moderator_interrupted: bool
    moderator_message: Optional[str]
    # Agent addressed directly by the moderator, parsed once from moderator_message
    # inside handle_interrupt, so downstream nodes don't re-parse raw text.
    moderator_target: Optional[str]

    # Turn tracking
    # Optional: no current speaker yet before select_agent has run for the first time.
    current_turn: Optional[str]
    previous_turn: Optional[str]

    # Merge reducers prevent single key updates from wiping full dictionary state
    turns_left_per_agent: Annotated[Dict[str, int], merge_dicts]
    cumulative_scores: Annotated[Dict[str, float], merge_dicts]

    # Agent details & scoring
    agent_ids: List[str]
    scores_per_turn: Annotated[List[TurnScore], operator.add]

    # Exit / Stop interrupt flag
    exit_interrupt: bool