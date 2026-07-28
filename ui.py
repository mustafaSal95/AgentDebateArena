"""
Terminal UI for DebateArena, built on `rich`.

render_turn() is called from nodes.display() after every single message.
It shows: the speaker's message, a table of any tool calls made this turn
(tool name, args, result), and the live scoreboard.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

_AGENT_COLORS = {}
_PALETTE = ["cyan", "magenta", "green", "yellow", "blue"]


def _color_for(agent_id: str) -> str:
    if agent_id not in _AGENT_COLORS:
        _AGENT_COLORS[agent_id] = _PALETTE[len(_AGENT_COLORS) % len(_PALETTE)]
    return _AGENT_COLORS[agent_id]


def _get_turn_tool_calls(state, agent_id: str) -> list[dict]:
    """
    Reconstructs this turn's tool calls by walking backward from the final
    argument message through any (AIMessage-with-tool_calls, ToolMessage*)
    pairs that preceded it, matching results to calls via tool_call_id.
    """
    messages = state["messages"]
    if len(messages) < 2:
        return []

    idx = len(messages) - 2  # skip the final argument message itself
    pending_calls: dict[str, tuple] = {}   # tool_call_id -> (tool_name, args)
    tool_messages = []

    while idx >= 0:
        m = messages[idx]
        cls = m.__class__.__name__

        if cls == "ToolMessage":
            tool_messages.append(m)
            idx -= 1
            continue

        if cls == "AIMessage" and getattr(m, "name", None) == agent_id:
            tool_calls = getattr(m, "tool_calls", None) or []
            if tool_calls:
                for tc in tool_calls:
                    pending_calls[tc["id"]] = (tc["name"], tc.get("args", {}))
            idx -= 1
            continue

        break

    info = []
    for tm in reversed(tool_messages):
        tc_id = getattr(tm, "tool_call_id", None)
        name, args = pending_calls.get(tc_id, (getattr(tm, "name", "tool"), {}))
        info.append({"tool": name, "args": args, "result": str(tm.content)})
    return info


def render_turn(state):
    """Renders the most recent turn: message panel, tool call table, scoreboard."""
    agent_id = state["current_turn"]
    last_msg = state["messages"][-1]
    last_score = state["scores_per_turn"][-1] if state["scores_per_turn"] else None
    color = _color_for(agent_id)

    turn_label = f"Turn {last_score['turn']}" if last_score else "Turn"
    console.print(
        Panel(
            Text(str(last_msg.content), style="white"),
            title=f"[bold {color}]{agent_id}[/bold {color}] — {turn_label}",
            border_style=color,
        )
    )

    tool_calls = _get_turn_tool_calls(state, agent_id)
    if tool_calls:
        table = Table(title="Tool calls this turn", show_lines=True)
        table.add_column("Tool", style="bold")
        table.add_column("Args")
        table.add_column("Result")
        for tc in tool_calls:
            result_preview = tc["result"][:200] + ("..." if len(tc["result"]) > 200 else "")
            table.add_row(tc["tool"], str(tc["args"]), result_preview)
        console.print(table)
    else:
        console.print("[dim]No tools used this turn.[/dim]")

    if last_score:
        delta = last_score["scores"].get(agent_id, 0)
        sign = "+" if delta >= 0 else ""
        console.print(f"[dim]Score delta: {sign}{delta:.0f} ({last_score['feedback']})[/dim]")

    scoreboard = Table(title="Scoreboard", show_header=True, header_style="bold")
    scoreboard.add_column("Agent")
    scoreboard.add_column("Score", justify="right")
    for name, score in state["cumulative_scores"].items():
        scoreboard.add_row(f"[{_color_for(name)}]{name}[/{_color_for(name)}]", f"{score:+.0f}")
    console.print(scoreboard)
    console.print()


def render_debate_start(topic: str, agent_ids: list[str], turns_per_agent: int):
    console.rule("[bold]DebateArena[/bold]")
    console.print(f"[bold]Topic:[/bold] {topic}")
    console.print(f"[bold]Agents:[/bold] {', '.join(agent_ids)} — {turns_per_agent} turns each\n")


def render_debate_end(final_state, summary: str):
    console.rule("[bold]Debate finished[/bold]")
    scoreboard = Table(title="Final Scores", show_header=True, header_style="bold")
    scoreboard.add_column("Agent")
    scoreboard.add_column("Score", justify="right")
    for name, score in final_state["cumulative_scores"].items():
        scoreboard.add_row(f"[{_color_for(name)}]{name}[/{_color_for(name)}]", f"{score:+.0f}")
    console.print(scoreboard)
    console.print(Panel(Text(summary, style="italic"), title="Summary", border_style="dim"))