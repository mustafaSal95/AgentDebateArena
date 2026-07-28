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


def render_eval_report(report):
    """Renders rich tables for the evaluation report."""
    console.rule("[bold magenta]Evaluation Report[/bold magenta]")
    console.print(f"[bold]Topic:[/bold] {report.topic}")
    console.print(f"[bold]Total Turns Evaluated:[/bold] {report.total_turns}\n")

    # 1. Summary Overview Table
    summary_table = Table(title="Overall Evaluation Metrics", show_header=True, header_style="bold green")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Score / Rate", justify="right")

    summary_table.add_row("Tool Compliance Rate", f"{report.overall_tool_compliance_rate:.1%}")
    summary_table.add_row("Tool Validity Rate", f"{report.overall_tool_validity_rate:.1%}")
    summary_table.add_row("Citation Support Rate", f"{report.overall_citation_support_rate:.1%}")
    summary_table.add_row("Format Compliance Rate", f"{report.overall_format_compliance_rate:.1%}")
    summary_table.add_row("Unsupported Claim Rate", f"{report.overall_unsupported_claim_rate:.1%}")

    console.print(summary_table)
    console.print()

    # 2. Per-Agent Breakdown
    if report.agent_summaries:
        agent_table = Table(title="Per-Agent Performance Summary", show_header=True, header_style="bold cyan")
        agent_table.add_column("Agent")
        agent_table.add_column("Turns", justify="right")
        agent_table.add_column("Avg Words", justify="right")
        agent_table.add_column("Format OK %", justify="right")
        agent_table.add_column("Tool Calls", justify="right")
        agent_table.add_column("Supported Claim %", justify="right")

        for agent_id, metrics in report.agent_summaries.items():
            agent_table.add_row(
                f"[{_color_for(agent_id)}]{agent_id}[/{_color_for(agent_id)}]",
                str(metrics["turns_count"]),
                f"{metrics['avg_word_count']:.1f}",
                f"{metrics['format_compliance_rate']:.1%}",
                str(metrics["tool_calls_count"]),
                f"{metrics['avg_supported_claim_rate']:.1%}",
            )
        console.print(agent_table)
        console.print()

    # 3. Turn-by-Turn Detail Table
    if report.turn_records:
        turns_table = Table(title="Turn-by-Turn Evaluation Breakdown", show_header=True, header_style="bold yellow", show_lines=True)
        turns_table.add_column("Turn", justify="right")
        turns_table.add_column("Agent")
        turns_table.add_column("Words")
        turns_table.add_column("Tool Calls")
        turns_table.add_column("Claims (Supp/Tot)")
        turns_table.add_column("Format OK")
        turns_table.add_column("Critique / Issues")

        for record in report.turn_records:
            fmt_str = "[green]Yes[/green]" if record.format_compliant else "[red]No[/red]"
            claims_str = f"{record.supported_claims_count}/{record.numeric_claims_count}"
            critique_str = ""
            if record.llm_critique:
                critique_str = record.llm_critique
            elif record.format_issues:
                critique_str = "; ".join(record.format_issues)
            else:
                critique_str = "[dim]No issues detected[/dim]"

            turns_table.add_row(
                str(record.turn_number),
                f"[{_color_for(record.agent_id)}]{record.agent_id}[/{_color_for(record.agent_id)}]",
                str(record.word_count),
                str(record.tool_calls_count),
                claims_str,
                fmt_str,
                critique_str,
            )
        console.print(turns_table)
        console.print()
