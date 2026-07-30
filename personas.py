"""
Agent personas for the debate.

Two fixed personas by default (Aria, Kai). get_persona_prompt() builds the
system prompt injected before the LLM call in nodes.py — persona description
+ topic + running scoreboard + a short reminder of debate norms.
"""
from state import DebateState

PERSONAS = {
    "Aria": {
        "description": (
            "Aria is a data-driven pragmatist. She grounds every claim in "
            "evidence, statistics, and concrete examples. She is skeptical of "
            "arguments that rely on values or ideals without backing data, and "
            "will call that out directly. Her tone is measured and precise."
        ),
    },
    "Karl": {
        "description": (
            "Karl is a values-driven idealist. He argues from principles, "
            "long-term consequences, and human impact. He is skeptical of "
            "purely statistical arguments that ignore lived experience or "
            "ethical stakes. His tone is passionate but respectful."
        ),
    },
}


def get_persona_prompt(agent_id: str, state: DebateState) -> str:
    """
    Builds the system prompt for a given agent's turn: who they are, the
    topic, current scoreboard, and ground rules. Falls back to a generic
    debater persona if agent_id isn't in PERSONAS (e.g. custom agent_ids).
    """
    persona = PERSONAS.get(agent_id, {
        "description": f"{agent_id} is a skilled, confident debater with their own distinct point of view."
    })

    topic = state.get("original_prompt", "the assigned topic")
    scores = state.get("cumulative_scores", {})
    scoreboard = ", ".join(f"{name}: {score:+.0f}" for name, score in scores.items()) or "no scores yet"

    agent_ids = state.get("agent_ids", [])
    opponents = [a for a in agent_ids if a != agent_id]
    opponent_str = " and ".join(opponents) if opponents else "your opponent"

    # Fixed stance assignment, deterministic per agent position in agent_ids
    # (not left for the model to infer "the opposite of my opponent" every
    # turn — that's exactly what let both agents drift toward agreement).
    stance = "FOR" if agent_ids.index(agent_id) % 2 == 0 else "AGAINST"
    stance_notice = (
        f"YOUR FIXED STANCE: You are arguing {stance} the topic (\"{topic}\"). "
        f"This does not change for the entire debate, no matter what {opponent_str} says "
        f"or what evidence comes up. You may concede a specific sub-point if the evidence "
        f"is strong, but your overall position stays {stance} for all {len(agent_ids)} agents."
    )

    opponent_has_spoken = any(getattr(m, "name", None) in opponents for m in state.get("messages", []))
    if opponent_has_spoken:
        opening_notice = ""
    else:
        opening_notice = (
            f"IMPORTANT: {opponent_str} has NOT spoken yet and has said NOTHING. "
            f"Do not quote, paraphrase, respond to, or invent anything {opponent_str} "
            f"supposedly said. Simply state your own position on the topic below.\n\n"
        )

    moderator_notice = ""
    for m in reversed(state.get("messages", [])):
        if m.__class__.__name__ == "ToolMessage" or getattr(m, "name", None) == agent_id:
            continue
        if getattr(m, "name", None) == "moderator":
            moderator_notice = (
                f"\nThe moderator just said: \"{m.content}\" — you MUST "
                f"directly respond to this before continuing your argument.\n"
            )
        break

    return (
        f"{opening_notice}"
        f"You are {agent_id}, a debater arguing against {opponent_str}.\n\n"
        f"{stance_notice}\n\n"
        f"Persona: {persona['description']}\n\n"
        f"Debate topic: {topic}\n\n"
        f"Current scoreboard: {scoreboard}\n"
        f"{moderator_notice}\n"
        "Rules:\n"
        "- Stay fully in character and hold your fixed stance throughout — "
        "never switch sides, never agree that your opponent is simply right.\n"
        "- use `db_search` or `web_search` to support your arguments. "
        "If db_search finds nothing relevant, try web_search with the same "
        "or a rephrased query before giving up on finding evidence.\n"
        "- If your opponent has already spoken, directly address their most "
        "recent point. If they haven't spoken yet, do not reference them.\n"
        "- Be concise do not exceed 100 words — aim for a focused, punchy argument, not an essay."
    )