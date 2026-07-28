"""
Offline fallback "model" used when config.is_offline is True.

Not a real LLM, but not context-blind either: it parses the system prompt
built by personas.get_persona_prompt() (agent name, topic, persona flavor)
and the most recent opponent message out of the conversation, then fills a
persona-typed template. Still fully deterministic and does no reasoning —
but it reads as a real turn in the debate rather than an obviously fake
placeholder, which is what an "offline mode" should actually look like.

When the local knowledge base (data/*.txt) has relevant facts for the
debate topic, the offline model will incorporate them into its response,
making it much more substantive than a pure template.

Mimics enough of ChatGroq's interface for nodes.py to use interchangeably:
  - .invoke(messages) -> AIMessage
  - .bind_tools([...]) -> returns self (offline model never calls tools)
"""
import re
from langchain_core.messages import AIMessage, SystemMessage


def _extract_from_system(messages) -> dict:
    system_text = ""
    for m in messages:
        if isinstance(m, SystemMessage):
            system_text += str(m.content) + "\n"

    agent_match = re.search(r"You are (\w+)", system_text)
    topic_match = re.search(r"Debate topic:\s*(.+)", system_text)
    persona_match = re.search(r"Persona:\s*(.+)", system_text)

    return {
        "agent": agent_match.group(1) if agent_match else "This debater",
        "topic": topic_match.group(1).strip() if topic_match else "the topic",
        "persona_desc": persona_match.group(1) if persona_match else "",
    }


def _last_opponent_message(messages, agent_name: str):
    for m in reversed(messages):
        name = getattr(m, "name", None)
        if name and name != agent_name and name != "moderator" and getattr(m, "content", None):
            return name, str(m.content)
    return None, None


def _get_db_facts(topic: str, max_facts: int = 3) -> list[str]:
    """Search the local knowledge base for facts relevant to the topic."""
    try:
        from tools import db_search
        result = db_search.invoke({"query": topic})
        if result and "No local documents matched" not in result and "unavailable" not in result:
            # Split the db_search results into individual facts
            facts = [f.strip() for f in result.split("\n\n") if f.strip()]
            return facts[:max_facts]
    except Exception:
        pass
    return []


def _build_fact_paragraph(facts: list[str]) -> str:
    """Turn raw db_search results into a clean paragraph of evidence."""
    if not facts:
        return ""

    # Strip the [filename] prefix from each fact
    cleaned = []
    for fact in facts:
        # Remove leading [filename.txt] tags
        cleaned_fact = re.sub(r"^\[.*?\]\s*", "", fact).strip()
        # Remove "Topic: ..." header lines
        cleaned_fact = re.sub(r"^Topic:.*?\n+", "", cleaned_fact).strip()
        if cleaned_fact:
            cleaned.append(cleaned_fact)

    if not cleaned:
        return ""

    return " ".join(cleaned)


_OPENING_WITH_FACTS = [
    "The evidence speaks for itself on {topic_ref}. {facts}",
    "When we look at the data on {topic_ref}, the picture is clear. {facts}",
    "Consider the facts about {topic_ref}: {facts}",
]

_OPENING_NO_FACTS = [
    "On the question of {topic_ref}, my position is clear.",
    "The debate around {topic_ref} deserves careful consideration.",
    "Let's examine what we know about {topic_ref}.",
]

_REBUTTAL_WITH_FACTS = [
    "{opponent} raised {opponent_gist}, but the evidence tells a different story. {facts}",
    "While {opponent} argues {opponent_gist}, the data suggests otherwise. {facts}",
]

_REBUTTAL_NO_FACTS = [
    "That doesn't fully address {opponent}'s point about {opponent_gist}.",
    "{opponent} raised {opponent_gist}, but that overlooks a key factor.",
]

_CLOSER_OFFLINE = (
    "\n\n[offline mode — templated response with local knowledge base, "
    "no live LLM reasoning — set GROQ_API_KEY for real arguments]"
)


class OfflineChatModel:
    """Drop-in stand-in for ChatGroq when no API key is configured."""

    def bind_tools(self, tools):
        # Offline model never calls tools — return self so
        # llm.bind_tools(...).invoke(...) chains work unchanged in nodes.py.
        return self

    def invoke(self, messages):
        info = _extract_from_system(messages)
        opponent, opponent_text = _last_opponent_message(messages, info["agent"])

        # Search the local knowledge base for relevant facts
        facts = _get_db_facts(info["topic"])
        fact_paragraph = _build_fact_paragraph(facts)

        parts = []

        if opponent and opponent_text:
            # Rebuttal turn
            gist = " ".join(opponent_text.split()[:8]) + "..."
            if fact_paragraph:
                templates = _REBUTTAL_WITH_FACTS
            else:
                templates = _REBUTTAL_NO_FACTS
            rebuttal = templates[
                hash(opponent_text) % len(templates)
            ].format(opponent=opponent, opponent_gist=gist, facts=fact_paragraph)
            parts.append(rebuttal)
        else:
            # Opening turn
            if fact_paragraph:
                templates = _OPENING_WITH_FACTS
            else:
                templates = _OPENING_NO_FACTS
            opener = templates[
                hash(info["topic"]) % len(templates)
            ].format(topic_ref=info["topic"], facts=fact_paragraph)
            parts.append(opener)

        parts.append(_CLOSER_OFFLINE)

        return AIMessage(content=" ".join(parts))


def get_offline_summary() -> str:
    return (
        "[offline mode] No live LLM was available to generate a real summary — "
        "the debate ran end-to-end using templated placeholder arguments. "
        "Set GROQ_API_KEY in .env and re-run for an actual debate and summary."
    )