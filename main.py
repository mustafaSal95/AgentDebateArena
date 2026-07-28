"""
Entry point for DebateArena.

Builds the graph once, prompts for a topic, initializes DebateState, and
invokes the graph. Since the graph itself loops internally (select_agent ->
... -> display -> check_human_interrupt -> back to select_agent) until END,
a single .invoke() runs the whole debate; recursion_limit is raised to allow
enough turns.
"""
import json
import random
from pathlib import Path
from graph import build_graph
from config import config

AGENT_IDS = ["Aria", "Karl"]
TURNS_PER_AGENT = config.max_turns_per_agent

def get_random_topic():
    default_topic = "Is remote work better than in-office work?"
    topics_file = Path(__file__).parent / "data" / "topics_list.json"
    if topics_file.exists():
        try:
            with open(topics_file, "r", encoding="utf-8") as f:
                topics = json.load(f)
            if topics:
                return random.choice(topics)
        except Exception:
            pass
    return default_topic



def build_initial_state(topic: str) -> dict:
    return {
        "messages": [],
        "original_prompt": topic,
        "moderator_interrupted": False,
        "moderator_message": None,
        "moderator_target": None,
        "current_turn": None,
        "previous_turn": None,
        "turns_left_per_agent": {a: TURNS_PER_AGENT for a in AGENT_IDS},
        "cumulative_scores": {a: 0.0 for a in AGENT_IDS},
        "agent_ids": AGENT_IDS,
        "scores_per_turn": [],
        "exit_interrupt": False,
    }


def main():
    if config.is_offline:
        print(
            "No GROQ_API_KEY found — running in offline mode with canned "
            "placeholder responses. Set GROQ_API_KEY in .env for real output."
        )

    suggested_topic = get_random_topic()
    topic = input(f"Debate topic [default: {suggested_topic}]: ").strip()
    if not topic:
        topic = suggested_topic
        print(f"No topic entered, using default: {topic}")

    app = build_graph()
    initial_state = build_initial_state(topic)

    from ui import render_debate_start, render_debate_end
    render_debate_start(topic, AGENT_IDS, TURNS_PER_AGENT)

    final_state = app.invoke(
        initial_state,
        config={"recursion_limit": (TURNS_PER_AGENT * len(AGENT_IDS) * 6) + 20},
    )

    summary = get_summary(final_state)
    render_debate_end(final_state, summary)


def get_summary(final_state: dict) -> str:
    if config.is_offline:
        from offline_model import get_offline_summary
        return get_offline_summary()

    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage

    llm = ChatGroq(
        model=config.model_name,
        temperature=0.3,
        api_key=config.groq_api_key,
        timeout=config.agent_timeout_seconds,
    )
    transcript = "\n".join(
        f"{getattr(m, 'name', 'unknown')}: {m.content}"
        for m in final_state["messages"]
        if getattr(m, "content", "")
    )
    try:
        response = llm.invoke([
            SystemMessage(content="Summarize this debate in 3-4 sentences:first summarise each of the debaters arguments then give a conclusion who made the stronger case and why."),
            SystemMessage(content=transcript),
        ])
        return response.content
    except Exception as e:
        if "rate_limit_exceeded" in str(e) or "Rate limit reached" in str(e):
            print("\n[warning] Groq rate limit reached while summarizing. Trying Gemini fallback...")
            from config import config as cfg
            if cfg.has_gemini:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    gemini = ChatGoogleGenerativeAI(
                        model=cfg.gemini_model,
                        temperature=0.7,
                        google_api_key=cfg.gemini_api_key,
                        timeout=cfg.agent_timeout_seconds,
                    )
                    response = gemini.invoke([
                        SystemMessage(content="Summarize this debate in 3-4 sentences:first summarise each of the debaters arguments then give a conclusion who made the stronger case and why."),
                        SystemMessage(content=transcript),
                    ])
                    return response.content
                except Exception as ge:
                    print(f"[warning] Gemini summary also failed ({ge}); using offline summary.")
            from offline_model import get_offline_summary
            return get_offline_summary()
        raise e


if __name__ == "__main__":
    main()