"""
Central configuration for DebateArena.
All environment-driven settings live here — nothing else should read
os.environ directly.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    model_name: str = os.getenv("DEBATE_MODEL", "llama-3.3-70b-versatile")
    temperature: float = float(os.getenv("DEBATE_TEMPERATURE", "0.7"))
    max_turns_per_agent: int = int(os.getenv("DEBATE_MAX_TURNS", "6"))
    agent_timeout_seconds: int = int(os.getenv("DEBATE_AGENT_TIMEOUT", "30"))

    @property
    def is_offline(self) -> bool:
        """No API key configured -> caller should fall back to an offline/echo model."""
        return not bool(self.groq_api_key)

    @property
    def has_gemini(self) -> bool:
        """Whether a Gemini API key is available for fallback."""
        return bool(self.gemini_api_key)


config = Config()