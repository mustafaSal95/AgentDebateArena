"""
Tools available to debate agents via LLM tool-calling.

  - web_search : live web search, no API key required (duckduckgo-search)
  - db_search   : keyword search over local .txt files in data/

Both are decorated with @tool so they can be passed straight to
llm.bind_tools([web_search, db_search]) in nodes.py.
"""
import os
import re
from pathlib import Path

from langchain_core.tools import tool

DATA_DIR = Path(__file__).parent / "data"


@tool
def web_search(query: str) -> str:
    """
    Search the live web for current information relevant to the debate topic.
    Use this for recent events, statistics, or facts you're not certain about.
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
            
        results = list(DDGS().text(query, max_results=4))
    except ImportError:
        return "web_search unavailable: install with `pip install duckduckgo-search`"
    except Exception as e:
        return f"web_search failed: {e}"

    if not results:
        return f"No web results found for '{query}'."

    formatted = []
    for r in results:
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()
        href = r.get("href", "").strip()
        formatted.append(f"- {title}: {body} ({href})")

    return "\n".join(formatted)


_DB_CACHE = None

def _load_db():
    global _DB_CACHE
    if _DB_CACHE is not None:
        return _DB_CACHE
    
    _DB_CACHE = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for para in paragraphs:
                para_terms = set(re.findall(r"\w+", para.lower()))
                _DB_CACHE.append((para_terms, path.name, para))
        except Exception:
            pass
    return _DB_CACHE

@tool
def db_search(query: str) -> str:
    """
    Search local reference documents (data/*.txt) for relevant facts or
    figures. Use this for grounded, citable evidence already vetted for
    this debate, before or instead of web_search.
    """
    if not DATA_DIR.exists():
        return "db_search unavailable: data/ directory not found."

    query_terms = set(re.findall(r"\w+", query.lower()))
    if not query_terms:
        return "db_search: empty query."

    scored_matches = []
    cache = _load_db()
    
    for para_terms, name, para in cache:
        overlap = len(query_terms & para_terms)
        if overlap > 0:
            scored_matches.append((overlap, name, para))

    if not scored_matches:
        return f"No local documents matched '{query}'."

    scored_matches.sort(key=lambda x: x[0], reverse=True)
    top = scored_matches[:3]

    formatted = [f"[{name}] {para}" for _, name, para in top]
    return "\n\n".join(formatted)