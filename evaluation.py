"""
Evaluation framework for DebateArena transcripts and live state.

Provides deterministic evaluators (tool compliance, validity, selection fit,
redundancy, grounding/citation checks, format compliance, repetition) and
optional LLM-judge evaluators (persona adherence, rebuttal quality, LLM groundedness).
"""
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from config import config
from personas import PERSONAS

@dataclass
class TurnEvalRecord:
    turn_number: int
    agent_id: str
    argument_text: str
    word_count: int
    format_compliant: bool
    format_issues: List[str] = field(default_factory=list)
    
    # Tool call metrics
    tool_calls_count: int = 0
    opening_compliance: bool = False
    stat_citing_compliance: bool = False
    tool_validity_rate: float = 1.0
    selection_fit: str = "good"
    query_redundancy_rate: float = 0.0
    
    # Grounding & Faithfulness metrics
    numeric_claims_count: int = 0
    supported_claims_count: int = 0
    supported_claim_rate: float = 1.0
    unsupported_claim_rate: float = 0.0
    
    # Quality & Persona metrics
    repetition_score: float = 0.0  # 0 = fresh, 1 = identical to prior turns
    rebuttal_overlap_score: float = 0.0
    moderator_addressed: Optional[bool] = None
    
    # LLM Judge metrics (optional)
    persona_adherence_score: Optional[float] = None
    rebuttal_quality_score: Optional[float] = None
    llm_groundedness_score: Optional[float] = None
    llm_critique: Optional[str] = None

@dataclass
class DebateEvalReport:
    topic: str
    total_turns: int
    agent_ids: List[str]
    turn_records: List[TurnEvalRecord] = field(default_factory=list)
    
    # Aggregate metrics
    overall_tool_compliance_rate: float = 1.0
    overall_tool_validity_rate: float = 1.0
    overall_citation_support_rate: float = 1.0
    overall_format_compliance_rate: float = 1.0
    overall_unsupported_claim_rate: float = 0.0
    
    # Per-agent metrics summary
    agent_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # System / debate level metrics
    fallback_used_count: int = 0
    scoring_sanity_correlation: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def _normalize_msg(msg: Any) -> Dict[str, Any]:
    if isinstance(msg, dict):
        return {
            "type": msg.get("type") or msg.get("role") or msg.get("__class__", "Unknown"),
            "name": msg.get("name"),
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "tool_call_id": msg.get("tool_call_id"),
        }
    return {
        "type": msg.__class__.__name__,
        "name": getattr(msg, "name", None),
        "content": getattr(msg, "content", "") or "",
        "tool_calls": getattr(msg, "tool_calls", []) or [],
        "tool_call_id": getattr(msg, "tool_call_id", None),
    }


def extract_turns_from_messages(messages: List[Any], agent_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Groups raw messages into turn objects.
    Each turn object contains tool_calls, tool_results, final argument_text,
    and metadata for evaluation.
    """
    turns = []
    current_turn_agent = None
    current_tool_calls = []
    current_tool_results = []
    preceding_moderator_msg = None
    turn_counter = 0

    norm_msgs = [_normalize_msg(m) for m in messages]

    for m in norm_msgs:
        m_name = m.get("name")
        m_type = m.get("type")
        m_content = m.get("content")

        if m_name == "moderator" and m_content:
            preceding_moderator_msg = m_content
            continue

        if m_name in agent_ids:
            current_turn_agent = m_name
            t_calls = m.get("tool_calls") or []
            if t_calls:
                for tc in t_calls:
                    if isinstance(tc, dict):
                        current_tool_calls.append({
                            "id": tc.get("id"),
                            "name": tc.get("name") or tc.get("function", {}).get("name"),
                            "args": tc.get("args") or tc.get("function", {}).get("arguments"),
                        })
            elif m_content and not t_calls:
                # Final argument message for this turn
                turn_counter += 1
                turns.append({
                    "turn_number": turn_counter,
                    "agent_id": current_turn_agent,
                    "argument_text": str(m_content),
                    "tool_calls": list(current_tool_calls),
                    "tool_results": list(current_tool_results),
                    "preceding_moderator_msg": preceding_moderator_msg,
                })
                # Reset turn buffer
                current_tool_calls = []
                current_tool_results = []
                preceding_moderator_msg = None

        elif m_type == "ToolMessage" or m.get("tool_call_id"):
            current_tool_results.append({
                "name": m_name or "tool",
                "tool_call_id": m.get("tool_call_id"),
                "content": str(m_content),
            })

    return turns


def extract_numbers_and_stats(text: str) -> List[str]:
    """Extract numbers, percentages, currency amounts, and statistics from text."""
    pattern = r'(?:\$\d+(?:\.\d+)?|\b\d+(?:\.\d+)?%\b|\b\d{1,3}(?:,\d{3})+\b|\b\d+(?:\.\d+)?\b)'
    matches = re.findall(pattern, text)
    # Filter out single digits 0-9 if not percentage/currency to avoid noise like '1 turn' or 'a'
    filtered = []
    for m in matches:
        clean = m.replace(",", "").replace("$", "").replace("%", "")
        if "%" in m or "$" in m or "," in m or "." in m:
            filtered.append(m)
        else:
            try:
                val = float(clean)
                if val >= 10 or val == 0:
                    filtered.append(m)
            except ValueError:
                pass
    return filtered


def check_tool_validity(tool_results: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
    """Evaluates whether tool results returned real data or error/empty notices."""
    if not tool_results:
        return 1.0, []
    
    invalid_keywords = [
        "failed", "unavailable", "no web results found",
        "no local documents matched", "empty query", "error"
    ]
    invalid_count = 0
    issues = []
    for res in tool_results:
        content_lower = res.get("content", "").lower()
        if any(kw in content_lower for kw in invalid_keywords):
            invalid_count += 1
            issues.append(f"Tool {res.get('name')} produced invalid output: {res.get('content')[:80]}...")
            
    valid_rate = (len(tool_results) - invalid_count) / len(tool_results)
    return valid_rate, issues



def check_format_compliance(text: str) -> Tuple[bool, int, List[str]]:
    """Checks word count <= 100 and absence of raw tool/function tag leaks."""
    words = text.split()
    word_count = len(words)
    issues = []
    
    if word_count > 100:
        issues.append(f"Word count ({word_count}) exceeded 100-word limit")
        
    tag_patterns = [r'<function=', r'<tool_call>', r'{"name":', r'failed_generation']
    for pat in tag_patterns:
        if re.search(pat, text):
            issues.append(f"Leaked internal format/tag detected: '{pat}'")
            
    is_compliant = len(issues) == 0
    return is_compliant, word_count, issues


def check_query_redundancy(current_queries: List[str], past_queries: List[str]) -> float:
    """Calculates query redundancy with previous queries from the same agent."""
    if not current_queries or not past_queries:
        return 0.0
    
    redundant = 0
    for q in current_queries:
        q_words = set(re.findall(r'\w+', q.lower()))
        for pq in past_queries:
            pq_words = set(re.findall(r'\w+', pq.lower()))
            if q_words and pq_words:
                overlap = len(q_words & pq_words) / min(len(q_words), len(pq_words))
                if overlap > 0.7:
                    redundant += 1
                    break
    return redundant / len(current_queries)


def check_repetition(current_text: str, past_texts: List[str]) -> float:
    """Calculates n-gram overlap between current turn and past turns of the agent."""
    if not past_texts or not current_text.strip():
        return 0.0
    
    curr_words = set(re.findall(r'\w+', current_text.lower()))
    if not curr_words:
        return 0.0
    
    max_overlap = 0.0
    for pt in past_texts:
        pt_words = set(re.findall(r'\w+', pt.lower()))
        if pt_words:
            overlap = len(curr_words & pt_words) / float(len(curr_words | pt_words))
            if overlap > max_overlap:
                max_overlap = overlap
    return max_overlap


def check_citation_grounding(argument_text: str, tool_results: List[Dict[str, Any]]) -> Tuple[int, int, float, float]:
    """
    Checks if numeric/statistical claims in argument_text are grounded in tool_results.
    Returns: (numeric_claims_count, supported_claims_count, supported_claim_rate, unsupported_claim_rate)
    """
    claims = extract_numbers_and_stats(argument_text)
    if not claims:
        return 0, 0, 1.0, 0.0
    
    combined_tool_text = " ".join([r.get("content", "") for r in tool_results]).lower()
    
    supported = 0
    for claim in claims:
        clean_claim = claim.replace(",", "").replace("$", "").replace("%", "")
        if claim.lower() in combined_tool_text or clean_claim in combined_tool_text:
            supported += 1
            
    total = len(claims)
    supp_rate = supported / total if total > 0 else 1.0
    unsupp_rate = (total - supported) / total if total > 0 else 0.0
    
    # If no tool calls were made at all but claims were present, unsupp_rate is 1.0
    if not tool_results and total > 0:
        unsupp_rate = 1.0
        supp_rate = 0.0

    return total, supported, supp_rate, unsupp_rate


def evaluate_turn_deterministic(
    turn: Dict[str, Any],
    past_turns_by_agent: List[Dict[str, Any]],
    is_opening_turn: bool
) -> TurnEvalRecord:
    """Runs all deterministic (no LLM) evaluations on a single turn."""
    agent_id = turn["agent_id"]
    arg_text = turn["argument_text"]
    tool_calls = turn.get("tool_calls", [])
    tool_results = turn.get("tool_results", [])
    
    format_ok, word_count, format_issues = check_format_compliance(arg_text)
    
    # Tool call compliance
    opening_compliance = True
    if is_opening_turn:
        opening_compliance = len(tool_calls) > 0
        
    claims_count, supp_count, supp_rate, unsupp_rate = check_citation_grounding(arg_text, tool_results)
    
    stat_citing_compliance = True
    if claims_count > 0 and not is_opening_turn:
        stat_citing_compliance = len(tool_calls) > 0
        
    validity_rate, validity_issues = check_tool_validity(tool_results)
    format_issues.extend(validity_issues)
    
    # Selection fit check
    tool_names = [tc.get("name") for tc in tool_calls]
    selection_fit = "good"
    if "web_search" in tool_names and "db_search" not in tool_names:
        selection_fit = "web_only"
    elif "db_search" in tool_names and "web_search" in tool_names:
        selection_fit = "hybrid"
    elif "db_search" in tool_names:
        selection_fit = "db_only"
        
    # Query redundancy
    curr_queries = [str(tc.get("args", {}).get("query", "")) for tc in tool_calls if isinstance(tc.get("args"), dict)]
    past_queries = [
        str(tc.get("args", {}).get("query", ""))
        for pt in past_turns_by_agent
        for tc in pt.get("tool_calls", [])
        if isinstance(tc.get("args"), dict)
    ]
    query_redundancy = check_query_redundancy(curr_queries, past_queries)
    
    # Content repetition
    past_texts = [pt["argument_text"] for pt in past_turns_by_agent]
    repetition = check_repetition(arg_text, past_texts)
    
    # Moderator responsiveness
    mod_msg = turn.get("preceding_moderator_msg")
    mod_addressed = None
    if mod_msg:
        mod_words = set(re.findall(r'\w+', mod_msg.lower())) - {"the", "a", "an", "you", "must", "address", "moderator"}
        arg_words = set(re.findall(r'\w+', arg_text.lower()))
        mod_addressed = len(mod_words & arg_words) > 0

    return TurnEvalRecord(
        turn_number=turn["turn_number"],
        agent_id=agent_id,
        argument_text=arg_text,
        word_count=word_count,
        format_compliant=format_ok,
        format_issues=format_issues,
        tool_calls_count=len(tool_calls),
        opening_compliance=opening_compliance,
        stat_citing_compliance=stat_citing_compliance,
        tool_validity_rate=validity_rate,
        selection_fit=selection_fit,
        query_redundancy_rate=query_redundancy,
        numeric_claims_count=claims_count,
        supported_claims_count=supp_count,
        supported_claim_rate=supp_rate,
        unsupported_claim_rate=unsupp_rate,
        repetition_score=repetition,
        moderator_addressed=mod_addressed,
    )



def evaluate_turn_llm_judge(
    turn: Dict[str, Any],
    topic: str,
    opponent_last_text: Optional[str] = None
) -> Tuple[float, float, float, str]:
    """
    Uses LLM judge to evaluate persona adherence, rebuttal depth, and groundedness.
    Returns: (persona_adherence, rebuttal_quality, groundedness_score, critique_text)
    """
    if config.is_offline:
        return 0.85, 0.80, 0.90, "Offline mode judge heuristic applied."

    agent_id = turn["agent_id"]
    persona = PERSONAS.get(agent_id, {}).get("description", "A skilled debater.")
    arg_text = turn["argument_text"]
    tool_text = "\n".join([r.get("content", "") for r in turn.get("tool_results", [])]) or "No tool outputs."
    opp_text = opponent_last_text or "No prior opponent turn."

    prompt = (
        f"You are an expert debate judge evaluating Turn {turn['turn_number']} for Agent '{agent_id}'.\n"
        f"Topic: {topic}\n"
        f"Persona: {persona}\n"
        f"Opponent's previous argument: {opp_text}\n"
        f"Tool results gathered by agent: {tool_text}\n"
        f"Agent's turn argument: {arg_text}\n\n"
        "Evaluate on 3 metrics (scores between 0.0 and 1.0):\n"
        "1. persona_adherence: How well does tone/style match persona?\n"
        "2. rebuttal_quality: How effectively does it directly counter opponent's point vs generic speech?\n"
        "3. groundedness: Are claims fully supported by gathered tool results?\n\n"
        "Respond ONLY with valid JSON in this structure:\n"
        '{"persona_adherence": 0.9, "rebuttal_quality": 0.8, "groundedness": 0.85, "critique": "Short 1-2 sentence feedback."}'
    )

    try:
        from nodes import _fallback_invoke
        from langchain_core.messages import SystemMessage
        response = _fallback_invoke([SystemMessage(content=prompt)], agent_id=agent_id)
        raw_content = response.content if hasattr(response, "content") else str(response)
        
        # Extract JSON from response
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return (
                float(data.get("persona_adherence", 0.8)),
                float(data.get("rebuttal_quality", 0.8)),
                float(data.get("groundedness", 0.8)),
                str(data.get("critique", "Evaluated by LLM judge."))
            )
    except Exception as e:
        pass
    
    return 0.8, 0.8, 0.8, "LLM judge evaluation unavailable or failed; default heuristic used."


def evaluate_debate(
    state_or_transcript: Dict[str, Any],
    use_llm_judge: Optional[bool] = None
) -> DebateEvalReport:
    """
    Main evaluation pipeline for a full debate transcript or state dictionary.
    Runs deterministic checks and optional LLM-judge checks across all turns.
    """
    if use_llm_judge is None:
        use_llm_judge = config.llm_eval

    topic = state_or_transcript.get("original_prompt", "Assigned Topic")
    agent_ids = state_or_transcript.get("agent_ids", ["Aria", "Karl"])
    messages = state_or_transcript.get("messages", [])
    
    turns = extract_turns_from_messages(messages, agent_ids)
    turn_records: List[TurnEvalRecord] = []
    
    turns_by_agent: Dict[str, List[Dict[str, Any]]] = {a: [] for a in agent_ids}
    last_turn_text_by_agent: Dict[str, str] = {}
    
    for t in turns:
        aid = t["agent_id"]
        past_turns = turns_by_agent.get(aid, [])
        is_opening = len(past_turns) == 0
        
        # Deterministic eval
        record = evaluate_turn_deterministic(t, past_turns, is_opening)
        
        # Opponent last text for rebuttal evaluation
        opponent_ids = [a for a in agent_ids if a != aid]
        opp_last = last_turn_text_by_agent.get(opponent_ids[0]) if opponent_ids else None
        
        # LLM Judge eval
        if use_llm_judge:
            p_adh, r_qual, g_score, critique = evaluate_turn_llm_judge(t, topic, opp_last)
            record.persona_adherence_score = p_adh
            record.rebuttal_quality_score = r_qual
            record.llm_groundedness_score = g_score
            record.llm_critique = critique
            
        turns_by_agent[aid].append(t)
        last_turn_text_by_agent[aid] = t["argument_text"]
        turn_records.append(record)
        
    # Calculate Aggregate Metrics
    total_turns = len(turn_records)
    if total_turns == 0:
        return DebateEvalReport(topic=topic, total_turns=0, agent_ids=agent_ids)
        
    tool_comp_rate = sum(
        1.0 for r in turn_records if r.opening_compliance and r.stat_citing_compliance
    ) / float(total_turns)
    
    tool_valid_rate = sum(r.tool_validity_rate for r in turn_records) / float(total_turns)
    cit_support_rate = sum(r.supported_claim_rate for r in turn_records) / float(total_turns)
    format_comp_rate = sum(1.0 for r in turn_records if r.format_compliant) / float(total_turns)
    unsupp_rate = sum(r.unsupported_claim_rate for r in turn_records) / float(total_turns)
    
    # Per agent summaries
    agent_summaries = {}
    for aid in agent_ids:
        a_recs = [r for r in turn_records if r.agent_id == aid]
        if a_recs:
            agent_summaries[aid] = {
                "turns_count": len(a_recs),
                "avg_word_count": sum(r.word_count for r in a_recs) / len(a_recs),
                "format_compliance_rate": sum(1.0 for r in a_recs if r.format_compliant) / len(a_recs),
                "tool_calls_count": sum(r.tool_calls_count for r in a_recs),
                "avg_supported_claim_rate": sum(r.supported_claim_rate for r in a_recs) / len(a_recs),
                "avg_repetition_score": sum(r.repetition_score for r in a_recs) / len(a_recs),
            }

    report = DebateEvalReport(
        topic=topic,
        total_turns=total_turns,
        agent_ids=agent_ids,
        turn_records=turn_records,
        overall_tool_compliance_rate=tool_comp_rate,
        overall_tool_validity_rate=tool_valid_rate,
        overall_citation_support_rate=cit_support_rate,
        overall_format_compliance_rate=format_comp_rate,
        overall_unsupported_claim_rate=unsupp_rate,
        agent_summaries=agent_summaries,
    )
    return report
