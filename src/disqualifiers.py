"""
src/disqualifiers.py
--------------------
Implements hard and soft disqualifier checks against the rules defined in
data/processed/jd_parsed.json → disqualifier_rules.

Each rule is implemented as a separate _check_* helper so the logic stays
readable and independently testable.

Public API
----------
check_disqualifiers(candidate: dict) -> tuple[bool, float, list[str]]
    Returns
    -------
    is_disqualified : bool
        True if ANY hard disqualifier fires.
    penalty : float [0.0 – 1.0]
        Accumulated soft penalty (capped at 1.0).  Hard disqualifiers
        contribute 1.0 each; soft penalties are additive but capped.
    reasons : list[str]
        Human-readable explanation for every rule that triggered.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Constants – consulting "body-shop" firms (from JD rule #6)
# ---------------------------------------------------------------------------
CONSULTING_FIRMS: set[str] = {
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "tech mahindra",
    "mphasis",
    "hexaware",
    "mindtree",   # Mindtree was merged into LTIMindtree (consulting)
    "ltimindtree",
}

# Titles that suggest non-coding / management / architecture roles
ARCH_LEAD_TITLE_PATTERNS: list[str] = [
    r"\barchitect\b",
    r"\btech(nical)?\s+lead\b",
    r"\btechnology\s+lead\b",
    r"\bdelivery\s+manager\b",
    r"\bengineering\s+manager\b",
    r"\bvp\s+of\s+engineering\b",
    r"\bhead\s+of\s+engineering\b",
    r"\bcto\b",
    r"\bchief\s+technology\b",
    r"\bprincipal\s+architect\b",
    r"\bsolution\s+architect\b",
    r"\bsystems\s+architect\b",
]

# Keywords that mark genuine *production / applied* ML work
# NOTE: Use compound phrases to avoid false positives (e.g. "no production deployment")
PRODUCTION_ML_KEYWORDS: list[str] = [
    "serving real users",
    "used by real users",
    "users in production",
    "a/b test",
    "a/b testing",
    "mlops",
    "ml serving",
    "feature store",
    "online serving",
    "retrieval system",
    "ranking system",
    "recommendation system",
    "search system",
    "embedding model deployed",
    "vector database",
    "vector store",
    "shipped to production",
    "launched to production",
    "production inference",
    "production serving",
    "production code",
    "production traffic",
    "deployed to production",
    "model serving",
    "model deployment",
    "deployed to real",
    "fine-tuned for production",
]

# Research / academic environment markers
RESEARCH_KEYWORDS: list[str] = [
    "research",
    "lab",
    "academic",
    "university",
    "phd",
    "ph.d",
    "postdoc",
    "scientist",
    "publication",
    "paper",
    "arxiv",
    "journal",
    "conference",
    "workshop",
    "nlp research",
    "ml research",
    "ai research",
]

# NLP / IR / ranking signal keywords (rule #7)
# Keep specific to LANGUAGE / TEXT / RETRIEVAL — avoid terms that overlap with CV
NLP_IR_KEYWORDS: list[str] = [
    "nlp",
    "natural language processing",
    "natural language understanding",
    "information retrieval",
    "text retrieval",
    "document retrieval",
    "search ranking",
    "search relevance",
    "semantic search",
    "text embedding",
    "sentence embedding",
    "sentence-transformers",
    "language model",
    "large language model",
    "llm",
    "bert",
    "transformer for text",
    "question answering",
    "named entity recognition",
    "sentiment analysis",
    "dialogue system",
    "machine translation",
    "text summarization",
    "text classification",
    "document classification",
    "word2vec",
    "fasttext",
    "bm25",
    "inverted index",
    "tf-idf",
    "tfidf",
    "query understanding",
    "query rewriting",
]

# CV / Speech / Robotics – non-NLP domains (rule #7)
NON_NLP_DOMAIN_KEYWORDS: list[str] = [
    "computer vision",
    "image classification",
    "object detection",
    "image segmentation",
    "image recognition",
    "face recognition",
    "optical flow",
    "speech recognition",
    "speech synthesis",
    "text-to-speech",
    "tts",
    "asr",
    "speaker recognition",
    "robotics",
    "ros",
    "slam",
    "autonomous",
    "lidar",
    "point cloud",
]

# Title seniority ladder used for title-hop detection (rule #4)
SENIORITY_LADDER: list[list[str]] = [
    ["intern", "trainee", "fresher"],
    ["junior", "associate", "entry"],
    ["engineer", "developer", "analyst", "scientist"],
    ["senior", "sr."],
    ["lead", "staff"],
    ["principal", "distinguished"],
    ["staff principal"],
    ["director", "vp", "head"],
]

# LangChain / wrapper-only AI markers (rule #2)
LANGCHAIN_WRAPPER_KEYWORDS: list[str] = [
    "langchain",
    "llamaindex",
    "llama index",
    "haystack",
    "flowise",
    "dify",
    "openai api",
    "chatgpt api",
    "chatgpt wrapper",
]
PRE_LLM_PRODUCTION_KEYWORDS: list[str] = [
    "ranking",
    "retrieval",
    "recommendation",
    "search engine",
    "information retrieval",
    "faiss",
    "elasticsearch",
    "opensearch",
    "solr",
    "lucene",
    "inverted index",
    "bm25",
    "learning to rank",
    "xgboost",
    "lightgbm",
    "word2vec",
    "fasttext",
    "sklearn",
    "scikit",
    "deployed model",
    "production ml",
    "ml pipeline",
    "ml platform",
    "feature store",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _text_lower(text: str | None) -> str:
    """Return lowercased text or empty string if None/missing."""
    return (text or "").lower()


def _career_text(candidate: dict) -> str:
    """Concatenate all role titles + descriptions across career history."""
    parts: list[str] = []
    for role in candidate.get("career_history", []):
        parts.append(_text_lower(role.get("title", "")))
        parts.append(_text_lower(role.get("description", "")))
        parts.append(_text_lower(role.get("company", "")))
    return " ".join(parts)


def _skills_text(candidate: dict) -> str:
    return " ".join(_text_lower(s.get("name", "")) for s in candidate.get("skills", []))


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(kw in text for kw in keywords)


def _title_seniority(title: str) -> int:
    """Return an integer seniority level for a job title (higher = more senior)."""
    t = title.lower()
    for level, synonyms in enumerate(SENIORITY_LADDER):
        if any(syn in t for syn in synonyms):
            return level
    return 2  # default: mid-level


def _is_consulting_company(company_name: str) -> bool:
    name = company_name.lower().strip()
    return any(firm in name for firm in CONSULTING_FIRMS)


def _parse_date(date_str: str | None) -> date | None:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _months_since(start_date_str: str | None) -> int | None:
    """Return months elapsed since start_date until today, or None if unparseable."""
    d = _parse_date(start_date_str)
    if d is None:
        return None
    today = date.today()
    return (today.year - d.year) * 12 + (today.month - d.month)


# ---------------------------------------------------------------------------
# Rule checkers  (each returns a tuple: hard_disqualify, penalty, reason|None)
# ---------------------------------------------------------------------------

class _RuleResult(NamedTuple):
    hard: bool
    penalty: float
    reason: str | None


def _check_pure_research(candidate: dict) -> _RuleResult:
    """
    RULE 1: Pure research background with no production deployment → hard disqualify.

    Signal: every role description is research/academic AND none mention production
    deployment or applied ML shipping.
    """
    history = candidate.get("career_history", [])
    if not history:
        return _RuleResult(False, 0.0, None)

    all_research = True
    any_production = False

    for role in history:
        desc = _text_lower(role.get("description", ""))
        title = _text_lower(role.get("title", ""))
        combined = desc + " " + title

        if _contains_any(combined, PRODUCTION_ML_KEYWORDS):
            any_production = True

        if not _contains_any(combined, RESEARCH_KEYWORDS):
            all_research = False

    if all_research and not any_production:
        return _RuleResult(
            True,
            1.0,
            "Pure research background: every role is research/academic with no evidence "
            "of production deployment or applied ML shipping.",
        )
    return _RuleResult(False, 0.0, None)


def _check_langchain_only(candidate: dict) -> _RuleResult:
    """
    RULE 2: AI experience = only recent LangChain/wrapper projects < 12 months,
    no pre-LLM production ML experience → hard disqualify.
    """
    history = candidate.get("career_history", [])
    if not history:
        return _RuleResult(False, 0.0, None)

    langchain_months_total = 0
    has_pre_llm_production = False

    for role in history:
        desc = _text_lower(role.get("description", ""))
        title = _text_lower(role.get("title", ""))
        combined = desc + " " + title
        duration = role.get("duration_months", 0) or 0

        if _contains_any(combined, PRE_LLM_PRODUCTION_KEYWORDS):
            has_pre_llm_production = True

        if _contains_any(combined, LANGCHAIN_WRAPPER_KEYWORDS):
            langchain_months_total += duration

    if langchain_months_total > 0 and langchain_months_total < 12 and not has_pre_llm_production:
        return _RuleResult(
            True,
            1.0,
            f"AI experience appears to be only LangChain/wrapper projects "
            f"({langchain_months_total} months) with no demonstrated pre-LLM production ML experience.",
        )
    return _RuleResult(False, 0.0, None)


def _check_no_production_code_recent(candidate: dict) -> _RuleResult:
    """
    RULE 3: Senior engineer who hasn't written production code in the last 18 months
    because they're in architect/tech-lead-only roles → hard disqualify.

    Logic:
    - Find the most recent role(s) within the last 18 months.
    - If ALL of them are architecture/tech-lead/non-coding titles AND the
      candidate has 5+ years experience (senior), disqualify.
    """
    history = candidate.get("career_history", [])
    years_exp = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    if years_exp < 5:
        return _RuleResult(False, 0.0, None)  # Only relevant for senior folks

    today = date.today()
    cutoff_months = 18
    recent_roles = []

    for role in history:
        if role.get("is_current"):
            recent_roles.append(role)
            continue
        end_d = _parse_date(role.get("end_date"))
        if end_d:
            months_ago = (today.year - end_d.year) * 12 + (today.month - end_d.month)
            if months_ago <= cutoff_months:
                recent_roles.append(role)

    if not recent_roles:
        return _RuleResult(False, 0.0, None)

    all_arch_lead = True
    for role in recent_roles:
        title = _text_lower(role.get("title", ""))
        desc = _text_lower(role.get("description", ""))
        is_arch = any(re.search(pat, title) for pat in ARCH_LEAD_TITLE_PATTERNS)
        # If description mentions hands-on coding they still qualify
        coding_signals = [
            "implemented", "wrote", "built", "developed", "coded", "shipped",
            "production code", "pull request", "commit", "deployed"
        ]
        has_coding_desc = _contains_any(desc, coding_signals)
        if not is_arch or has_coding_desc:
            all_arch_lead = False

    if all_arch_lead:
        role_titles = [r.get("title", "") for r in recent_roles]
        return _RuleResult(
            True,
            1.0,
            f"Senior candidate ({years_exp:.1f} yrs) with no evidence of hands-on "
            f"production coding in the last 18 months. Recent roles: "
            f"{', '.join(role_titles)}. Role is hands-on; this is a hard disqualifier.",
        )
    return _RuleResult(False, 0.0, None)


def _check_title_hopping(candidate: dict) -> _RuleResult:
    """
    RULE 4: Title-hopping — 3+ jobs each < 18 months with rising seniority titles
    → soft penalty 0.35 (not a hard disqualifier).
    """
    history = candidate.get("career_history", [])
    if len(history) < 3:
        return _RuleResult(False, 0.0, None)

    short_hops = [
        role for role in history
        if (role.get("duration_months") or 0) < 18
    ]

    if len(short_hops) < 3:
        return _RuleResult(False, 0.0, None)

    # Check if seniority was *rising* across the short hops
    sorted_hops = sorted(
        short_hops,
        key=lambda r: _parse_date(r.get("start_date")) or date.min,
    )
    seniority_levels = [_title_seniority(r.get("title", "")) for r in sorted_hops]

    # Rising = last seniority level > first seniority level
    rising = seniority_levels[-1] > seniority_levels[0]

    if rising:
        hop_summary = ", ".join(
            f"{r.get('title', '?')} @ {r.get('company', '?')} ({r.get('duration_months', '?')}m)"
            for r in sorted_hops
        )
        return _RuleResult(
            False,
            0.35,
            f"Title-hopping pattern detected: {len(short_hops)} roles each under 18 months "
            f"with rising seniority. Signals optimising for title rather than depth. "
            f"Details: {hop_summary}.",
        )
    elif len(short_hops) >= 3:
        # Even without obvious rising seniority, 3+ short stints is a soft flag
        return _RuleResult(
            False,
            0.15,
            f"Frequent job-switching: {len(short_hops)} roles each under 18 months. "
            f"Candidate may not stay for the 3+ years this role requires.",
        )
    return _RuleResult(False, 0.0, None)


def _check_framework_enthusiast(candidate: dict) -> _RuleResult:
    """
    RULE 5: Framework enthusiast — heavy LangChain/hot-framework usage with no
    evidence of systems-level thinking → soft penalty 0.2.
    """
    career_text = _career_text(candidate)
    skills_text = _skills_text(candidate)
    combined = career_text + " " + skills_text

    langchain_hits = sum(1 for kw in LANGCHAIN_WRAPPER_KEYWORDS if kw in combined)
    if langchain_hits == 0:
        return _RuleResult(False, 0.0, None)

    # Systems thinking markers
    systems_keywords = [
        "distributed", "scalab", "latency", "throughput", "reliability",
        "fault toleran", "consistency", "sharding", "replication",
        "load balanc", "caching", "evaluation framework", "a/b test",
        "offline eval", "ndcg", "mrr", "map@", "precision@",
        "recall@", "embedding drift", "index refresh",
    ]
    has_systems_thinking = _contains_any(combined, systems_keywords)

    if langchain_hits >= 2 and not has_systems_thinking:
        return _RuleResult(
            False,
            0.2,
            "Framework-enthusiast signal: heavy use of LangChain / wrapper libraries "
            "with no evidence of systems-level design thinking (evaluation, latency, "
            "scalability, drift handling).",
        )
    return _RuleResult(False, 0.0, None)


def _check_consulting_only(candidate: dict) -> _RuleResult:
    """
    RULE 6: Entire career history at consulting firms (TCS, Infosys, Wipro,
    Accenture, Cognizant, Capgemini, etc.) with NO product-company experience
    → hard disqualify.

    If currently at consulting but has prior product-company experience → OK.
    """
    history = candidate.get("career_history", [])
    if not history:
        return _RuleResult(False, 0.0, None)

    all_consulting = all(_is_consulting_company(r.get("company", "")) for r in history)

    if all_consulting:
        firms = list({r.get("company", "") for r in history})
        return _RuleResult(
            True,
            1.0,
            f"Entire career at consulting/services firms only "
            f"({', '.join(firms)}) with no product-company experience. "
            "The JD explicitly calls this out as a fit issue.",
        )
    return _RuleResult(False, 0.0, None)


def _check_cv_speech_robotics_only(candidate: dict) -> _RuleResult:
    """
    RULE 7: Expertise primarily in CV / speech / robotics with no NLP / IR signal
    → hard disqualify.

    Looks across career descriptions + skills list.
    """
    career_text = _career_text(candidate)
    skills_text = _skills_text(candidate)
    combined = career_text + " " + skills_text

    has_nlp_ir = _contains_any(combined, NLP_IR_KEYWORDS)
    has_non_nlp = _contains_any(combined, NON_NLP_DOMAIN_KEYWORDS)

    if has_non_nlp and not has_nlp_ir:
        matched = [kw for kw in NON_NLP_DOMAIN_KEYWORDS if kw in combined]
        return _RuleResult(
            True,
            1.0,
            f"Expertise is primarily in non-NLP/IR domains "
            f"({', '.join(matched[:5])}) with no NLP/IR/search signal. "
            "Candidate would need to re-learn retrieval/ranking fundamentals.",
        )
    return _RuleResult(False, 0.0, None)


def _check_closed_source_isolation(candidate: dict) -> _RuleResult:
    """
    RULE 8: Worked entirely on closed-source proprietary systems for 5+ years
    with no external validation (papers, talks, open-source) → soft penalty 0.25.
    """
    history = candidate.get("career_history", [])
    years_exp = candidate.get("profile", {}).get("years_of_experience", 0) or 0

    if years_exp < 5:
        return _RuleResult(False, 0.0, None)

    # Markers of external validation
    external_validation_keywords = [
        "open-source contribution",
        "open source contribution",
        "open sourced",
        "github.com",
        "published paper",
        "published at",
        "arxiv.org",
        "arxiv preprint",
        "blog post",
        "tech talk",
        "gave a talk",
        "conference paper",
        "workshop paper",
        "kaggle competition",
        "kaggle notebook",
        "maintained open",
        "open source library",
        "contributed to open",
    ]
    career_text = _career_text(candidate)

    # Check github activity from redrob_signals
    github_score = candidate.get("redrob_signals", {}).get("github_activity_score", -1)
    has_github_activity = isinstance(github_score, (int, float)) and github_score > 10

    has_external_validation = (
        _contains_any(career_text, external_validation_keywords) or has_github_activity
    )

    # Check for closed/proprietary markers — must appear in role descriptions
    closed_keywords = [
        "proprietary",
        "confidential",
        "under nda",
        "closed-source",
        "internal tool",
        "internal platform",
        "internal ai",
        "no open-source",
        "enterprise b2b",
    ]
    closed_months = sum(
        (r.get("duration_months") or 0)
        for r in history
        if _contains_any(
            _text_lower(r.get("description", "")),
            closed_keywords,
        )
    )
    total_months = sum((r.get("duration_months") or 0) for r in history)

    # 5+ years of experience with heavy closed work and no external validation
    # Threshold: ≥40% of career is explicitly closed/proprietary (more sensitive than 60%)
    heavily_closed = (
        years_exp >= 5
        and total_months > 0
        and (closed_months / total_months) >= 0.4
    )

    if heavily_closed and not has_external_validation:
        return _RuleResult(
            False,
            0.25,
            f"Closed-source isolation: {years_exp:.1f} years of experience with "
            f"≥40% tenure in internal/proprietary projects and no external validation "
            "(no open-source, papers, talks, or GitHub activity > 10). "
            "We need to see how you think.",
        )
    return _RuleResult(False, 0.0, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_disqualifiers(candidate: dict) -> tuple[bool, float, list[str]]:
    """
    Evaluate all disqualifier rules for a candidate profile.

    Parameters
    ----------
    candidate : dict
        A single candidate record conforming to the Redrob candidate schema.

    Returns
    -------
    is_disqualified : bool
        True if at least one *hard* disqualifier fires.
    penalty : float
        Accumulated penalty in [0.0, 1.0].  1.0 means definitely out.
        Hard disqualifiers set this to 1.0; soft penalties are additive (capped).
    reasons : list[str]
        A human-readable explanation for every rule that triggered (both hard
        and soft).  Empty list → candidate passed all disqualifier checks.
    """
    checkers = [
        _check_pure_research,
        _check_langchain_only,
        _check_no_production_code_recent,
        _check_title_hopping,
        _check_framework_enthusiast,
        _check_consulting_only,
        _check_cv_speech_robotics_only,
        _check_closed_source_isolation,
    ]

    hard_fired = False
    total_penalty = 0.0
    reasons: list[str] = []

    for checker in checkers:
        result = checker(candidate)
        if result.reason:
            reasons.append(result.reason)
        if result.hard:
            hard_fired = True
            total_penalty = 1.0
        elif not hard_fired:
            total_penalty = min(1.0, total_penalty + result.penalty)

    # If any hard rule fired, clamp penalty to 1.0 regardless of soft penalties
    if hard_fired:
        total_penalty = 1.0

    return hard_fired, round(total_penalty, 4), reasons


# ---------------------------------------------------------------------------
# CLI demo — run against sample_candidates.json
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    sample_path = root / "data" / "raw" / "sample_candidates.json"

    if not sample_path.exists():
        print(f"[ERROR] Could not find {sample_path}", file=sys.stderr)
        sys.exit(1)

    with open(sample_path, encoding="utf-8") as fh:
        candidates: list[dict] = json.load(fh)

    print(f"Loaded {len(candidates)} candidates from {sample_path.name}\n")
    print("=" * 72)

    disqualified_count = 0
    for cand in candidates:
        cid = cand.get("candidate_id", "???")
        name = cand.get("profile", {}).get("anonymized_name", "N/A")
        is_dq, penalty, reasons = check_disqualifiers(cand)

        if is_dq or reasons:
            status = "[DISQUALIFIED]" if is_dq else "[SOFT PENALTY]"
            print(f"\n{status}  [{cid}] {name}")
            print(f"   Penalty : {penalty:.2f}")
            for i, r in enumerate(reasons, 1):
                print(f"   Reason {i}: {r}")
            if is_dq:
                disqualified_count += 1

    print("\n" + "=" * 72)
    total = len(candidates)
    print(
        f"\nSummary: {disqualified_count} hard-disqualified out of {total} candidates "
        f"({disqualified_count / total * 100:.1f}%)"
    )

