"""
src/reasoning_generator.py
--------------------------
Generates a concise 1-2 sentence human-readable reasoning string for a
ranked candidate.

Design constraints (strictly enforced)
---------------------------------------
1. ONLY use facts that are literally present in the candidate record.
   Never invent, infer, or hallucinate information not in the profile.
2. Pull from: years_of_experience, current_title, current_company,
   top matching skill (highest endorsements or proficiency), and one
   behavioral note derived from redrob_signals.
3. Output is a single clean string — no JSON, no markdown, no bullet points.
4. If a field is missing, omit gracefully rather than saying "unknown" or
   inventing a placeholder.

Public API
----------
  generate_reasoning(candidate: dict, scores: dict) -> str
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Proficiency rank used to pick the "top" skill when endorsements are equal
# ---------------------------------------------------------------------------
_PROFICIENCY_RANK: dict[str, int] = {
    "expert":       4,
    "advanced":     3,
    "intermediate": 2,
    "beginner":     1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _top_skill(candidate: dict[str, Any]) -> str | None:
    """
    Return the name of the candidate's most impressive skill.

    Priority order:
      1. Highest endorsements (most externally-validated)
      2. Highest proficiency level (as a tie-breaker)
      3. Longest duration_months (deepest practice)

    Returns None if no skills are present.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return None

    def _skill_key(s: dict) -> tuple[int, int, int]:
        endorsements = s.get("endorsements", 0) or 0
        proficiency  = _PROFICIENCY_RANK.get(
            (s.get("proficiency") or "").lower(), 0
        )
        duration = s.get("duration_months", 0) or 0
        return (endorsements, proficiency, duration)

    best = max(skills, key=_skill_key)
    return best.get("name") or None


def _behavioral_note(
    candidate: dict[str, Any],
    scores: dict[str, Any],
) -> str | None:
    """
    Derive one behavioral note from redrob_signals.

    Priority (first match wins — strongest signal first):
      1. open_to_work_flag = True           → "is actively open to new roles"
      2. recruiter_response_rate >= 0.8     → "has a high recruiter response rate"
      3. last_active_date (recent < 14d)    → "was recently active on the platform"
      4. notice_period_days <= 30           → "can join within 30 days"
      5. github_activity_score >= 60        → "has strong GitHub activity"
      6. interview_completion_rate >= 0.85  → "has a high interview completion rate"

    Returns None if no signal is strong enough to be noteworthy.
    """
    signals = candidate.get("redrob_signals", {}) or {}

    if signals.get("open_to_work_flag"):
        return "are actively open to new roles"

    rrr = signals.get("recruiter_response_rate")
    if isinstance(rrr, (int, float)) and rrr >= 0.8:
        return "have a high recruiter response rate"

    # Active in last 14 days — only mention if we know the date
    from datetime import date, datetime
    last_active_raw = signals.get("last_active_date")
    if last_active_raw:
        try:
            last_active = datetime.strptime(last_active_raw, "%Y-%m-%d").date()
            days_ago = (date.today() - last_active).days
            if 0 <= days_ago <= 14:
                return "were recently active on the platform"
        except ValueError:
            pass

    notice = signals.get("notice_period_days")
    if isinstance(notice, (int, float)) and 0 <= notice <= 30:
        days_label = "immediately" if notice == 0 else f"within {int(notice)} days"
        return f"can join {days_label}"

    gh = signals.get("github_activity_score")
    if isinstance(gh, (int, float)) and gh >= 60:
        return "have strong public GitHub activity"

    icr = signals.get("interview_completion_rate")
    if isinstance(icr, (int, float)) and icr >= 0.85:
        return "have a high interview completion rate"

    return None


def _format_experience(years: float | int | None) -> str | None:
    """Return a human-readable experience string, or None if missing/zero."""
    if years is None:
        return None
    years = float(years)
    if years <= 0:
        return None
    if years == int(years):
        label = f"{int(years)} year{'s' if years != 1 else ''}"
    else:
        label = f"{years:.1f} years"
    return label


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_reasoning(
    candidate: dict[str, Any],
    scores: dict[str, Any],
) -> str:
    """
    Generate a 1-2 sentence factual reasoning string for a ranked candidate.

    The reasoning is built purely from fields that are literally present in
    the candidate record.  No inference or hallucination is performed.

    Parameters
    ----------
    candidate : dict
        A single candidate record conforming to the Redrob candidate schema.
    scores : dict
        The scoring result dict returned by fusion_ranker.rank_candidate()
        (or rank_candidates_batch() element).  Used only to decide whether
        to mention disqualification or hard-floor status in the reasoning.

    Returns
    -------
    str
        A 1-2 sentence reasoning string. Always returns a non-empty string
        even if all profile fields are missing.
    """
    profile      = candidate.get("profile", {}) or {}
    candidate_id = candidate.get("candidate_id", "this candidate")

    yoe          = profile.get("years_of_experience")
    title        = (profile.get("current_title") or "").strip()
    company      = (profile.get("current_company") or "").strip()
    skill_name   = _top_skill(candidate)
    beh_note     = _behavioral_note(candidate, scores)

    is_dq        = scores.get("is_disqualified", False)
    dq_reasons   = scores.get("disqualifier_reasons", [])
    hp_score     = scores.get("honeypot_score", 0.0)
    hard_floored = scores.get("hard_floored", False)

    # ------------------------------------------------------------------
    # Case 1: Hard-floored due to honeypot — keep it brief and factual
    # ------------------------------------------------------------------
    if hp_score > 0.7 and hard_floored:
        return (
            f"Candidate {candidate_id} was flagged by automated consistency "
            f"checks and excluded from ranking consideration."
        )

    # ------------------------------------------------------------------
    # Case 2: Disqualified — surface the first disqualifier reason
    # ------------------------------------------------------------------
    if is_dq and hard_floored:
        first_reason = dq_reasons[0] if dq_reasons else "a hard disqualifier rule"
        # Truncate long reasons to keep the output readable
        if len(first_reason) > 120:
            first_reason = first_reason[:117] + "..."
        return (
            f"Candidate {candidate_id} did not meet a mandatory job requirement: "
            f"{first_reason}"
        )

    # ------------------------------------------------------------------
    # Case 3: Standard reasoning — build from profile fields
    # ------------------------------------------------------------------
    sentences: list[str] = []

    # --- Sentence 1: profile identity + top skill ---
    parts: list[str] = []

    exp_str = _format_experience(yoe)

    if title and company:
        if exp_str:
            parts.append(
                f"A {exp_str} professional currently serving as {title} at {company}"
            )
        else:
            parts.append(f"Currently serving as {title} at {company}")
    elif title:
        if exp_str:
            parts.append(f"A {exp_str} {title}")
        else:
            parts.append(f"A {title}")
    elif company:
        if exp_str:
            parts.append(f"A {exp_str} professional at {company}")
        else:
            parts.append(f"A professional at {company}")
    elif exp_str:
        parts.append(f"A professional with {exp_str} of experience")
    else:
        parts.append("This candidate")

    if skill_name:
        parts.append(f"with notable proficiency in {skill_name}")

    sentences.append(" ".join(parts) + ".")

    # --- Sentence 2: behavioral note ---
    if beh_note:
        sentences.append(f"They {beh_note}.")

    reasoning = " ".join(sentences)

    logger.debug("[%s] reasoning: %r", candidate_id, reasoning)
    return reasoning
