"""
src/behavioral_scorer.py
------------------------
Converts a candidate's redrob_signals dict into a single behavioural
engagement score in [0, 1].

Design philosophy (per redrob_signals_doc.docx):
  "A perfect-on-paper candidate who hasn't logged in for 6 months and has
   a 5% response rate is, for hiring purposes, not actually available."

This scorer is therefore a *reachability × reliability × readiness* signal,
NOT a skill-match signal. It is intended to act as a multiplier / modifier
on top of the semantic embedding similarity score inside the fusion ranker.

JD context
----------
  notice_period_preference_days = 30   (from jd_parsed.json)

All weights are documented inline with the rationale for each choice.
Changing a weight here propagates automatically to the fusion ranker.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JD-level constants (sourced from jd_parsed.json)
# ---------------------------------------------------------------------------
_JD_NOTICE_PERIOD_DAYS: int = 30   # JD prefers candidates who can join in ~30 days

# ---------------------------------------------------------------------------
# Component weights — must sum to 1.0
# The weights reflect how much each signal predicts *hireable reachability*
# rather than raw skill quality (which is handled by the embedding scorer).
# ---------------------------------------------------------------------------
_W = {
    "response_rate":         0.25,  # Strongest proxy for "will they actually talk to us?"
    "activity_recency":      0.25,  # Stale profiles = unavailable candidates; recency matters most alongside response
    "notice_period":         0.20,  # JD is explicit: prefer ≤30d notice; 90d+ = real hiring delay
    "interview_completion":  0.12,  # Reliable process behaviour — candidates who ghost interviews waste recruiter time
    "offer_acceptance":      0.08,  # Historical intent signal; high acceptors are lower flight-risk
    "github_activity":       0.05,  # Mild boost for visible public work; neutral if not linked (no penalty)
    "open_to_work":          0.05,  # Small boost — explicit availability declaration
}

assert abs(sum(_W.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ---------------------------------------------------------------------------
# Individual component scorers (each returns a float in [0, 1])
# ---------------------------------------------------------------------------

def _score_response_rate(signals: dict[str, Any]) -> float:
    """
    recruiter_response_rate is already 0-1, so use it directly.
    It is the single strongest indicator that a recruiter can actually
    reach this person; weight 0.25 reflects that.
    """
    val = signals.get("recruiter_response_rate", 0.5)
    return float(max(0.0, min(1.0, val)))


def _score_activity_recency(signals: dict[str, Any]) -> float:
    """
    last_active_date decay:
      < 30 days  → 1.0   (fully active)
      30–90 days → linear decay from 1.0 → 0.5
      90–180 days → linear decay from 0.5 → 0.1
      > 180 days → 0.05  (near-zero; candidate is effectively dark)

    A sharp cliff at 6 months (180 days) reflects the doc's observation
    that "not logged in for 6 months" ≈ not available.
    """
    raw = signals.get("last_active_date")
    if not raw:
        return 0.3  # Missing date → assume moderately stale; don't fully penalize

    try:
        last_active = datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        logger.warning("Unparseable last_active_date: %r", raw)
        return 0.3

    days_ago = (date.today() - last_active).days
    days_ago = max(0, days_ago)  # Guard against future dates in test data

    if days_ago < 30:
        return 1.0
    elif days_ago < 90:
        # Linear: 1.0 at day 30 → 0.5 at day 90
        return 1.0 - 0.5 * (days_ago - 30) / 60.0
    elif days_ago < 180:
        # Linear: 0.5 at day 90 → 0.1 at day 180
        return 0.5 - 0.4 * (days_ago - 90) / 90.0
    else:
        return 0.05


def _score_notice_period(signals: dict[str, Any]) -> float:
    """
    Score proximity to JD preference of 30 days.

    Scoring logic:
      ≤ 30 days  → 1.0   (ideal: can join immediately or within preference)
      30–60 days → linear decay 1.0 → 0.7  (acceptable; 1–2 months is common)
      60–90 days → linear decay 0.7 → 0.4  (long but negotiable)
      > 90 days  → hard penalised → 0.1    (doc: "penalize 90+ days"; hiring
                                             delay creates real business risk)

    We do not score 0 even at 180 days — the candidate might negotiate
    an early exit; we just mark it as low-confidence.
    """
    days = signals.get("notice_period_days", _JD_NOTICE_PERIOD_DAYS)
    days = int(max(0, days))

    if days <= _JD_NOTICE_PERIOD_DAYS:          # ≤ 30 days
        return 1.0
    elif days <= 60:
        return 1.0 - 0.3 * (days - _JD_NOTICE_PERIOD_DAYS) / 30.0
    elif days <= 90:
        return 0.7 - 0.3 * (days - 60) / 30.0
    else:
        # 90+ days: sharp drop, min floor of 0.1
        excess = days - 90
        penalty = min(0.3, 0.3 * excess / 90.0)  # additional decay up to 0.3
        return max(0.1, 0.4 - penalty)


def _score_github_activity(signals: dict[str, Any]) -> float:
    """
    github_activity_score is 0-100 or -1 (no GitHub linked).

    -1  → return 0.5 (neutral; no data, no penalty — many strong engineers
          don't maintain public GitHub activity)
    0   → 0.0  (linked but completely inactive)
    100 → 1.0

    Uses a mild log-curve so that the first ~30 points matter more than
    the last 30; prevents this from being gamed by commit-spamming.
    """
    val = signals.get("github_activity_score", -1)

    if val < 0:
        return 0.5  # Not linked; treat as neutral

    val = float(max(0.0, min(100.0, val)))
    # log1p curve: log(1+x)/log(101) maps [0,100] → [0,1], slightly concave
    return math.log1p(val) / math.log(101)


def _score_interview_completion(signals: dict[str, Any]) -> float:
    """
    interview_completion_rate: fraction of scheduled interviews attended.

    -1 is not a valid value per schema (range 0-1), but we treat any
    missing / None value as neutral (0.5) per the spec.

    High completion (>= 0.8) → strong reliability signal.
    Low completion (< 0.5)   → flag; candidates who ghost waste everyone's time.
    """
    val = signals.get("interview_completion_rate")
    if val is None or val == -1:
        return 0.5  # No history; neutral

    return float(max(0.0, min(1.0, val)))


def _score_offer_acceptance(signals: dict[str, Any]) -> float:
    """
    offer_acceptance_rate: -1 means no prior offer history.

    -1 → 0.5 (neutral; first-time or early-career candidates shouldn't be penalized)
    0.0 → 0.2  (never accepted an offer — possible serial-rejector risk)
    1.0 → 1.0  (always accepted)

    We compress the scale slightly at the bottom (floor 0.2) because a
    0.0 acceptance rate might reflect salary mismatches, not bad faith.
    """
    val = signals.get("offer_acceptance_rate", -1)

    if val < 0:
        return 0.5  # No offer history; neutral

    val = float(max(0.0, min(1.0, val)))
    # Remap [0,1] → [0.2, 1.0] to soften the floor
    return 0.2 + 0.8 * val


def _score_open_to_work(signals: dict[str, Any]) -> float:
    """
    open_to_work_flag: explicit availability declaration.

    True  → 1.0 (small boost — they've deliberately signalled availability)
    False → 0.3 (not a disqualifier; passive candidates are still reachable,
                 but we reward explicit intent)

    Weight 0.05 keeps this as a tie-breaker, not a gate.
    """
    flag = signals.get("open_to_work_flag", False)
    return 1.0 if flag else 0.3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_behavior(redrob_signals: dict[str, Any]) -> float:
    """
    Compute a single behavioural engagement score for a candidate.

    This score captures *reachability × reliability × readiness* — how
    likely the candidate is to actually engage with and complete a hiring
    process, independent of their skills. It is designed to be used as a
    modifier on top of semantic similarity scores in the fusion ranker.

    Parameters
    ----------
    redrob_signals : dict
        The ``redrob_signals`` object from a candidate profile, conforming
        to the candidate_schema.json specification.

    Returns
    -------
    float
        A score in [0, 1]. Higher = more behaviourally engaged and hireable.
        Scores below ~0.3 indicate candidates who are likely unreachable or
        unavailable despite potentially strong profiles.
    """
    components = {
        "response_rate":        _score_response_rate(redrob_signals),
        "activity_recency":     _score_activity_recency(redrob_signals),
        "notice_period":        _score_notice_period(redrob_signals),
        "interview_completion": _score_interview_completion(redrob_signals),
        "offer_acceptance":     _score_offer_acceptance(redrob_signals),
        "github_activity":      _score_github_activity(redrob_signals),
        "open_to_work":         _score_open_to_work(redrob_signals),
    }

    # Weighted sum — each component is already in [0, 1]
    score = sum(_W[k] * v for k, v in components.items())

    # Clamp to [0, 1] as a defensive guard against floating-point drift
    score = float(max(0.0, min(1.0, score)))

    logger.debug(
        "Behavioral score=%.4f  components=%s",
        score,
        {k: f"{v:.3f}" for k, v in components.items()},
    )
    return score


# ---------------------------------------------------------------------------
# Diagnostic helper (not used in production pipeline)
# ---------------------------------------------------------------------------

def score_behavior_verbose(redrob_signals: dict[str, Any]) -> dict[str, float]:
    """
    Same as score_behavior but returns a breakdown dict for debugging/explainability.

    Returns
    -------
    dict with keys:
        "total"              : final weighted score
        "response_rate"      : raw component score
        "activity_recency"   : raw component score
        "notice_period"      : raw component score
        "interview_completion": raw component score
        "offer_acceptance"   : raw component score
        "github_activity"    : raw component score
        "open_to_work"       : raw component score
    """
    components = {
        "response_rate":        _score_response_rate(redrob_signals),
        "activity_recency":     _score_activity_recency(redrob_signals),
        "notice_period":        _score_notice_period(redrob_signals),
        "interview_completion": _score_interview_completion(redrob_signals),
        "offer_acceptance":     _score_offer_acceptance(redrob_signals),
        "github_activity":      _score_github_activity(redrob_signals),
        "open_to_work":         _score_open_to_work(redrob_signals),
    }

    total = float(max(0.0, min(1.0, sum(_W[k] * v for k, v in components.items()))))
    return {"total": total, **components}
