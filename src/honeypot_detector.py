"""
src/honeypot_detector.py
------------------------
Detects statistically impossible / fabricated ("honeypot") candidate profiles
by running a battery of consistency checks across a candidate dict that conforms
to the Redrob candidate schema.

Public API
----------
detect_honeypot(candidate: dict) -> tuple[float, list[str]]
    Returns
    -------
    honeypot_score : float [0.0 – 1.0]
        0.0  → nothing suspicious detected.
        1.0  → multiple high-confidence impossibilities; almost certainly fabricated.
        Scores accumulate additively and are capped at 1.0.
    triggered_checks : list[str]
        Human-readable description of every check that fired, with the evidence
        that triggered it.

Checks implemented
------------------
  CHECK-1  Expert skill claimed but duration_months < 12
           (takes years to reach expert-level proficiency).
  CHECK-2  Career history total duration significantly exceeds or falls far
           short of profile.years_of_experience (inflated / deflated CV).
  CHECK-3  A career_history entry implies tenure starting before the company
           could plausibly have existed based on context clues.
  CHECK-4  skill_assessment_scores wildly contradict stated proficiency
           (e.g., "expert" but score < 30, or "beginner" but score > 85).
  CHECK-5  profile.years_of_experience is inconsistent with the career
           timeline span derived from earliest start_date to today.

Each check returns (weight, reason_str | None).  Weights are tuned to
reflect severity — hard impossibilities score higher than soft anomalies.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Tuneable thresholds
# ---------------------------------------------------------------------------

# CHECK-1: Minimum months needed to reach a given proficiency level
PROFICIENCY_MIN_MONTHS: dict[str, int] = {
    "beginner":     0,
    "intermediate": 6,
    "advanced":     18,
    "expert":       36,   # < 36 months is suspicious; < 12 is highly suspicious
}

# Expert with < 12 months is almost impossible
EXPERT_HARD_MIN_MONTHS = 12
EXPERT_SOFT_MIN_MONTHS = 24   # < 24 months gets a lighter flag

# CHECK-2: Allowed ratio of (sum career months) / (years_of_experience * 12)
#   Overlapping roles can push total > 1.0, gaps can pull it < 1.0.
#   Thresholds beyond which we flag.
CAREER_DURATION_UPPER_RATIO = 2.0   # >200% of stated experience → suspicious
CAREER_DURATION_LOWER_RATIO = 0.35  # <35% of stated experience → suspicious

# Weights for suspicious total-duration cases
CAREER_DURATION_UPPER_WEIGHT = 0.35  # could be overlapping legitimate roles
CAREER_DURATION_LOWER_WEIGHT = 0.30  # unexplained gaps

# CHECK-3: Founding date proxies
# If a role's start_date is before the minimum plausible founding year of its
# company, flag it. We use a heuristic: if start_date < 2000 and company name
# contains strong signals of a modern tech company (cloud, SaaS, AI, etc.),
# or if the role start_date implies more tenure than the company could have had.
MODERN_TECH_SIGNALS = [
    "cloud", "saas", "ai", "ml", "deep learning", "gpt", "llm",
    "generative", "openai", "blockchain", "crypto", "nft", "web3",
    "fintech", "edtech", "healthtech", "proptech", "insurtech",
    "rideshare", "gig economy",
]
# Companies with known founding years (lowercase) — used for exact matching
KNOWN_COMPANY_FOUNDING: dict[str, int] = {
    "google": 1998,
    "facebook": 2004,
    "meta": 2021,
    "twitter": 2006,
    "x corp": 2023,
    "linkedin": 2002,
    "uber": 2009,
    "lyft": 2012,
    "airbnb": 2008,
    "stripe": 2010,
    "openai": 2015,
    "anthropic": 2021,
    "hugging face": 2016,
    "snowflake": 2012,
    "databricks": 2013,
    "confluent": 2014,
    "pinecone": 2019,
    "weaviate": 2019,
    "qdrant": 2021,
    "milvus": 2019,
    "zomato": 2008,
    "swiggy": 2014,
    "flipkart": 2007,
    "paytm": 2010,
    "razorpay": 2014,
    "cred": 2018,
    "meesho": 2015,
    "ola": 2010,
    "byju": 2011,
    "unacademy": 2010,
    "phonepe": 2015,
    "zepto": 2021,
    "blinkit": 2013,
    "groww": 2016,
    "zerodha": 2010,
}

# CHECK-4: proficiency → expected assessment score ranges
PROFICIENCY_SCORE_BOUNDS: dict[str, tuple[float, float]] = {
    # (min_plausible, max_plausible)
    "beginner":     (0.0,  50.0),
    "intermediate": (20.0, 80.0),
    "advanced":     (40.0, 100.0),
    "expert":       (60.0, 100.0),
}
# Below this score for "expert" → hard contradiction
EXPERT_SCORE_HARD_THRESHOLD = 30.0
# Above this score for "beginner" → contradiction
BEGINNER_SCORE_HARD_THRESHOLD = 85.0

# CHECK-5: Timeline span vs stated YoE tolerance
YOE_SPAN_TOLERANCE_MONTHS = 24   # Allow ±24 months (career gaps, part-time, etc.)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Check(NamedTuple):
    weight: float
    reason: str | None   # None means check did not fire


def _parse_date(date_str: str | None) -> date | None:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _months_between(d1: date, d2: date) -> int:
    """Signed months from d1 to d2 (positive if d2 > d1)."""
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_expert_skill_short_duration(candidate: dict) -> _Check:
    """
    CHECK-1 — Skill proficiency is 'expert' but duration_months < 12.

    An expert-level skill generally requires at least 2-3 years of deliberate
    practice.  Claiming expert status in under 12 months is implausible for
    most technical skills.

    Weight: 0.30 per hard violation (< 12 months), 0.15 per soft (12-24 months).
    """
    skills = candidate.get("skills", [])
    hard_violations: list[str] = []
    soft_violations: list[str] = []

    for skill in skills:
        proficiency = (skill.get("proficiency") or "").lower()
        duration = skill.get("duration_months", 0) or 0
        name = skill.get("name", "?")

        if proficiency == "expert":
            if duration < EXPERT_HARD_MIN_MONTHS:
                hard_violations.append(
                    f"'{name}' (expert, only {duration}mo experience)"
                )
            elif duration < EXPERT_SOFT_MIN_MONTHS:
                soft_violations.append(
                    f"'{name}' (expert, only {duration}mo experience)"
                )

    if not hard_violations and not soft_violations:
        return _Check(0.0, None)

    weight = min(
        0.30 * len(hard_violations) + 0.15 * len(soft_violations),
        0.55,   # cap contribution from this single check
    )
    parts = []
    if hard_violations:
        parts.append(
            f"HARD: expert proficiency with < {EXPERT_HARD_MIN_MONTHS}mo experience — "
            + ", ".join(hard_violations)
        )
    if soft_violations:
        parts.append(
            f"SOFT: expert proficiency with < {EXPERT_SOFT_MIN_MONTHS}mo experience — "
            + ", ".join(soft_violations)
        )
    return _Check(weight, "CHECK-1 (Expert skill, implausible duration): " + "; ".join(parts))


def _check_career_duration_mismatch(candidate: dict) -> _Check:
    """
    CHECK-2 — Sum of career_history duration_months vs profile.years_of_experience.

    Significant over-inflation: total career months >> stated YoE * 12.
    Significant under-reporting: total career months << stated YoE * 12.
    Note: some overlap between concurrent roles is legitimate; we allow a
    generous upper ratio.
    """
    history = candidate.get("career_history", [])
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0

    if yoe <= 0 or not history:
        return _Check(0.0, None)

    total_career_months = sum((r.get("duration_months") or 0) for r in history)
    expected_months = yoe * 12

    if expected_months == 0:
        return _Check(0.0, None)

    ratio = total_career_months / expected_months

    if ratio > CAREER_DURATION_UPPER_RATIO:
        excess = total_career_months - expected_months
        return _Check(
            CAREER_DURATION_UPPER_WEIGHT,
            f"CHECK-2 (Career duration inflation): total career history "
            f"({total_career_months}mo) is {ratio:.1f}x stated YoE "
            f"({yoe}y = {expected_months:.0f}mo expected). "
            f"Excess: {excess}mo beyond what concurrent roles alone can explain.",
        )

    if ratio < CAREER_DURATION_LOWER_RATIO:
        shortfall = expected_months - total_career_months
        return _Check(
            CAREER_DURATION_LOWER_WEIGHT,
            f"CHECK-2 (Career duration deflation): total career history "
            f"({total_career_months}mo) is only {ratio:.1f}x of stated YoE "
            f"({yoe}y = {expected_months:.0f}mo expected). "
            f"Unexplained gap of ~{shortfall:.0f}mo.",
        )

    return _Check(0.0, None)


def _check_impossible_tenure(candidate: dict) -> _Check:
    """
    CHECK-3 — Career entry implies tenure starting before the company plausibly
    existed.

    Two sub-checks:
      a) start_date vs KNOWN_COMPANY_FOUNDING lookup.
      b) start_date before 1990 combined with modern-tech language in
         the description (proxy for companies that couldn't have existed then).
      c) duration_months > 480 (claiming 40+ years at one place).
    """
    history = candidate.get("career_history", [])
    today = date.today()
    violations: list[str] = []

    for role in history:
        company = (role.get("company") or "").lower().strip()
        start_d = _parse_date(role.get("start_date"))
        duration = role.get("duration_months") or 0
        desc = (role.get("description") or "").lower()

        if start_d is None:
            continue

        # Sub-check a: known company founding year
        for known_company, founding_year in KNOWN_COMPANY_FOUNDING.items():
            if known_company in company:
                founding_date = date(founding_year, 1, 1)
                if start_d < founding_date:
                    months_before = _months_between(start_d, founding_date)
                    violations.append(
                        f"Role at '{role.get('company')}' starts {start_d} but "
                        f"company founded ~{founding_year} "
                        f"({months_before}mo before founding)"
                    )
                break

        # Sub-check b: pre-1990 start + modern tech signals in description
        if start_d.year < 1990:
            modern_hits = [kw for kw in MODERN_TECH_SIGNALS if kw in desc or kw in company]
            if modern_hits:
                violations.append(
                    f"Role at '{role.get('company')}' starts {start_d.year} "
                    f"but description mentions modern-tech concepts: "
                    f"{', '.join(modern_hits[:3])}"
                )
            elif start_d.year < 1970:
                # Implausibly old regardless of context
                violations.append(
                    f"Role at '{role.get('company')}' starts {start_d.year} — "
                    f"implausibly early for any tech career"
                )

        # Sub-check c: single role duration > 480 months (40 years)
        if duration > 480:
            violations.append(
                f"Role at '{role.get('company')}' claims {duration}mo "
                f"({duration / 12:.0f} years) tenure — exceeds a full career"
            )

        # Sub-check d: start_date in the future
        if start_d > today:
            months_future = _months_between(today, start_d)
            violations.append(
                f"Role at '{role.get('company')}' starts in the future "
                f"({start_d}, {months_future}mo from now)"
            )

    if not violations:
        return _Check(0.0, None)

    weight = min(0.40 * len(violations), 0.65)
    return _Check(
        weight,
        "CHECK-3 (Impossible tenure/company timeline): "
        + "; ".join(violations),
    )


def _check_assessment_vs_proficiency(candidate: dict) -> _Check:
    """
    CHECK-4 — skill_assessment_scores contradict stated skill proficiency.

    Hard contradictions (high weight):
      • proficiency = 'expert' but score < 30
      • proficiency = 'beginner' but score > 85

    Soft contradictions (lower weight):
      • proficiency = 'advanced' but score < 25
      • proficiency = 'intermediate' but score < 15 or > 90
    """
    skills = candidate.get("skills", [])
    scores: dict[str, float] = (
        candidate.get("redrob_signals", {}).get("skill_assessment_scores") or {}
    )

    if not scores:
        return _Check(0.0, None)

    hard_violations: list[str] = []
    soft_violations: list[str] = []

    for skill in skills:
        name = skill.get("name", "?")
        proficiency = (skill.get("proficiency") or "").lower()
        if name not in scores:
            continue
        score = float(scores[name])

        if proficiency == "expert":
            if score < EXPERT_SCORE_HARD_THRESHOLD:
                hard_violations.append(
                    f"'{name}': expert but score={score:.0f}/100 "
                    f"(threshold: >{EXPERT_SCORE_HARD_THRESHOLD:.0f})"
                )
            elif score < PROFICIENCY_SCORE_BOUNDS["expert"][0]:
                soft_violations.append(
                    f"'{name}': expert but score={score:.0f}/100"
                )

        elif proficiency == "beginner":
            if score > BEGINNER_SCORE_HARD_THRESHOLD:
                hard_violations.append(
                    f"'{name}': beginner but score={score:.0f}/100 "
                    f"(should be <{BEGINNER_SCORE_HARD_THRESHOLD:.0f})"
                )

        elif proficiency == "advanced":
            if score < 25:
                hard_violations.append(
                    f"'{name}': advanced but score={score:.0f}/100 "
                    f"(implausibly low for advanced level)"
                )

        elif proficiency == "intermediate":
            if score < 15:
                soft_violations.append(
                    f"'{name}': intermediate but score={score:.0f}/100"
                )
            elif score > 92:
                soft_violations.append(
                    f"'{name}': intermediate but score={score:.0f}/100 "
                    f"(unusually high for intermediate)"
                )

    if not hard_violations and not soft_violations:
        return _Check(0.0, None)

    weight = min(
        0.35 * len(hard_violations) + 0.15 * len(soft_violations),
        0.60,
    )
    parts = []
    if hard_violations:
        parts.append("HARD contradictions: " + ", ".join(hard_violations))
    if soft_violations:
        parts.append("SOFT contradictions: " + ", ".join(soft_violations))

    return _Check(
        weight,
        "CHECK-4 (Assessment vs proficiency contradiction): " + "; ".join(parts),
    )


def _check_yoe_vs_career_span(candidate: dict) -> _Check:
    """
    CHECK-5 — profile.years_of_experience vs career timeline span.

    Derives the earliest career start date from career_history and computes
    the span to today.  If this differs from stated YoE by more than the
    tolerance, flag it.

    Also checks for end_date on current role (is_current=True should have
    end_date=null) and duplicate overlapping roles with incompatible dates.
    """
    history = candidate.get("career_history", [])
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    today = date.today()
    violations: list[str] = []

    # --- Sub-check a: span from earliest start_date ---
    start_dates = []
    for role in history:
        d = _parse_date(role.get("start_date"))
        if d:
            start_dates.append(d)

    if start_dates and yoe > 0:
        earliest = min(start_dates)
        span_months = _months_between(earliest, today)
        yoe_months = yoe * 12
        diff_months = abs(span_months - yoe_months)

        if diff_months > YOE_SPAN_TOLERANCE_MONTHS + yoe_months * 0.20:
            direction = "longer" if span_months > yoe_months else "shorter"
            violations.append(
                f"Career span from earliest start ({earliest}) to today = "
                f"{span_months}mo, but stated YoE={yoe}y ({yoe_months:.0f}mo). "
                f"Span is {direction} by {diff_months:.0f}mo "
                f"(tolerance: {YOE_SPAN_TOLERANCE_MONTHS}mo)."
            )

    # --- Sub-check b: is_current=True but end_date is NOT null ---
    for role in history:
        if role.get("is_current") and role.get("end_date") is not None:
            violations.append(
                f"Role at '{role.get('company')}' marked is_current=True "
                f"but has end_date='{role.get('end_date')}' (should be null)"
            )

    # --- Sub-check c: overlapping roles with identical companies & titles ---
    sorted_roles = sorted(
        [r for r in history if _parse_date(r.get("start_date"))],
        key=lambda r: _parse_date(r.get("start_date")),
    )
    for i in range(len(sorted_roles) - 1):
        r1 = sorted_roles[i]
        r2 = sorted_roles[i + 1]
        r1_end = _parse_date(r1.get("end_date")) if not r1.get("is_current") else today
        r2_start = _parse_date(r2.get("start_date"))
        if r1_end and r2_start and r1_end < r2_start:
            gap = _months_between(r1_end, r2_start)
            # A large gap (>18 months) without explanation is suspicious
            if gap > 24 and yoe > 3:
                violations.append(
                    f"Unexplained gap of {gap}mo between "
                    f"'{r1.get('company')}/{r1.get('title')}' (ends {r1_end}) and "
                    f"'{r2.get('company')}/{r2.get('title')}' (starts {r2_start})"
                )

    if not violations:
        return _Check(0.0, None)

    weight = min(0.25 * len(violations), 0.50)
    return _Check(
        weight,
        "CHECK-5 (YoE vs career timeline inconsistency): "
        + "; ".join(violations),
    )


# ---------------------------------------------------------------------------
# Bonus checks (lightweight, lower weight)
# ---------------------------------------------------------------------------

def _check_implausible_salary(candidate: dict) -> _Check:
    """
    BONUS-A — Expected salary contradicts experience level.

    A freshers claiming 100+ LPA is suspicious.  A 15-year veteran claiming
    3 LPA is unusual but not impossible (career change).  We only flag
    the clearly impossible upper bound for experience level.
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    salary = candidate.get("redrob_signals", {}).get("expected_salary_range_inr_lpa", {})
    salary_max = salary.get("max", 0) or 0
    salary_min = salary.get("min", 0) or 0

    if salary_min > salary_max and salary_max > 0:
        return _Check(
            0.15,
            f"BONUS-A (Implausible salary range): min ({salary_min} LPA) > max ({salary_max} LPA)",
        )

    # Fresher (<1yr) claiming senior-engineer salary
    if yoe < 1.5 and salary_min > 50:
        return _Check(
            0.20,
            f"BONUS-A (Implausible salary for experience): "
            f"{yoe}y experience but expects min {salary_min} LPA "
            f"(typical senior-engineer range for <1.5yr profile)",
        )

    return _Check(0.0, None)


def _check_redrob_signal_consistency(candidate: dict) -> _Check:
    """
    BONUS-B — Internal Redrob signal contradictions.

    • interview_completion_rate = 1.0 but applications_submitted_30d = 0
      (no applications yet completion rate is 100%? Suspicious if verified).
    • offer_acceptance_rate > 1.0 or < -1.0 (out-of-range value).
    • profile_completeness_score > 100 or < 0.
    • avg_response_time_hours < 0.
    """
    signals = candidate.get("redrob_signals", {}) or {}
    violations: list[str] = []

    icr = signals.get("interview_completion_rate")
    if icr is not None and not (0.0 <= icr <= 1.0):
        violations.append(
            f"interview_completion_rate={icr} is outside [0, 1]"
        )

    oar = signals.get("offer_acceptance_rate")
    if oar is not None and not (-1.0 <= oar <= 1.0):
        violations.append(
            f"offer_acceptance_rate={oar} is outside [-1, 1]"
        )

    pcs = signals.get("profile_completeness_score")
    if pcs is not None and not (0.0 <= pcs <= 100.0):
        violations.append(
            f"profile_completeness_score={pcs} is outside [0, 100]"
        )

    art = signals.get("avg_response_time_hours")
    if art is not None and art < 0:
        violations.append(
            f"avg_response_time_hours={art} is negative"
        )

    rr = signals.get("recruiter_response_rate")
    if rr is not None and not (0.0 <= rr <= 1.0):
        violations.append(
            f"recruiter_response_rate={rr} is outside [0, 1]"
        )

    if not violations:
        return _Check(0.0, None)

    weight = min(0.20 * len(violations), 0.40)
    return _Check(
        weight,
        "BONUS-B (Redrob signal out-of-range): " + "; ".join(violations),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_honeypot(candidate: dict) -> tuple[float, list[str]]:
    """
    Detect statistically impossible or fabricated candidate profile signals.

    Parameters
    ----------
    candidate : dict
        A single candidate record conforming to the Redrob candidate schema.

    Returns
    -------
    honeypot_score : float [0.0, 1.0]
        Accumulated suspicion score. 0.0 = clean; 1.0 = very likely fabricated.
    triggered_checks : list[str]
        Human-readable description of each check that fired.
    """
    checkers = [
        _check_expert_skill_short_duration,
        _check_career_duration_mismatch,
        _check_impossible_tenure,
        _check_assessment_vs_proficiency,
        _check_yoe_vs_career_span,
        _check_implausible_salary,
        _check_redrob_signal_consistency,
    ]

    total_score = 0.0
    triggered: list[str] = []

    for checker in checkers:
        result = checker(candidate)
        if result.reason is not None:
            total_score += result.weight
            triggered.append(result.reason)

    total_score = round(min(total_score, 1.0), 4)
    return total_score, triggered


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

    flagged = 0
    for cand in candidates:
        cid = cand.get("candidate_id", "???")
        name = cand.get("profile", {}).get("anonymized_name", "N/A")
        score, checks = detect_honeypot(cand)

        if checks:
            status = "[HONEYPOT]" if score >= 0.5 else "[SUSPICIOUS]"
            print(f"\n{status}  [{cid}] {name}  score={score:.2f}")
            for i, c in enumerate(checks, 1):
                # Truncate long lines for readability
                short = c[:200] + "..." if len(c) > 200 else c
                print(f"   Check {i}: {short}")
            if score >= 0.5:
                flagged += 1

    print("\n" + "=" * 72)
    total = len(candidates)
    print(
        f"\nSummary: {flagged} likely-honeypots (score>=0.5) out of {total} candidates"
    )
