"""
tests/test_honeypot_detector.py
--------------------------------
Pytest suite for src/honeypot_detector.py.

Structure
---------
• 5 synthetic *honeypot* (fabricated/impossible) candidate examples —
  each should trigger at least one check and return honeypot_score > 0.
• 5 synthetic *clean* candidate examples —
  each should return honeypot_score == 0.0 and no triggered_checks.
• Additional unit-level tests for each individual checker function.

Helpers
-------
_make_candidate() — builds a minimal valid candidate dict.
_role()           — builds a minimal career history entry.
_skill()          — builds a minimal skill entry.
"""

from __future__ import annotations

import pytest
from datetime import date, timedelta

from src.honeypot_detector import (
    detect_honeypot,
    _check_expert_skill_short_duration,
    _check_career_duration_mismatch,
    _check_impossible_tenure,
    _check_assessment_vs_proficiency,
    _check_yoe_vs_career_span,
    _check_implausible_salary,
    _check_redrob_signal_consistency,
)

# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

TODAY = date.today()


def _make_candidate(
    candidate_id: str = "CAND_TEST001",
    years_of_experience: float = 5.0,
    current_title: str = "ML Engineer",
    current_company: str = "ProductCo",
    career_history: list[dict] | None = None,
    skills: list[dict] | None = None,
    skill_assessment_scores: dict[str, float] | None = None,
    expected_salary_min: float = 20.0,
    expected_salary_max: float = 40.0,
    interview_completion_rate: float = 0.8,
    offer_acceptance_rate: float = 0.7,
    profile_completeness_score: float = 80.0,
    avg_response_time_hours: float = 24.0,
    recruiter_response_rate: float = 0.5,
) -> dict:
    """Return a minimal but fully valid candidate dict."""
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test User",
            "headline": "Engineer",
            "summary": "Professional summary.",
            "location": "Bangalore",
            "country": "India",
            "years_of_experience": years_of_experience,
            "current_title": current_title,
            "current_company": current_company,
            "current_company_size": "51-200",
            "current_industry": "Software",
        },
        "career_history": career_history or [],
        "education": [],
        "skills": skills or [],
        "redrob_signals": {
            "profile_completeness_score": profile_completeness_score,
            "signup_date": "2025-01-01",
            "last_active_date": TODAY.isoformat(),
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": recruiter_response_rate,
            "avg_response_time_hours": avg_response_time_hours,
            "skill_assessment_scores": skill_assessment_scores or {},
            "connection_count": 200,
            "endorsements_received": 10,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {
                "min": expected_salary_min,
                "max": expected_salary_max,
            },
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 40.0,
            "search_appearance_30d": 50,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": interview_completion_rate,
            "offer_acceptance_rate": offer_acceptance_rate,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


def _role(
    company: str = "ProductCo",
    title: str = "ML Engineer",
    description: str = "Built ML systems.",
    duration_months: int = 24,
    is_current: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    industry: str = "Software",
) -> dict:
    """Return a minimal career history role dict."""
    if start_date is None:
        sd = TODAY - timedelta(days=duration_months * 30)
        start_date = sd.isoformat()
    if end_date is None and not is_current:
        end_date = TODAY.isoformat()
    return {
        "company": company,
        "title": title,
        "start_date": start_date,
        "end_date": None if is_current else end_date,
        "duration_months": duration_months,
        "is_current": is_current,
        "industry": industry,
        "company_size": "51-200",
        "description": description,
    }


def _skill(
    name: str,
    proficiency: str = "intermediate",
    duration_months: int = 24,
    endorsements: int = 5,
) -> dict:
    return {
        "name": name,
        "proficiency": proficiency,
        "duration_months": duration_months,
        "endorsements": endorsements,
    }


# ===========================================================================
# HONEYPOT EXAMPLES  (5 fabricated/impossible profiles)
# ===========================================================================

class TestHoneypotExamples:
    """
    Five synthetic honeypot candidates.  Each should trigger at least one check
    and produce honeypot_score > 0.
    """

    def test_honeypot_1_expert_skills_in_weeks(self):
        """
        HONEYPOT-1: Candidate claims 'expert' proficiency in multiple skills
        with only 3-6 months of experience each.
        Triggers CHECK-1.
        """
        candidate = _make_candidate(
            years_of_experience=4.0,
            career_history=[
                _role(duration_months=48, is_current=True),
            ],
            skills=[
                _skill("Python", proficiency="expert", duration_months=4),
                _skill("PyTorch", proficiency="expert", duration_months=6),
                _skill("Kubernetes", proficiency="expert", duration_months=3),
                _skill("Elasticsearch", proficiency="expert", duration_months=5),
            ],
        )
        score, checks = detect_honeypot(candidate)
        assert score > 0.0, "Expert skills with 3-6mo experience should trigger a check"
        assert any("CHECK-1" in c for c in checks), "CHECK-1 should fire"
        # 4 hard violations × 0.30 weight each (capped at 0.55) → significant score
        assert score >= 0.30, f"Score should be meaningful, got {score}"

    def test_honeypot_2_inflated_career_duration(self):
        """
        HONEYPOT-2: Total career history months are 3× the stated YoE.
        A candidate claims 3 years (YoE=3.0) but lists 9+ years of roles.
        Triggers CHECK-2.
        """
        candidate = _make_candidate(
            years_of_experience=3.0,
            career_history=[
                _role(company="Alpha Corp",  duration_months=36, is_current=True),
                _role(company="Beta Inc",    duration_months=30),
                _role(company="Gamma Ltd",   duration_months=30),
                _role(company="Delta Co",    duration_months=15),
            ],
        )
        # Total = 111 months vs expected 36 months → ratio ~3.1
        score, checks = detect_honeypot(candidate)
        assert score > 0.0, "Inflated career duration should be detected"
        assert any("CHECK-2" in c for c in checks), "CHECK-2 should fire"

    def test_honeypot_3_impossible_tenure_before_founding(self):
        """
        HONEYPOT-3: Candidate claims to have worked at Stripe starting in 2005,
        but Stripe was founded in 2010.
        Triggers CHECK-3.
        """
        candidate = _make_candidate(
            years_of_experience=20.0,
            career_history=[
                _role(
                    company="Stripe",
                    title="Software Engineer",
                    description="Worked on payment infrastructure.",
                    duration_months=240,
                    start_date="2005-01-01",
                    end_date="2025-01-01",
                ),
            ],
        )
        score, checks = detect_honeypot(candidate)
        assert score > 0.0, "Pre-founding tenure should be detected"
        assert any("CHECK-3" in c for c in checks), "CHECK-3 should fire"
        assert any("Stripe" in c for c in checks), "Should mention the company"

    def test_honeypot_4_assessment_contradicts_expert_proficiency(self):
        """
        HONEYPOT-4: Candidate lists 'expert' NLP and 'expert' Python,
        but assessment scores are 18 and 22 respectively — far below
        what an expert should score.
        Triggers CHECK-4.
        """
        candidate = _make_candidate(
            years_of_experience=6.0,
            career_history=[
                _role(duration_months=72, is_current=True),
            ],
            skills=[
                _skill("NLP", proficiency="expert", duration_months=48),
                _skill("Python", proficiency="expert", duration_months=60),
                _skill("Elasticsearch", proficiency="advanced", duration_months=30),
            ],
            skill_assessment_scores={
                "NLP": 18.0,
                "Python": 22.0,
                "Elasticsearch": 15.0,  # advanced but score=15 (hard contradiction)
            },
        )
        score, checks = detect_honeypot(candidate)
        assert score > 0.0, "Assessment-proficiency contradiction should be detected"
        assert any("CHECK-4" in c for c in checks), "CHECK-4 should fire"
        assert score >= 0.35, f"Multiple hard contradictions should score high, got {score}"

    def test_honeypot_5_yoe_vs_span_gross_mismatch(self):
        """
        HONEYPOT-5: Candidate states YoE=15 years but their career history
        only spans 2 years from earliest start_date to today.
        Triggers CHECK-5.
        """
        start_2yr_ago = (TODAY - timedelta(days=730)).isoformat()
        end_1yr_ago   = (TODAY - timedelta(days=365)).isoformat()

        candidate = _make_candidate(
            years_of_experience=15.0,
            career_history=[
                _role(
                    company="RecentCo",
                    title="ML Lead",
                    description="Led ML team.",
                    duration_months=12,
                    start_date=start_2yr_ago,
                    end_date=end_1yr_ago,
                ),
                _role(
                    company="NewCo",
                    title="Staff Engineer",
                    description="Worked on infrastructure.",
                    duration_months=12,
                    is_current=True,
                    start_date=end_1yr_ago,
                ),
            ],
        )
        # Stated YoE=15yrs (180mo) but career only spans ~24mo
        score, checks = detect_honeypot(candidate)
        assert score > 0.0, "YoE vs career span mismatch should be detected"
        assert any("CHECK-5" in c for c in checks), "CHECK-5 should fire"

    # ------- Edge-case honeypot combinations -------

    def test_honeypot_6_multi_check_fabricated_profile(self):
        """
        HONEYPOT-6 (bonus): Multiple checks fire simultaneously —
        expert skills in 5 months AND assessment scores contradict them.
        Should produce a high combined score.
        """
        candidate = _make_candidate(
            years_of_experience=2.0,
            career_history=[
                _role(duration_months=24, is_current=True),
            ],
            skills=[
                _skill("BERT", proficiency="expert", duration_months=5),
                _skill("PyTorch", proficiency="expert", duration_months=7),
            ],
            skill_assessment_scores={
                "BERT":    12.0,
                "PyTorch": 9.0,
            },
        )
        score, checks = detect_honeypot(candidate)
        assert score >= 0.50, f"Multiple hard checks should yield score>=0.5, got {score}"
        assert len(checks) >= 2, "Should trigger at least 2 distinct checks"

    def test_honeypot_7_out_of_range_signals(self):
        """
        HONEYPOT-7 (bonus): Redrob signals contain out-of-range values —
        interview_completion_rate = 1.5, offer_acceptance_rate = 2.0.
        Triggers BONUS-B.
        """
        candidate = _make_candidate(
            years_of_experience=4.0,
            career_history=[_role(duration_months=48, is_current=True)],
            interview_completion_rate=1.5,
            offer_acceptance_rate=2.0,
        )
        score, checks = detect_honeypot(candidate)
        assert score > 0.0, "Out-of-range signals should be detected"
        assert any("BONUS-B" in c for c in checks), "BONUS-B should fire"


# ===========================================================================
# CLEAN EXAMPLES  (5 realistic, well-formed profiles)
# ===========================================================================

class TestCleanExamples:
    """
    Five synthetic clean candidates.  Each should return
    honeypot_score == 0.0 with no triggered_checks.
    """

    def test_clean_1_standard_ml_engineer(self):
        """
        CLEAN-1: A standard 5-year ML engineer.
        Consistent YoE, appropriate proficiency, matching assessment scores.
        """
        start_5yr = (TODAY - timedelta(days=5 * 365)).isoformat()
        start_2yr = (TODAY - timedelta(days=2 * 365)).isoformat()

        candidate = _make_candidate(
            years_of_experience=5.0,
            career_history=[
                _role(
                    company="Flipkart",
                    title="ML Engineer",
                    duration_months=36,
                    start_date=start_5yr,
                    end_date=start_2yr,
                ),
                _role(
                    company="Swiggy",
                    title="Senior ML Engineer",
                    duration_months=24,
                    is_current=True,
                    start_date=start_2yr,
                ),
            ],
            skills=[
                _skill("Python", proficiency="expert", duration_months=60),
                _skill("PyTorch", proficiency="advanced", duration_months=42),
                _skill("Elasticsearch", proficiency="intermediate", duration_months=30),
            ],
            skill_assessment_scores={
                "Python": 82.0,
                "PyTorch": 70.0,
                "Elasticsearch": 55.0,
            },
        )
        score, checks = detect_honeypot(candidate)
        assert score == 0.0, f"Clean candidate should score 0.0, got {score}. Checks: {checks}"
        assert checks == [], f"Should have no triggered checks, got: {checks}"

    def test_clean_2_senior_engineer_long_tenure(self):
        """
        CLEAN-2: A 10-year engineer with two long-tenure roles.
        Expert proficiency backed by 5+ years of usage.
        """
        start_10yr = (TODAY - timedelta(days=10 * 365)).isoformat()
        start_4yr  = (TODAY - timedelta(days=4  * 365)).isoformat()

        candidate = _make_candidate(
            years_of_experience=10.0,
            career_history=[
                _role(
                    company="Amazon",
                    title="Software Engineer",
                    duration_months=72,
                    start_date=start_10yr,
                    end_date=start_4yr,
                ),
                _role(
                    company="Microsoft",
                    title="Senior Engineer",
                    duration_months=48,
                    is_current=True,
                    start_date=start_4yr,
                ),
            ],
            skills=[
                _skill("Java", proficiency="expert", duration_months=90),
                _skill("AWS", proficiency="expert", duration_months=72),
                _skill("Distributed Systems", proficiency="advanced", duration_months=60),
            ],
            skill_assessment_scores={
                "Java": 88.0,
                "AWS": 76.0,
            },
        )
        score, checks = detect_honeypot(candidate)
        assert score == 0.0, f"Senior with long tenure should score 0.0, got {score}. Checks: {checks}"
        assert checks == []

    def test_clean_3_early_career_graduate(self):
        """
        CLEAN-3: 1.5-year candidate straight out of college.
        Beginner/intermediate proficiency, no assessments attempted.
        """
        start_1_5yr = (TODAY - timedelta(days=int(1.5 * 365))).isoformat()

        candidate = _make_candidate(
            years_of_experience=1.5,
            career_history=[
                _role(
                    company="StartupXYZ",
                    title="Junior Developer",
                    duration_months=18,
                    is_current=True,
                    start_date=start_1_5yr,
                ),
            ],
            skills=[
                _skill("Python", proficiency="intermediate", duration_months=18),
                _skill("SQL", proficiency="beginner", duration_months=12),
                _skill("React", proficiency="beginner", duration_months=8),
            ],
            skill_assessment_scores={},
            expected_salary_min=8.0,
            expected_salary_max=12.0,
        )
        score, checks = detect_honeypot(candidate)
        assert score == 0.0, f"Early-career clean profile should score 0.0, got {score}. Checks: {checks}"
        assert checks == []

    def test_clean_4_career_changer_with_gaps(self):
        """
        CLEAN-4: A career changer with a legitimate 12-month gap (MBA/study).
        The gap should NOT trigger CHECK-5's gap detection because it's < 24 months.
        """
        start_8yr  = (TODAY - timedelta(days=8 * 365)).isoformat()
        end_6yr    = (TODAY - timedelta(days=6 * 365)).isoformat()
        # Gap of ~12 months (MBA)
        start_5yr  = (TODAY - timedelta(days=5 * 365)).isoformat()
        start_2yr  = (TODAY - timedelta(days=2 * 365)).isoformat()

        candidate = _make_candidate(
            years_of_experience=7.0,
            career_history=[
                _role(
                    company="TechCo",
                    title="Software Engineer",
                    duration_months=24,
                    start_date=start_8yr,
                    end_date=end_6yr,
                ),
                _role(
                    company="ProductCo",
                    title="ML Engineer",
                    duration_months=36,
                    start_date=start_5yr,
                    end_date=start_2yr,
                ),
                _role(
                    company="AIStartup",
                    title="Senior ML Engineer",
                    duration_months=24,
                    is_current=True,
                    start_date=start_2yr,
                ),
            ],
            skills=[
                _skill("Python", proficiency="expert", duration_months=72),
                _skill("ML", proficiency="advanced", duration_months=48),
            ],
            skill_assessment_scores={
                "Python": 85.0,
                "ML": 68.0,
            },
        )
        score, checks = detect_honeypot(candidate)
        assert score == 0.0, f"Career changer with small gap should score 0.0, got {score}. Checks: {checks}"
        assert checks == []

    def test_clean_5_consultant_with_concurrent_projects(self):
        """
        CLEAN-5: Freelance consultant with slightly overlapping/concurrent client roles.
        Total career months may slightly exceed YoE × 12, but within the safe ratio.
        """
        start_6yr = (TODAY - timedelta(days=6 * 365)).isoformat()
        mid_3yr   = (TODAY - timedelta(days=3 * 365)).isoformat()

        candidate = _make_candidate(
            years_of_experience=6.0,
            career_history=[
                _role(
                    company="ClientAlpha",
                    title="NLP Consultant",
                    duration_months=36,
                    start_date=start_6yr,
                    end_date=mid_3yr,
                ),
                _role(
                    company="ClientBeta",
                    title="Search Engineer",
                    duration_months=36,
                    is_current=True,
                    start_date=mid_3yr,
                ),
            ],
            skills=[
                _skill("NLP", proficiency="expert", duration_months=60),
                _skill("Elasticsearch", proficiency="advanced", duration_months=48),
            ],
            skill_assessment_scores={
                "NLP": 79.0,
                "Elasticsearch": 62.0,
            },
        )
        # Total = 72 months, expected = 72 months → perfect ratio
        score, checks = detect_honeypot(candidate)
        assert score == 0.0, f"Consultant with clean history should score 0.0, got {score}. Checks: {checks}"
        assert checks == []


# ===========================================================================
# Unit tests for individual check functions
# ===========================================================================

class TestCheckExpertSkillShortDuration:
    """Unit tests for _check_expert_skill_short_duration (CHECK-1)."""

    def test_triggers_hard_violation_under_12_months(self):
        """Expert proficiency with only 6 months experience → hard flag."""
        candidate = _make_candidate(
            skills=[_skill("PyTorch", proficiency="expert", duration_months=6)]
        )
        result = _check_expert_skill_short_duration(candidate)
        assert result.weight > 0.0
        assert result.reason is not None
        assert "HARD" in result.reason

    def test_triggers_soft_violation_12_to_24_months(self):
        """Expert proficiency with 18 months experience → soft flag."""
        candidate = _make_candidate(
            skills=[_skill("BERT", proficiency="expert", duration_months=18)]
        )
        result = _check_expert_skill_short_duration(candidate)
        assert result.weight > 0.0
        assert result.reason is not None
        assert "SOFT" in result.reason

    def test_passes_expert_with_36_plus_months(self):
        """Expert with 36+ months → perfectly plausible, no flag."""
        candidate = _make_candidate(
            skills=[_skill("Python", proficiency="expert", duration_months=48)]
        )
        result = _check_expert_skill_short_duration(candidate)
        assert result.weight == 0.0
        assert result.reason is None

    def test_passes_non_expert_proficiency(self):
        """Intermediate/Advanced with any duration → not flagged by this check."""
        candidate = _make_candidate(
            skills=[
                _skill("SQL", proficiency="intermediate", duration_months=4),
                _skill("React", proficiency="advanced", duration_months=8),
            ]
        )
        result = _check_expert_skill_short_duration(candidate)
        assert result.weight == 0.0

    def test_weight_capped_with_many_violations(self):
        """Even with 10 expert-short skills, weight is capped at 0.55."""
        skills = [_skill(f"Skill{i}", proficiency="expert", duration_months=2) for i in range(10)]
        candidate = _make_candidate(skills=skills)
        result = _check_expert_skill_short_duration(candidate)
        assert result.weight <= 0.55


class TestCheckCareerDurationMismatch:
    """Unit tests for _check_career_duration_mismatch (CHECK-2)."""

    def test_triggers_inflation_over_2x(self):
        """Career months > 2× stated YoE → inflation flag."""
        candidate = _make_candidate(
            years_of_experience=2.0,
            career_history=[
                _role(duration_months=36),
                _role(duration_months=24),
                _role(duration_months=24),
            ],
        )
        # 84 months vs expected 24 months → ratio 3.5
        result = _check_career_duration_mismatch(candidate)
        assert result.weight > 0.0
        assert result.reason is not None
        assert "CHECK-2" in result.reason
        assert "inflation" in result.reason.lower()

    def test_triggers_deflation_under_35_percent(self):
        """Career months < 35% of stated YoE → deflation flag."""
        candidate = _make_candidate(
            years_of_experience=10.0,
            career_history=[
                _role(duration_months=30),   # only 30mo vs 120mo expected
            ],
        )
        result = _check_career_duration_mismatch(candidate)
        assert result.weight > 0.0
        assert "deflation" in result.reason.lower()

    def test_passes_reasonable_ratio(self):
        """Normal career with slight overlap → no flag."""
        candidate = _make_candidate(
            years_of_experience=5.0,
            career_history=[
                _role(duration_months=36),
                _role(duration_months=24),
            ],
        )
        # 60 months vs 60 expected → ratio exactly 1.0
        result = _check_career_duration_mismatch(candidate)
        assert result.weight == 0.0

    def test_passes_zero_yoe(self):
        """Zero YoE → division guard, no crash."""
        candidate = _make_candidate(
            years_of_experience=0.0,
            career_history=[_role(duration_months=6)],
        )
        result = _check_career_duration_mismatch(candidate)
        assert result.weight == 0.0


class TestCheckImpossibleTenure:
    """Unit tests for _check_impossible_tenure (CHECK-3)."""

    def test_triggers_known_company_pre_founding(self):
        """Role at Airbnb starting in 2004 — Airbnb founded 2008."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    company="Airbnb",
                    title="Engineer",
                    description="Worked on home-sharing platform.",
                    duration_months=60,
                    start_date="2004-01-01",
                    end_date="2009-01-01",
                )
            ]
        )
        result = _check_impossible_tenure(candidate)
        assert result.weight > 0.0
        assert result.reason is not None
        assert "CHECK-3" in result.reason

    def test_triggers_future_start_date(self):
        """Role starts in the future → impossible."""
        future_date = (date.today() + timedelta(days=365)).isoformat()
        candidate = _make_candidate(
            career_history=[
                _role(
                    company="FutureCo",
                    title="Engineer",
                    description="Future role.",
                    duration_months=12,
                    start_date=future_date,
                    end_date=None,
                    is_current=False,
                )
            ]
        )
        result = _check_impossible_tenure(candidate)
        assert result.weight > 0.0
        assert "future" in result.reason.lower()

    def test_passes_legitimate_tenure_at_known_company(self):
        """Role at Google starting 2015 — Google founded 1998, so fine."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    company="Google",
                    title="SWE",
                    description="Worked on search infrastructure.",
                    duration_months=36,
                    start_date="2015-03-01",
                    end_date="2018-03-01",
                )
            ]
        )
        result = _check_impossible_tenure(candidate)
        assert result.weight == 0.0

    def test_passes_unknown_company(self):
        """Unknown company with plausible dates → no flag."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    company="LocalStartup",
                    title="Engineer",
                    description="Built backend services.",
                    duration_months=24,
                    start_date="2020-01-01",
                    end_date="2022-01-01",
                )
            ]
        )
        result = _check_impossible_tenure(candidate)
        assert result.weight == 0.0


class TestCheckAssessmentVsProficiency:
    """Unit tests for _check_assessment_vs_proficiency (CHECK-4)."""

    def test_triggers_expert_low_score(self):
        """Expert proficiency but assessment score = 15 → hard contradiction."""
        candidate = _make_candidate(
            skills=[_skill("Python", proficiency="expert", duration_months=60)],
            skill_assessment_scores={"Python": 15.0},
        )
        result = _check_assessment_vs_proficiency(candidate)
        assert result.weight > 0.0
        assert "HARD" in result.reason

    def test_triggers_advanced_very_low_score(self):
        """Advanced proficiency but score = 12 → hard contradiction."""
        candidate = _make_candidate(
            skills=[_skill("SQL", proficiency="advanced", duration_months=36)],
            skill_assessment_scores={"SQL": 12.0},
        )
        result = _check_assessment_vs_proficiency(candidate)
        assert result.weight > 0.0

    def test_triggers_beginner_very_high_score(self):
        """Beginner proficiency but score = 92 → contradiction (under-claiming)."""
        candidate = _make_candidate(
            skills=[_skill("JavaScript", proficiency="beginner", duration_months=6)],
            skill_assessment_scores={"JavaScript": 92.0},
        )
        result = _check_assessment_vs_proficiency(candidate)
        assert result.weight > 0.0
        assert "HARD" in result.reason

    def test_passes_expert_high_score(self):
        """Expert proficiency and score = 85 → consistent."""
        candidate = _make_candidate(
            skills=[_skill("Python", proficiency="expert", duration_months=60)],
            skill_assessment_scores={"Python": 85.0},
        )
        result = _check_assessment_vs_proficiency(candidate)
        assert result.weight == 0.0

    def test_passes_no_assessment_scores(self):
        """No assessment data → no contradiction possible."""
        candidate = _make_candidate(
            skills=[_skill("Python", proficiency="expert", duration_months=60)],
            skill_assessment_scores={},
        )
        result = _check_assessment_vs_proficiency(candidate)
        assert result.weight == 0.0

    def test_passes_skill_not_in_assessment(self):
        """Skill listed but not assessed → no contradiction."""
        candidate = _make_candidate(
            skills=[
                _skill("Python", proficiency="expert", duration_months=60),
                _skill("Rust", proficiency="advanced", duration_months=24),
            ],
            skill_assessment_scores={"Python": 80.0},   # Rust not assessed
        )
        result = _check_assessment_vs_proficiency(candidate)
        assert result.weight == 0.0


class TestCheckYoeVsCareerSpan:
    """Unit tests for _check_yoe_vs_career_span (CHECK-5)."""

    def test_triggers_yoe_far_exceeds_span(self):
        """Stated YoE=12 but career only spans 2 years."""
        start_2yr = (TODAY - timedelta(days=2 * 365)).isoformat()
        candidate = _make_candidate(
            years_of_experience=12.0,
            career_history=[
                _role(
                    duration_months=24,
                    start_date=start_2yr,
                    is_current=True,
                )
            ],
        )
        result = _check_yoe_vs_career_span(candidate)
        assert result.weight > 0.0
        assert "CHECK-5" in result.reason

    def test_triggers_current_role_with_end_date(self):
        """is_current=True role but end_date is not null → inconsistency."""
        candidate = _make_candidate(
            years_of_experience=3.0,
            career_history=[
                {
                    "company": "BadCo",
                    "title": "Engineer",
                    "start_date": (TODAY - timedelta(days=365)).isoformat(),
                    "end_date": TODAY.isoformat(),  # should be null for current
                    "duration_months": 12,
                    "is_current": True,    # contradiction!
                    "industry": "Software",
                    "company_size": "51-200",
                    "description": "Worked here.",
                }
            ],
        )
        result = _check_yoe_vs_career_span(candidate)
        assert result.weight > 0.0
        assert "is_current=True" in result.reason

    def test_passes_consistent_yoe_and_span(self):
        """Stated YoE=5, career spans ~5 years → no flag."""
        start_5yr = (TODAY - timedelta(days=5 * 365)).isoformat()
        candidate = _make_candidate(
            years_of_experience=5.0,
            career_history=[
                _role(
                    duration_months=60,
                    start_date=start_5yr,
                    is_current=True,
                )
            ],
        )
        result = _check_yoe_vs_career_span(candidate)
        assert result.weight == 0.0

    def test_passes_within_tolerance(self):
        """Small discrepancy within ±24mo tolerance → no flag."""
        start_6yr = (TODAY - timedelta(days=6 * 365)).isoformat()
        candidate = _make_candidate(
            years_of_experience=5.0,
            career_history=[
                _role(
                    duration_months=60,
                    start_date=start_6yr,   # span = 72mo, stated=60mo → diff=12mo < tolerance
                    is_current=True,
                )
            ],
        )
        result = _check_yoe_vs_career_span(candidate)
        assert result.weight == 0.0


class TestCheckImplausibleSalary:
    """Unit tests for _check_implausible_salary (BONUS-A)."""

    def test_triggers_salary_min_greater_than_max(self):
        """min salary > max salary → impossible range."""
        candidate = _make_candidate(
            expected_salary_min=50.0,
            expected_salary_max=30.0,
        )
        result = _check_implausible_salary(candidate)
        assert result.weight > 0.0
        assert "BONUS-A" in result.reason

    def test_triggers_fresher_claiming_senior_salary(self):
        """<1.5 years experience but claiming 60 LPA minimum."""
        candidate = _make_candidate(
            years_of_experience=0.8,
            expected_salary_min=60.0,
            expected_salary_max=80.0,
        )
        result = _check_implausible_salary(candidate)
        assert result.weight > 0.0

    def test_passes_normal_salary_range(self):
        """Normal salary range for 5yr engineer → no flag."""
        candidate = _make_candidate(
            years_of_experience=5.0,
            expected_salary_min=20.0,
            expected_salary_max=40.0,
        )
        result = _check_implausible_salary(candidate)
        assert result.weight == 0.0


class TestCheckRedrobSignalConsistency:
    """Unit tests for _check_redrob_signal_consistency (BONUS-B)."""

    def test_triggers_out_of_range_interview_rate(self):
        """interview_completion_rate = 1.5 → out of [0, 1]."""
        candidate = _make_candidate(interview_completion_rate=1.5)
        result = _check_redrob_signal_consistency(candidate)
        assert result.weight > 0.0
        assert "BONUS-B" in result.reason

    def test_triggers_negative_response_time(self):
        """avg_response_time_hours = -5 → impossible."""
        candidate = _make_candidate(avg_response_time_hours=-5.0)
        result = _check_redrob_signal_consistency(candidate)
        assert result.weight > 0.0

    def test_triggers_out_of_range_offer_acceptance(self):
        """offer_acceptance_rate = 2.5 → out of [-1, 1]."""
        candidate = _make_candidate(offer_acceptance_rate=2.5)
        result = _check_redrob_signal_consistency(candidate)
        assert result.weight > 0.0

    def test_passes_all_valid_signals(self):
        """All signals within valid ranges → no flag."""
        candidate = _make_candidate(
            interview_completion_rate=0.8,
            offer_acceptance_rate=0.7,
            profile_completeness_score=85.0,
            avg_response_time_hours=24.0,
            recruiter_response_rate=0.5,
        )
        result = _check_redrob_signal_consistency(candidate)
        assert result.weight == 0.0


# ===========================================================================
# Integration tests for the public API
# ===========================================================================

class TestDetectHoneypotIntegration:
    """Tests for the top-level detect_honeypot() API."""

    def test_returns_correct_types(self):
        """Return types must be (float, list[str])."""
        candidate = _make_candidate()
        score, checks = detect_honeypot(candidate)
        assert isinstance(score, float)
        assert isinstance(checks, list)
        assert all(isinstance(c, str) for c in checks)

    def test_score_bounded_0_to_1(self):
        """Score must always be in [0.0, 1.0] even with many violations."""
        # Construct maximally suspicious candidate
        candidate = _make_candidate(
            years_of_experience=1.0,
            career_history=[
                _role(duration_months=200),
                _role(duration_months=200),
                _role(duration_months=200),
            ],
            skills=[
                _skill(f"Skill{i}", proficiency="expert", duration_months=2)
                for i in range(10)
            ],
            skill_assessment_scores={f"Skill{i}": 5.0 for i in range(10)},
            interview_completion_rate=5.0,
            offer_acceptance_rate=3.0,
            expected_salary_min=999.0,
            expected_salary_max=10.0,  # min > max
        )
        score, checks = detect_honeypot(candidate)
        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"

    def test_empty_candidate_does_not_crash(self):
        """Minimal candidate with no career/skills → graceful return."""
        candidate = _make_candidate(career_history=[], skills=[])
        score, checks = detect_honeypot(candidate)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_clean_profile_scores_zero(self):
        """A well-formed, internally consistent profile should score 0.0."""
        start_5yr = (TODAY - timedelta(days=5 * 365)).isoformat()
        candidate = _make_candidate(
            years_of_experience=5.0,
            career_history=[
                _role(duration_months=60, start_date=start_5yr, is_current=True)
            ],
            skills=[_skill("Python", proficiency="expert", duration_months=48)],
            skill_assessment_scores={"Python": 82.0},
        )
        score, checks = detect_honeypot(candidate)
        assert score == 0.0, f"Clean profile should score 0.0, got {score}: {checks}"

    def test_triggered_checks_non_empty_when_score_positive(self):
        """If score > 0, there must be at least one triggered check."""
        candidate = _make_candidate(
            skills=[_skill("Rust", proficiency="expert", duration_months=3)]
        )
        score, checks = detect_honeypot(candidate)
        if score > 0.0:
            assert len(checks) > 0, "Positive score must have at least one reason"
