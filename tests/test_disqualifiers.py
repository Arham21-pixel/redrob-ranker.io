"""
tests/test_disqualifiers.py
---------------------------
Pytest suite for src/disqualifiers.py.

Coverage: at least 2 test cases per disqualifier rule — one that triggers the
rule (should_trigger=True) and one that passes cleanly (should_trigger=False).

Rule index (mirrors disqualifiers.py docstring):
    1  _check_pure_research
    2  _check_langchain_only
    3  _check_no_production_code_recent
    4  _check_title_hopping
    5  _check_framework_enthusiast
    6  _check_consulting_only
    7  _check_cv_speech_robotics_only
    8  _check_closed_source_isolation

All public API is tested through check_disqualifiers() to ensure integration;
individual helpers are also imported for focused unit-tests.
"""

from __future__ import annotations

import pytest
from datetime import date, timedelta

from src.disqualifiers import (
    check_disqualifiers,
    _check_pure_research,
    _check_langchain_only,
    _check_no_production_code_recent,
    _check_title_hopping,
    _check_framework_enthusiast,
    _check_consulting_only,
    _check_cv_speech_robotics_only,
    _check_closed_source_isolation,
)

# ---------------------------------------------------------------------------
# Helpers to build minimal candidate dicts
# ---------------------------------------------------------------------------

def _make_candidate(
    candidate_id: str = "CAND_0000000",
    years_of_experience: float = 5.0,
    current_title: str = "ML Engineer",
    current_company: str = "ProductCo",
    current_industry: str = "Software",
    career_history: list[dict] | None = None,
    skills: list[dict] | None = None,
    github_activity_score: float = -1,
) -> dict:
    """Return a minimal candidate dict suitable for check_disqualifiers()."""
    today_str = date.today().isoformat()
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "Engineer",
            "summary": "Professional summary.",
            "location": "Bangalore",
            "country": "India",
            "years_of_experience": years_of_experience,
            "current_title": current_title,
            "current_company": current_company,
            "current_company_size": "51-200",
            "current_industry": current_industry,
        },
        "career_history": career_history or [],
        "education": [],
        "skills": skills or [],
        "redrob_signals": {
            "profile_completeness_score": 80.0,
            "signup_date": "2025-01-01",
            "last_active_date": today_str,
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.5,
            "avg_response_time_hours": 24.0,
            "skill_assessment_scores": {},
            "connection_count": 200,
            "endorsements_received": 10,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20.0, "max": 40.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": github_activity_score,
            "search_appearance_30d": 50,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.7,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


def _role(
    company: str = "ProductCo",
    title: str = "ML Engineer",
    description: str = "Built and deployed production models.",
    duration_months: int = 24,
    is_current: bool = False,
    industry: str = "Software",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Return a minimal career history role dict."""
    today = date.today()
    if start_date is None:
        start_d = today - timedelta(days=duration_months * 30)
        start_date = start_d.isoformat()
    if end_date is None and not is_current:
        end_d = today - timedelta(days=1)
        end_date = end_d.isoformat()
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


# ===========================================================================
# RULE 1 — Pure research background, no production deployment
# ===========================================================================

class TestPureResearch:
    """_check_pure_research"""

    def test_triggers_when_all_roles_are_academic_research(self):
        """Every role is in academic labs with zero production signal → HARD disqualify."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    company="MIT AI Lab",
                    title="Research Scientist",
                    description=(
                        "Conducted research on deep learning architectures. Published 3 papers "
                        "at top NLP conferences (ACL, EMNLP). Worked in the academic lab "
                        "setting. Everything was theoretical or prototype; nothing shipped to real users."
                    ),
                    industry="Research",
                ),
                _role(
                    company="IISc",
                    title="PhD Research Fellow",
                    description=(
                        "PhD research on graph neural networks. University lab work. "
                        "Publications in IEEE journals. No ML pipeline or serving work."
                    ),
                    industry="Research",
                ),
            ]
        )
        result = _check_pure_research(candidate)
        assert result.hard is True, "Should hard-disqualify pure research profile"
        assert result.penalty == 1.0
        assert result.reason is not None

    def test_passes_when_at_least_one_production_role(self):
        """Has a production role alongside a research role → should NOT trigger."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    company="Google",
                    title="Software Engineer, ML",
                    description=(
                        "Deployed embedding-based retrieval models to production serving "
                        "real users. Handled model inference, A/B testing, and monitoring."
                    ),
                ),
                _role(
                    company="Stanford NLP Group",
                    title="Research Intern",
                    description=(
                        "Research internship studying transformer architectures. "
                        "Published workshop paper."
                    ),
                ),
            ]
        )
        result = _check_pure_research(candidate)
        assert result.hard is False, "Should NOT disqualify when production experience exists"

    def test_passes_when_no_research_keywords_present(self):
        """Candidate has only industry / product roles → rule must not fire."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    title="Backend Engineer",
                    description="Built and deployed REST APIs. Production code shipped weekly.",
                ),
            ]
        )
        result = _check_pure_research(candidate)
        assert result.hard is False


# ===========================================================================
# RULE 2 — LangChain-only AI experience, no pre-LLM production ML
# ===========================================================================

class TestLangchainOnly:
    """_check_langchain_only"""

    def test_triggers_recent_langchain_no_prellm(self):
        """Only 6 months of LangChain work, no pre-LLM production experience."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    title="AI Developer",
                    description=(
                        "Built chatbots using LangChain and OpenAI API. "
                        "Created RAG pipelines with LangChain document loaders."
                    ),
                    duration_months=6,
                ),
            ]
        )
        result = _check_langchain_only(candidate)
        assert result.hard is True, "Short LangChain-only stint with no pre-LLM background → disqualify"

    def test_passes_with_substantial_prellm_production(self):
        """Has 3 years of production retrieval/ranking BEFORE LangChain hype → passes."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    title="Search Engineer",
                    description=(
                        "Built and deployed Elasticsearch-based search ranking system "
                        "serving 5M queries/day. Designed BM25 + learning-to-rank pipeline. "
                        "Owned production ML pipeline from training to inference."
                    ),
                    duration_months=36,
                ),
                _role(
                    title="AI Developer",
                    description=(
                        "Experimenting with LangChain for internal chatbot prototype."
                    ),
                    duration_months=8,
                ),
            ]
        )
        result = _check_langchain_only(candidate)
        assert result.hard is False, "Pre-LLM production experience should neutralise LangChain-only concern"

    def test_passes_when_langchain_experience_is_long(self):
        """12+ months of LangChain use is not flagged (under-12-month rule)."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    title="ML Engineer",
                    description="Used LangChain and llamaindex for production document search.",
                    duration_months=18,
                ),
            ]
        )
        result = _check_langchain_only(candidate)
        assert result.hard is False, "12+ months of LangChain usage is above the threshold"


# ===========================================================================
# RULE 3 — No production code in last 18 months (arch/lead-only senior)
# ===========================================================================

class TestNoProductionCodeRecent:
    """_check_no_production_code_recent"""

    def test_triggers_senior_architect_no_coding(self):
        """
        Senior candidate (7 yrs) whose current role is 'Solutions Architect'
        with no coding evidence in the description.
        """
        today = date.today().isoformat()
        candidate = _make_candidate(
            years_of_experience=7.0,
            current_title="Solutions Architect",
            career_history=[
                _role(
                    title="Solutions Architect",
                    description=(
                        "Responsible for defining the technology roadmap and vendor selection. "
                        "Conducted architecture reviews and stakeholder presentations. "
                        "No hands-on coding involved."
                    ),
                    duration_months=20,
                    is_current=True,
                ),
            ],
        )
        result = _check_no_production_code_recent(candidate)
        assert result.hard is True, "Senior architect with no coding signal should be disqualified"
        assert result.penalty == 1.0

    def test_passes_architect_who_still_codes(self):
        """Architect role but description explicitly mentions implementing production code."""
        candidate = _make_candidate(
            years_of_experience=8.0,
            current_title="Principal Architect",
            career_history=[
                _role(
                    title="Principal Architect",
                    description=(
                        "Architected the ML serving platform AND implemented the core "
                        "inference engine in Python. Wrote the embedding pipeline code, "
                        "shipped to production with load tests."
                    ),
                    duration_months=18,
                    is_current=True,
                ),
            ],
        )
        result = _check_no_production_code_recent(candidate)
        assert result.hard is False, "Architect who still writes production code should pass"

    def test_passes_junior_architect(self):
        """Under 5 yrs experience — rule only applies to senior engineers."""
        candidate = _make_candidate(
            years_of_experience=3.0,
            current_title="Tech Lead",
            career_history=[
                _role(
                    title="Tech Lead",
                    description="Led architecture discussions. No production coding.",
                    duration_months=12,
                    is_current=True,
                ),
            ],
        )
        result = _check_no_production_code_recent(candidate)
        assert result.hard is False, "Junior candidate with arch title should not be disqualified by this rule"

    def test_passes_senior_with_coding_role(self):
        """Senior engineer in a regular SWE role with clear coding evidence."""
        candidate = _make_candidate(
            years_of_experience=6.0,
            current_title="Senior ML Engineer",
            career_history=[
                _role(
                    title="Senior ML Engineer",
                    description=(
                        "Implemented and deployed retrieval models. Wrote production "
                        "inference code in Python. Committed daily to the main codebase."
                    ),
                    duration_months=24,
                    is_current=True,
                ),
            ],
        )
        result = _check_no_production_code_recent(candidate)
        assert result.hard is False


# ===========================================================================
# RULE 4 — Title-hopping with rising seniority
# ===========================================================================

class TestTitleHopping:
    """_check_title_hopping"""

    def test_triggers_three_short_stints_rising_seniority(self):
        """3 jobs each < 18 months, escalating titles → soft penalty."""
        today = date.today()
        candidate = _make_candidate(
            career_history=[
                _role(
                    title="Junior Engineer",
                    description="Entry-level engineering work.",
                    duration_months=10,
                    start_date=(today - timedelta(days=48 * 30)).isoformat(),
                    end_date=(today - timedelta(days=38 * 30)).isoformat(),
                ),
                _role(
                    title="Senior Engineer",
                    description="Moved to a more senior role.",
                    duration_months=12,
                    start_date=(today - timedelta(days=36 * 30)).isoformat(),
                    end_date=(today - timedelta(days=24 * 30)).isoformat(),
                ),
                _role(
                    title="Staff Engineer",
                    description="Staff-level work.",
                    duration_months=14,
                    start_date=(today - timedelta(days=22 * 30)).isoformat(),
                    end_date=(today - timedelta(days=8 * 30)).isoformat(),
                ),
            ]
        )
        result = _check_title_hopping(candidate)
        assert result.hard is False, "Title-hopping is a soft penalty, not hard disqualify"
        assert result.penalty > 0.0, "Should accumulate a penalty for title-hopping"
        assert result.reason is not None

    def test_passes_long_tenure_roles(self):
        """All roles are 24+ months → no title-hopping concern."""
        candidate = _make_candidate(
            career_history=[
                _role(title="Engineer", description="Built systems.", duration_months=30),
                _role(title="Senior Engineer", description="Led ML work.", duration_months=36),
            ]
        )
        result = _check_title_hopping(candidate)
        assert result.penalty == 0.0, "Long-tenure roles should have zero title-hop penalty"

    def test_passes_fewer_than_three_short_stints(self):
        """Only 2 short stints — threshold is 3."""
        candidate = _make_candidate(
            career_history=[
                _role(title="Junior Engineer", description="Entry role.", duration_months=12),
                _role(title="Senior Engineer", description="Next role.", duration_months=14),
            ]
        )
        result = _check_title_hopping(candidate)
        assert result.penalty == 0.0


# ===========================================================================
# RULE 5 — Framework enthusiast (LangChain tutorials, no systems thinking)
# ===========================================================================

class TestFrameworkEnthusiast:
    """_check_framework_enthusiast"""

    def test_triggers_heavy_langchain_no_systems_thinking(self):
        """Two+ LangChain references, zero mention of evaluation/latency/scaling."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    title="AI Developer",
                    description=(
                        "Built several LangChain applications. Created LangChain + "
                        "llamaindex demos for customers. Wrote tutorials on LangChain "
                        "for internal blog."
                    ),
                )
            ],
            skills=[
                {"name": "LangChain", "proficiency": "advanced", "endorsements": 10, "duration_months": 12},
            ],
        )
        result = _check_framework_enthusiast(candidate)
        assert result.hard is False, "Framework enthusiast is a soft penalty only"
        assert result.penalty > 0.0, "Should have a non-zero penalty"

    def test_passes_langchain_plus_systems_thinking(self):
        """Uses LangChain but also demonstrates evaluation frameworks and latency concerns."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    title="ML Engineer",
                    description=(
                        "Used LangChain for prototyping but moved to custom retrieval stack. "
                        "Built offline evaluation framework using NDCG and MRR. Handled "
                        "low-latency inference with caching. Ran A/B tests."
                    ),
                )
            ]
        )
        result = _check_framework_enthusiast(candidate)
        assert result.penalty == 0.0, "Systems-thinking candidate should not get framework-enthusiast penalty"

    def test_passes_no_langchain_at_all(self):
        """No LangChain usage → rule is irrelevant."""
        candidate = _make_candidate(
            career_history=[
                _role(title="ML Engineer", description="Built retrieval systems using Elasticsearch."),
            ]
        )
        result = _check_framework_enthusiast(candidate)
        assert result.penalty == 0.0


# ===========================================================================
# RULE 6 — Entire career at consulting firms only
# ===========================================================================

class TestConsultingOnly:
    """_check_consulting_only"""

    def test_triggers_all_consulting_firms(self):
        """Every role is at a named consulting firm → HARD disqualify."""
        candidate = _make_candidate(
            career_history=[
                _role(company="TCS", title="Software Engineer", description="Client delivery work."),
                _role(company="Infosys", title="Senior Engineer", description="Consulting project work."),
                _role(company="Wipro", title="Tech Lead", description="Managed client delivery."),
            ]
        )
        result = _check_consulting_only(candidate)
        assert result.hard is True, "All-consulting career should be a hard disqualifier"
        assert result.penalty == 1.0

    def test_passes_mixed_consulting_and_product_company(self):
        """Currently at Accenture but has prior product-company experience → passes."""
        candidate = _make_candidate(
            career_history=[
                _role(company="Accenture", title="Consultant", description="Consulting delivery."),
                _role(company="Razorpay", title="ML Engineer", description="Built production ML models."),
            ]
        )
        result = _check_consulting_only(candidate)
        assert result.hard is False, "Prior product-company experience should save this candidate"

    def test_passes_no_consulting_at_all(self):
        """Pure product-company career → rule must not fire."""
        candidate = _make_candidate(
            career_history=[
                _role(company="Swiggy", title="Data Scientist", description="Recommendation systems."),
                _role(company="Zomato", title="ML Engineer", description="Search ranking."),
            ]
        )
        result = _check_consulting_only(candidate)
        assert result.hard is False

    def test_triggers_single_consulting_firm_entire_career(self):
        """Entire career at a single consulting giant."""
        candidate = _make_candidate(
            career_history=[
                _role(company="Cognizant", title="Junior Dev", description="Client project."),
                _role(company="Cognizant", title="Senior Dev", description="Same company, different project."),
            ]
        )
        result = _check_consulting_only(candidate)
        assert result.hard is True


# ===========================================================================
# RULE 7 — CV / speech / robotics only, no NLP / IR signal
# ===========================================================================

class TestCvSpeechRoboticsOnly:
    """_check_cv_speech_robotics_only"""

    def test_triggers_pure_computer_vision_no_nlp(self):
        """Career and skills are entirely CV (image classification, object detection), no NLP."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    title="Computer Vision Engineer",
                    description=(
                        "Worked on image classification and object detection systems. "
                        "Deployed YOLO-based detection pipeline for defect inspection. "
                        "No language work whatsoever, zero IR or retrieval involvement."
                    ),
                )
            ],
            skills=[
                {"name": "Image Classification", "proficiency": "expert", "endorsements": 50, "duration_months": 36},
                {"name": "Object Detection", "proficiency": "advanced", "endorsements": 30, "duration_months": 24},
            ],
        )
        result = _check_cv_speech_robotics_only(candidate)
        assert result.hard is True, "Pure CV background without NLP/IR → hard disqualify"
        assert result.penalty == 1.0

    def test_triggers_pure_speech_recognition_no_nlp(self):
        """Speech ASR / TTS expert with no NLP or search work."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    title="Speech Recognition Scientist",
                    description=(
                        "Built ASR models for voice interfaces. Speech synthesis and TTS pipelines. "
                        "All work in acoustic modelling, speaker recognition, and audio signal processing."
                    ),
                )
            ],
            skills=[
                {"name": "Speech Recognition", "proficiency": "expert", "endorsements": 40, "duration_months": 36},
                {"name": "TTS", "proficiency": "advanced", "endorsements": 20, "duration_months": 24},
            ],
        )
        result = _check_cv_speech_robotics_only(candidate)
        assert result.hard is True

    def test_passes_cv_engineer_with_nlp_experience(self):
        """CV background BUT also has NLP/search work → should pass."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    title="ML Engineer",
                    description=(
                        "Worked on image classification AND text classification. "
                        "Built an NLP pipeline for document retrieval. "
                        "Semantic search using sentence-transformers."
                    ),
                )
            ],
            skills=[
                {"name": "Image Classification", "proficiency": "advanced", "endorsements": 20, "duration_months": 24},
                {"name": "NLP", "proficiency": "advanced", "endorsements": 30, "duration_months": 18},
            ],
        )
        result = _check_cv_speech_robotics_only(candidate)
        assert result.hard is False, "Combined CV + NLP background should not be disqualified"

    def test_passes_robotics_with_nlp_signal(self):
        """Robotics + dialogue systems → NLP presence saves the candidate."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    title="Robotics and NLP Engineer",
                    description=(
                        "Worked on ROS-based robotics AND natural language understanding "
                        "for human-robot interaction. Information retrieval for command parsing."
                    ),
                )
            ]
        )
        result = _check_cv_speech_robotics_only(candidate)
        assert result.hard is False

    def test_passes_no_cv_speech_domain_at_all(self):
        """Pure backend/search engineer with no CV or speech."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    title="Search Engineer",
                    description="Built Elasticsearch-based search with NLP query rewriting.",
                )
            ]
        )
        result = _check_cv_speech_robotics_only(candidate)
        assert result.hard is False


# ===========================================================================
# RULE 8 — Closed-source isolation for 5+ years with no external validation
# ===========================================================================

class TestClosedSourceIsolation:
    """_check_closed_source_isolation"""

    def test_triggers_long_proprietary_work_no_validation(self):
        """5+ years of internal/proprietary projects with no GitHub, papers, or open-source."""
        candidate = _make_candidate(
            years_of_experience=7.0,
            github_activity_score=2,  # very low GitHub activity
            career_history=[
                _role(
                    title="Senior Engineer",
                    description=(
                        "Worked on internal platform under NDA (confidential). "
                        "Internal AI tools for enterprise use. Enterprise B2B product. "
                        "All work is internal and closed."
                    ),
                    duration_months=50,
                ),
                _role(
                    title="ML Engineer",
                    description=(
                        "Built internal AI tools for the enterprise. "
                        "All work is confidential and proprietary. "
                        "Everything kept private, nothing released externally."
                    ),
                    duration_months=35,
                ),
            ],
        )
        result = _check_closed_source_isolation(candidate)
        assert result.hard is False, "Closed-source isolation is a soft penalty only"
        assert result.penalty > 0.0, "Should apply a soft penalty"

    def test_passes_closed_source_with_open_source_contributions(self):
        """Proprietary work but has notable open-source contributions → penalty waived."""
        candidate = _make_candidate(
            years_of_experience=6.0,
            github_activity_score=55,  # high GitHub activity
            career_history=[
                _role(
                    title="ML Engineer",
                    description=(
                        "Built internal proprietary ML models. Confidential enterprise work."
                        "Also contributed to open-source sentence-transformers library."
                    ),
                    duration_months=40,
                ),
            ],
        )
        result = _check_closed_source_isolation(candidate)
        assert result.penalty == 0.0, "Open-source contributions should waive the closed-source penalty"

    def test_passes_junior_candidate(self):
        """Under 5 years experience — rule should not apply."""
        candidate = _make_candidate(
            years_of_experience=3.0,
            career_history=[
                _role(
                    title="Engineer",
                    description="Internal proprietary systems. Confidential client work.",
                    duration_months=36,
                ),
            ],
        )
        result = _check_closed_source_isolation(candidate)
        assert result.penalty == 0.0, "Rule only applies to 5+ year senior profiles"

    def test_passes_senior_with_papers_and_talks(self):
        """Senior with published papers and conference talks → external validation present."""
        candidate = _make_candidate(
            years_of_experience=8.0,
            github_activity_score=-1,
            career_history=[
                _role(
                    title="Staff ML Engineer",
                    description=(
                        "Built internal recommendation engine (proprietary). "
                        "Published paper at RecSys 2023 (conference paper, peer-reviewed). "
                        "Gave a talk at MLConf India. "
                        "Contributed to open source library for embeddings."
                    ),
                    duration_months=60,
                ),
            ],
        )
        result = _check_closed_source_isolation(candidate)
        assert result.penalty == 0.0, "Papers and talks are sufficient external validation"


# ===========================================================================
# Integration tests — check_disqualifiers() public API
# ===========================================================================

class TestCheckDisqualifiersIntegration:
    """Tests for the top-level check_disqualifiers() function."""

    def test_returns_correct_types(self):
        """Return values must match the declared signature."""
        candidate = _make_candidate(career_history=[
            _role(title="ML Engineer", description="Built production retrieval systems.")
        ])
        is_dq, penalty, reasons = check_disqualifiers(candidate)
        assert isinstance(is_dq, bool)
        assert isinstance(penalty, float)
        assert isinstance(reasons, list)
        assert 0.0 <= penalty <= 1.0

    def test_clean_candidate_passes_all_rules(self):
        """
        A well-rounded ML engineer at a product company with production NLP
        experience should pass every rule with zero disqualification.
        """
        candidate = _make_candidate(
            years_of_experience=6.0,
            current_title="ML Engineer",
            current_company="Swiggy",
            github_activity_score=60,
            career_history=[
                _role(
                    company="Swiggy",
                    title="ML Engineer",
                    description=(
                        "Deployed production embedding retrieval system using "
                        "sentence-transformers and Elasticsearch. Implemented NDCG evaluation "
                        "framework for ranking quality. Ran A/B tests. Wrote and shipped "
                        "production code weekly. Contributed to open-source retrieval libraries."
                    ),
                    duration_months=36,
                    is_current=True,
                ),
                _role(
                    company="Flipkart",
                    title="Data Scientist",
                    description=(
                        "NLP-based semantic search for product catalogue. Built BM25 + "
                        "dense retrieval hybrid pipeline deployed to 50M users."
                    ),
                    duration_months=30,
                ),
            ],
            skills=[
                {"name": "NLP", "proficiency": "expert", "endorsements": 40, "duration_months": 48},
                {"name": "Elasticsearch", "proficiency": "advanced", "endorsements": 30, "duration_months": 36},
            ],
        )
        is_dq, penalty, reasons = check_disqualifiers(candidate)
        assert is_dq is False, "Clean candidate should not be disqualified"
        assert penalty < 0.3, f"Clean candidate should have low penalty, got {penalty}"

    def test_penalty_capped_at_1(self):
        """Even with multiple soft penalties, total penalty must not exceed 1.0."""
        # Construct a candidate that might trigger multiple soft rules
        today = date.today()
        candidate = _make_candidate(
            years_of_experience=3.0,  # keeps rule 3 and 8 silent
            career_history=[
                # Title-hopping
                _role(
                    title="Junior Engineer",
                    description="LangChain tutorial writer.",
                    duration_months=10,
                    start_date=(today - timedelta(days=50 * 30)).isoformat(),
                    end_date=(today - timedelta(days=40 * 30)).isoformat(),
                ),
                _role(
                    title="Senior Engineer",
                    description=(
                        "Built LangChain and llamaindex demos. No systems architecture work."
                    ),
                    duration_months=11,
                    start_date=(today - timedelta(days=38 * 30)).isoformat(),
                    end_date=(today - timedelta(days=27 * 30)).isoformat(),
                ),
                _role(
                    title="Staff Engineer",
                    description="LangChain integrations. No evaluation framework.",
                    duration_months=12,
                    start_date=(today - timedelta(days=25 * 30)).isoformat(),
                    end_date=(today - timedelta(days=13 * 30)).isoformat(),
                ),
            ],
        )
        _, penalty, _ = check_disqualifiers(candidate)
        assert penalty <= 1.0, f"Penalty must never exceed 1.0, got {penalty}"

    def test_hard_disqualifier_overrides_penalty_to_1(self):
        """If a hard rule fires, penalty must be exactly 1.0."""
        candidate = _make_candidate(
            career_history=[
                _role(
                    company="TCS",
                    title="Consultant",
                    description="All consulting, all client delivery.",
                ),
                _role(
                    company="Infosys",
                    title="Tech Lead",
                    description="Consulting firm work only.",
                ),
            ]
        )
        is_dq, penalty, reasons = check_disqualifiers(candidate)
        assert is_dq is True
        assert penalty == 1.0
        assert len(reasons) >= 1

    def test_empty_career_history_does_not_crash(self):
        """Candidate with no career history should return gracefully."""
        candidate = _make_candidate(career_history=[])
        is_dq, penalty, reasons = check_disqualifiers(candidate)
        assert isinstance(is_dq, bool)
        assert isinstance(penalty, float)

    def test_multiple_hard_rules_still_gives_single_disqualified_flag(self):
        """
        Candidate who triggers both 'pure research' AND 'CV/speech only' rules
        should be hard-disqualified with penalty=1.0 regardless.
        """
        candidate = _make_candidate(
            career_history=[
                _role(
                    company="MIT AI Lab",
                    title="Research Scientist",
                    description=(
                        "Research on computer vision at university lab. Published papers on image classification. "
                        "Academic lab work only; nothing shipped to real users. "
                        "Focused entirely on visual perception and object recognition."
                    ),
                    industry="Research",
                )
            ],
            skills=[
                {"name": "Image Classification", "proficiency": "expert", "endorsements": 50, "duration_months": 60},
                {"name": "Object Detection", "proficiency": "expert", "endorsements": 40, "duration_months": 48},
            ],
        )
        is_dq, penalty, reasons = check_disqualifiers(candidate)
        assert is_dq is True
        assert penalty == 1.0
        assert len(reasons) >= 2, "Should have reasons from multiple rules"
