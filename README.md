# Redrob Ranker — India Runs Hackathon

## Setup
pip install -r requirements.txt

## Reproduce Submission (exact commands judges will run)
Step 1 - Precompute embeddings (one-time, no time limit):
python scripts/precompute.py

Step 2 - Generate ranked submission CSV (runs in ~68 seconds):
python scripts/rank.py --candidates data/raw/candidates.jsonl --out outputs/submission.csv

Step 3 - Validate output:
python validate_submission.py outputs/submission.csv

## Architecture
- JD Parser: Extracts structured requirements and disqualifiers from job description
- Disqualifier Engine: Rule-based hard filters derived directly from JD's explicit 
  "do NOT want" list (consulting-only background, pure research, title-hopping, etc.)
- Honeypot Detector: Flags internally impossible profiles (expert skill with 0 months 
  experience, impossible tenure durations, assessment score contradictions)
- Semantic Embedder: paraphrase-MiniLM-L3-v2 for candidate narrative vs JD similarity
- Behavioral Scorer: Weights recruiter_response_rate, last_active_date recency, 
  notice_period_days, github_activity_score from redrob_signals
- Fusion Ranker: Combines all signals into final score
- Reasoning Generator: Template-grounded, zero hallucination, field-sourced only

## Results
- Candidates processed: 100,000
- Candidates disqualified: 48,796
- Honeypots detected and floored: 5
- Ranking runtime: 67.8 seconds (CPU only, no network calls)
- Validator: PASSED (0 errors)

## Sandbox
Live demo: https://redrob-rankerio.streamlit.app/
