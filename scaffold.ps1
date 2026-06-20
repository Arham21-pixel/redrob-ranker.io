$baseDir = "c:\Users\arham\OneDrive\Documents\redrob ranker"
cd $baseDir

$directories = @(
    "data/raw",
    "data/processed",
    "src",
    "scripts",
    "notebooks",
    "tests",
    "sandbox",
    "outputs"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$files = @(
    "README.md",
    "requirements.txt",
    "submission_metadata.yaml",
    ".gitignore",
    "data/raw/candidates.jsonl",
    "data/raw/sample_candidates.json",
    "data/raw/job_description.docx",
    "data/raw/candidate_schema.json",
    "data/raw/redrob_signals_doc.docx",
    "data/raw/sample_submission.csv",
    "data/raw/submission_spec.docx",
    "data/processed/candidate_embeddings.npy",
    "data/processed/jd_parsed.json",
    "src/__init__.py",
    "src/jd_parser.py",
    "src/honeypot_detector.py",
    "src/embeddings.py",
    "src/disqualifiers.py",
    "src/behavioral_scorer.py",
    "src/fusion_ranker.py",
    "src/reasoning_generator.py",
    "src/config.py",
    "src/utils.py",
    "scripts/precompute.py",
    "scripts/rank.py",
    "notebooks/exploration.ipynb",
    "tests/test_honeypot_detector.py",
    "tests/test_disqualifiers.py",
    "tests/test_fusion_ranker.py",
    "sandbox/app.py",
    "sandbox/requirements.txt",
    "outputs/submission.csv",
    "validate_submission.py"
)

foreach ($file in $files) {
    New-Item -ItemType File -Force -Path $file | Out-Null
}

$gitignoreContent = @"
# Ignore large raw candidates file
data/raw/candidates.jsonl

# Ignore processed binary file
data/processed/candidate_embeddings.npy

# Ignore outputs until final
outputs/submission.csv

# Python
__pycache__/
*.py[cod]
*$py.class

# Jupyter Notebook
.ipynb_checkpoints

# Virtual environments
venv/
.venv/
env/
"@

Set-Content -Path ".gitignore" -Value $gitignoreContent

git init
git add .
git commit -m "Initial scaffold for redrob-ranker"
git branch -M main
git remote add origin https://github.com/Arham21-pixel/redrob-ranker.io.git
git push -u origin main
