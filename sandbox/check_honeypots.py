import pandas as pd
import json
from src.honeypot_detector import detect_honeypot

df = pd.read_csv('outputs/submission.csv')
top_10_ids = df['candidate_id'].head(10).tolist()

candidates = []
with open('data/raw/candidates.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        c = json.loads(line)
        if c.get("candidate_id") in top_10_ids:
            candidates.append(c)

id_to_cand = {c['candidate_id']: c for c in candidates}

print("=== TOP 10 HONEYPOT CHECK ===")
for cid in top_10_ids:
    cand = id_to_cand.get(cid)
    if not cand:
        print(f"{cid} not found!")
        continue
    hp_score, checks = detect_honeypot(cand)
    print(f"{cid}: honeypot_score = {hp_score:.3f}")
    if hp_score > 0.7:
        print(f"  WARNING: High honeypot score! {checks}")

print("Check completed.")
