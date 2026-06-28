import json
import numpy as np
from src.embeddings import embed_jd, embed_candidates_batch
from src.fusion_ranker import rank_candidates_batch
from src.reasoning_generator import generate_reasoning

jd = json.load(open('data/processed/jd_parsed.json'))
candidates = json.load(open('data/raw/sample_candidates.json'))[:10]

print('Embedding JD ...')
jd_emb = embed_jd(jd)

print('Embedding 10 candidates ...')
cand_embs = embed_candidates_batch(candidates, show_progress=False)

print('Ranking ...')
results = rank_candidates_batch(candidates, jd_emb, cand_embs)

print()
print('=' * 72)
header = f"{'Rank':<5} {'ID':<14} {'Final':>7} {'Sem':>7} {'Beh':>7} {'DQ':>6} {'HP':>6} {'Floor':<7}"
print(header)
print('-' * 72)
for rank, r in enumerate(results, 1):
    floor_label = 'YES' if r['hard_floored'] else 'no'
    print(
        f"{rank:<5} {r['candidate_id']:<14} {r['final_score']:>7.4f} "
        f"{r['semantic_sim']:>7.4f} {r['behavioral_score']:>7.4f} "
        f"{r['disqualifier_penalty']:>6.3f} {r['honeypot_score']:>6.3f} "
        f"{floor_label:<7}"
    )

print()
print('--- REASONING SAMPLES (top 3 + any disqualified) ---')
cand_map = {c['candidate_id']: c for c in candidates}
shown = set()
for r in results[:3]:
    cand = cand_map[r['candidate_id']]
    reasoning = generate_reasoning(cand, r)
    print(f"[{r['candidate_id']}] score={r['final_score']:.4f}")
    print(f"  {reasoning}")
    print()
    shown.add(r['candidate_id'])

# Show one disqualified if any
for r in results:
    if r['hard_floored'] and r['candidate_id'] not in shown:
        cand = cand_map[r['candidate_id']]
        reasoning = generate_reasoning(cand, r)
        print(f"[{r['candidate_id']}] FLOORED score={r['final_score']:.4f}")
        print(f"  {reasoning}")
        print()
        break
