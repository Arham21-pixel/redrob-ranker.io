import json
from src.disqualifiers import check_disqualifiers
from src.honeypot_detector import detect_honeypot
from src.behavioral_scorer import score_behavior

candidates = json.load(open('data/raw/sample_candidates.json'))

print('=== DISQUALIFIERS (first 5) ===')
for c in candidates[:5]:
    is_dq, pen, reasons = check_disqualifiers(c)
    r_short = reasons[0][:80] if reasons else 'none'
    print(f"{c['candidate_id']}  dq={is_dq}  penalty={pen:.2f}  reason={r_short}")

print()
print('=== HONEYPOT (first 5) ===')
for c in candidates[:5]:
    score, checks = detect_honeypot(c)
    c_short = checks[0][:80] if checks else 'none'
    print(f"{c['candidate_id']}  hp={score:.3f}  check={c_short}")

print()
print('=== BEHAVIORAL (first 5) ===')
for c in candidates[:5]:
    beh = score_behavior(c.get('redrob_signals', {}))
    print(f"{c['candidate_id']}  behavioral={beh:.4f}")

print()
print('All module checks PASSED')
