import json
from src.embeddings import build_candidate_text, _get_model

candidates = json.load(open('data/raw/sample_candidates.json'))[:20]
model = _get_model()
tokenizer = model.tokenizer

lengths = []
for c in candidates:
    text = build_candidate_text(c)
    tokens = tokenizer.encode(text)
    lengths.append(len(tokens))
    print(f"{c['candidate_id']}  chars={len(text):5d}  tokens={len(tokens):4d}")

print(f"\nAvg tokens: {sum(lengths)/len(lengths):.0f}  Max: {max(lengths)}  Min: {min(lengths)}")
print(f"Model max_seq_length: {model.max_seq_length}")
