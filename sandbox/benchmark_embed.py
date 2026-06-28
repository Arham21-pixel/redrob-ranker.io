import json, time
import sys, os
sys.path.insert(0, '.')
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from src.embeddings import build_candidate_text, _get_model

candidates = json.load(open('data/raw/sample_candidates.json'))[:20]
model = _get_model()

texts = [build_candidate_text(c) for c in candidates]
tok_lens = [len(model.tokenizer.encode(t)) for t in texts]
print(f"Avg tokens: {sum(tok_lens)/len(tok_lens):.1f}  Max: {max(tok_lens)}  Min: {min(tok_lens)}")
print(f"Model max_seq_length: {model.max_seq_length}")
print()

# Time 1000 candidates
candidates_1k = (candidates * 20)[:1000]
texts_1k = [build_candidate_text(c) for c in candidates_1k]
t0 = time.perf_counter()
embs = model.encode(texts_1k, batch_size=256, normalize_embeddings=True,
                    show_progress_bar=False, convert_to_numpy=True)
elapsed = time.perf_counter() - t0
cps = 1000 / elapsed
eta_min = (100000 / cps) / 60
print(f"1000 candidates in {elapsed:.1f}s = {cps:.0f} cands/sec")
print(f"ETA for 100K: {eta_min:.1f} minutes")
print(f"Shape: {embs.shape}")
