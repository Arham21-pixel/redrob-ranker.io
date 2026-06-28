"""
scripts/rank.py
---------------
Full ranking pipeline — must complete in < 5 minutes on CPU.

Usage:
    python scripts/rank.py --candidates data/raw/candidates.jsonl --out outputs/submission.csv

Algorithm
---------
1. Load pre-computed candidate embeddings from data/processed/
   (run scripts/precompute.py first if they don't exist)
2. Embed the JD (fast, single vector)
3. Compute cosine similarity for ALL 100K candidates via vectorised numpy
4. Score behavioral, disqualifier, honeypot for all candidates
5. Apply fusion formula, sort, take top 100
6. Generate reasoning for top 100
7. Write submission CSV

Performance design
------------------
- Vectorised cosine (numpy dot) over 100K x 384 matrix: ~0.05s
- Behavioral scorer: pure Python, ~5-10s for 100K
- Disqualifier + honeypot: more complex, ~60-90s for 100K total
- Reasoning: only 100 candidates, negligible
- Total target: < 5 minutes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT      = Path(__file__).resolve().parent.parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_RAW       = REPO_ROOT / "data" / "raw"
OUTPUTS_DIR    = REPO_ROOT / "outputs"

JD_PATH            = DATA_PROCESSED / "jd_parsed.json"
EMBEDDINGS_PATH    = DATA_PROCESSED / "candidate_embeddings.npy"
IDS_PATH           = DATA_PROCESSED / "candidate_ids.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fusion constants (mirrored from fusion_ranker.py for speed — avoid re-import
# overhead and keep ranking self-contained)
# ---------------------------------------------------------------------------
_W_SEMANTIC   = 0.45
_W_BEHAVIORAL = 0.25
_W_DQ_PENALTY = 0.50
_W_HONEYPOT   = 1.00
_HP_FLOOR_THRESHOLD = 0.70
_HARD_FLOOR_VALUE   = 0.01


def _load_candidates_jsonl(path: Path) -> list[dict]:
    """Stream-load candidates.jsonl. Skips blank / malformed lines."""
    logger.info("Loading candidates from %s ...", path)
    candidates = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d", i + 1)
    logger.info("Loaded %d candidates.", len(candidates))
    return candidates


def _vectorised_cosine(jd_emb: np.ndarray, cand_embs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between JD embedding and all candidate embeddings
    in a single numpy matmul. Both are assumed to be L2-normalised already
    (from SentenceTransformer with normalize_embeddings=True).

    Returns shape (N,) float32.
    """
    jd_norm = jd_emb / (np.linalg.norm(jd_emb) + 1e-12)
    cand_norms = cand_embs / (
        np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-12
    )
    sims = cand_norms @ jd_norm          # (N,)
    return np.clip(sims, 0.0, 1.0).astype(np.float32)


def _score_all(
    candidates: list[dict],
    sem_sims: np.ndarray,
) -> list[dict]:
    """
    Score behavioral + disqualifier + honeypot for all candidates.
    Returns list of score dicts (same order as candidates).
    """
    from src.behavioral_scorer import score_behavior
    from src.disqualifiers import check_disqualifiers
    from src.honeypot_detector import detect_honeypot

    n = len(candidates)
    results = []

    for i, cand in enumerate(candidates):
        sem = float(sem_sims[i])
        signals = cand.get("redrob_signals", {})

        beh    = score_behavior(signals)
        is_dq, dq_pen, dq_reasons = check_disqualifiers(cand)
        hp, hp_checks              = detect_honeypot(cand)

        raw = float(max(0.0, min(1.0,
            sem    * _W_SEMANTIC
            + beh  * _W_BEHAVIORAL
            - dq_pen * _W_DQ_PENALTY
            - hp   * _W_HONEYPOT
        )))

        if is_dq or hp > _HP_FLOOR_THRESHOLD:
            final = _HARD_FLOOR_VALUE
            hard_floored = True
        else:
            final = raw
            hard_floored = False

        results.append({
            "candidate_id":         cand.get("candidate_id", f"IDX_{i}"),
            "final_score":          final,
            "semantic_sim":         sem,
            "behavioral_score":     beh,
            "disqualifier_penalty": dq_pen,
            "is_disqualified":      is_dq,
            "disqualifier_reasons": dq_reasons,
            "honeypot_score":       hp,
            "honeypot_checks":      hp_checks,
            "hard_floored":         hard_floored,
        })

        if (i + 1) % 10000 == 0:
            logger.info("  Scored %d / %d ...", i + 1, n)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank candidates and produce submission.csv"
    )
    parser.add_argument(
        "--candidates",
        default=str(DATA_RAW / "candidates.jsonl"),
        help="Path to candidates JSONL file (default: data/raw/candidates.jsonl)",
    )
    parser.add_argument(
        "--out",
        default=str(OUTPUTS_DIR / "submission.csv"),
        help="Output CSV path (default: outputs/submission.csv)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top candidates to include (default: 100)",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: Load pre-computed embeddings
    # ------------------------------------------------------------------
    if not EMBEDDINGS_PATH.exists() or not IDS_PATH.exists():
        logger.error(
            "Pre-computed embeddings not found. Run first:\n"
            "  python scripts/precompute.py\n"
        )
        sys.exit(1)

    logger.info("Loading pre-computed embeddings from %s ...", EMBEDDINGS_PATH)
    cand_embs = np.load(EMBEDDINGS_PATH)          # (N, 384) float32
    with IDS_PATH.open(encoding="utf-8") as fh:
        precomputed_ids: list[str] = json.load(fh)

    logger.info("Embeddings shape: %s  |  %d pre-computed IDs", cand_embs.shape, len(precomputed_ids))

    # ------------------------------------------------------------------
    # Step 2: Load candidates JSONL
    # ------------------------------------------------------------------
    cands_path = Path(args.candidates)
    candidates = _load_candidates_jsonl(cands_path)

    # Build lookup: candidate_id -> index in precomputed embeddings
    id_to_emb_idx = {cid: i for i, cid in enumerate(precomputed_ids)}

    # Align candidates with embeddings (only keep candidates with embeddings)
    aligned_candidates = []
    aligned_emb_rows   = []
    missing_emb        = 0

    for cand in candidates:
        cid = cand.get("candidate_id", "")
        idx = id_to_emb_idx.get(cid)
        if idx is None:
            missing_emb += 1
            continue
        aligned_candidates.append(cand)
        aligned_emb_rows.append(idx)

    if missing_emb:
        logger.warning("%d candidates have no pre-computed embedding (skipped).", missing_emb)

    aligned_embs = cand_embs[aligned_emb_rows]   # (M, 384)
    logger.info("Aligned %d candidates with embeddings.", len(aligned_candidates))

    # ------------------------------------------------------------------
    # Step 3: Embed JD
    # ------------------------------------------------------------------
    logger.info("Embedding JD ...")
    from src.embeddings import embed_jd
    jd = json.loads(JD_PATH.read_text(encoding="utf-8"))
    jd_emb = embed_jd(jd)                         # (384,)

    # ------------------------------------------------------------------
    # Step 4: Vectorised cosine similarity (one numpy op for all 100K)
    # ------------------------------------------------------------------
    logger.info("Computing cosine similarities (vectorised) ...")
    t_cos = time.perf_counter()
    sem_sims = _vectorised_cosine(jd_emb, aligned_embs)
    logger.info("Cosine done in %.2fs", time.perf_counter() - t_cos)

    # ------------------------------------------------------------------
    # Step 5: Score all candidates
    # ------------------------------------------------------------------
    logger.info("Scoring behavioral / disqualifier / honeypot for %d candidates ...",
                len(aligned_candidates))
    t_score = time.perf_counter()
    all_scores = _score_all(aligned_candidates, sem_sims)
    logger.info("Scoring done in %.1fs", time.perf_counter() - t_score)

    # ------------------------------------------------------------------
    # Step 6: Sort and take top N
    # ------------------------------------------------------------------
    all_scores.sort(key=lambda r: r["final_score"], reverse=True)
    top_scores = all_scores[:args.top_n]

    n_disqualified = sum(1 for r in all_scores if r["is_disqualified"])
    n_honeypots    = sum(1 for r in all_scores if r["honeypot_score"] > _HP_FLOOR_THRESHOLD)
    n_floored      = sum(1 for r in all_scores if r["hard_floored"])

    logger.info(
        "Stats: disqualified=%d  honeypots=%d  hard-floored=%d",
        n_disqualified, n_honeypots, n_floored,
    )

    # ------------------------------------------------------------------
    # Step 7: Generate reasoning for top 100
    # ------------------------------------------------------------------
    logger.info("Generating reasoning for top %d candidates ...", args.top_n)
    from src.reasoning_generator import generate_reasoning
    cand_map = {c.get("candidate_id"): c for c in aligned_candidates}

    rows = []
    for rank, score_dict in enumerate(top_scores, start=1):
        cid   = score_dict["candidate_id"]
        cand  = cand_map.get(cid, {"candidate_id": cid})
        score = score_dict["final_score"]

        reasoning = generate_reasoning(cand, score_dict)

        rows.append({
            "candidate_id": cid,
            "rank":         rank,
            "score":        round(score, 6),
            "reasoning":    reasoning,
        })

    # ------------------------------------------------------------------
    # Step 8: Write CSV
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Written %d rows to %s", len(df), out_path)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    t_total = time.perf_counter() - t_start
    print()
    print("=" * 65)
    print(f"  Total candidates processed : {len(aligned_candidates):,}")
    print(f"  Candidates disqualified    : {n_disqualified:,}")
    print(f"  Honeypots detected (>0.7)  : {n_honeypots:,}")
    print(f"  Hard-floored total         : {n_floored:,}")
    print(f"  Top-{args.top_n} written to         : {out_path}")
    print()
    print(f"  Top 10 candidates:")
    for r in rows[:10]:
        print(f"    #{r['rank']:>3}  {r['candidate_id']}  score={r['score']:.4f}")
    print()
    print(f"  Total ranking runtime      : {t_total:.1f}s")
    print("=" * 65)

    if t_total > 300:
        logger.warning("Runtime %.1fs exceeds 5-minute limit!", t_total)
    else:
        logger.info("Runtime %.1fs — within 5-minute limit.", t_total)


if __name__ == "__main__":
    main()
