"""
src/fusion_ranker.py
--------------------
Combines four independent signals into a single final ranking score for
each candidate, then sorts a batch of candidates by that score.

Signal sources
--------------
  semantic_similarity  : cosine similarity between JD embedding and candidate
                         embedding, produced by src/embeddings.py
  behavioral_score     : engagement / reachability score from
                         src/behavioral_scorer.py  (0-1)
  disqualifier_penalty : hard/soft penalty from src/disqualifiers.py  (0-1)
  honeypot_score       : fabrication suspicion from src/honeypot_detector.py (0-1)

Final score formula
-------------------
  score = (semantic_similarity * 0.45)
        + (behavioral_score    * 0.25)
        - (disqualifier_penalty * 0.50)
        - (honeypot_score       * 1.00)

Hard floors
-----------
  • is_disqualified == True  →  score clamped to [0.00, 0.05)
  • honeypot_score  > 0.70   →  score clamped to [0.00, 0.05)

Both floors exist because a profile that is definitively disqualified or
very likely fabricated must never surface near the top of results regardless
of how strong the semantic match appears.

Public API
----------
  rank_candidate(candidate, jd_embedding, candidate_embedding) -> dict
  rank_candidates_batch(candidates, jd_embedding, candidate_embeddings) -> list[dict]
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.behavioral_scorer import score_behavior
from src.disqualifiers import check_disqualifiers
from src.honeypot_detector import detect_honeypot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score formula weights
# ---------------------------------------------------------------------------
_W_SEMANTIC   = 0.45   # Primary signal: JD-candidate semantic alignment
_W_BEHAVIORAL = 0.25   # Reachability / engagement modifier
_W_DQ_PENALTY = 0.50   # Disqualifier penalty subtracted from final score
_W_HONEYPOT   = 1.00   # Full deduction: fabricated profiles must not rank

# Hard-floor threshold and floor value
_HONEYPOT_HARD_FLOOR_THRESHOLD = 0.70
_HARD_FLOOR_VALUE = 0.01   # Non-zero so scores are still sortable at the bottom


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.

    Both embeddings from SentenceTransformer are already L2-normalised
    (normalize_embeddings=True in embeddings.py), so this reduces to a
    dot product.  We still normalise defensively here in case embeddings
    come from another source.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _apply_hard_floors(
    raw_score: float,
    is_disqualified: bool,
    honeypot_score: float,
) -> tuple[float, bool]:
    """
    Apply hard floors to the raw fusion score.

    Returns
    -------
    final_score : float
    hard_floored : bool  — True if any floor was applied
    """
    if is_disqualified or honeypot_score > _HONEYPOT_HARD_FLOOR_THRESHOLD:
        return _HARD_FLOOR_VALUE, True
    return raw_score, False


# ---------------------------------------------------------------------------
# Single candidate scoring
# ---------------------------------------------------------------------------

def rank_candidate(
    candidate: dict[str, Any],
    jd_embedding: np.ndarray,
    candidate_embedding: np.ndarray,
) -> dict[str, Any]:
    """
    Compute a final score for a single candidate and return a rich result dict.

    Parameters
    ----------
    candidate : dict
        A single candidate record conforming to the Redrob candidate schema.
    jd_embedding : np.ndarray
        Shape (384,) — pre-computed JD embedding from embed_jd().
    candidate_embedding : np.ndarray
        Shape (384,) — pre-computed candidate embedding from
        embed_candidates_batch().

    Returns
    -------
    dict with keys:
        candidate_id      : str
        final_score       : float  [0, 1]  (higher = better)
        semantic_sim      : float  [0, 1]
        behavioral_score  : float  [0, 1]
        disqualifier_penalty : float  [0, 1]
        is_disqualified   : bool
        disqualifier_reasons : list[str]
        honeypot_score    : float  [0, 1]
        honeypot_checks   : list[str]
        hard_floored      : bool   — True if a hard floor was applied
    """
    candidate_id = candidate.get("candidate_id", "UNKNOWN")

    # --- 1. Semantic similarity ---
    sem_sim = _cosine_similarity(jd_embedding, candidate_embedding)
    # Clamp to [0, 1]; cosine can be slightly negative for very poor matches
    sem_sim = float(max(0.0, min(1.0, sem_sim)))

    # --- 2. Behavioral score ---
    signals = candidate.get("redrob_signals", {})
    beh_score = score_behavior(signals)

    # --- 3. Disqualifier check ---
    is_dq, dq_penalty, dq_reasons = check_disqualifiers(candidate)

    # --- 4. Honeypot detection ---
    hp_score, hp_checks = detect_honeypot(candidate)

    # --- 5. Raw fusion score ---
    raw = (
        sem_sim     * _W_SEMANTIC
        + beh_score * _W_BEHAVIORAL
        - dq_penalty * _W_DQ_PENALTY
        - hp_score   * _W_HONEYPOT
    )
    # Clamp raw to [0, 1] before applying floors
    raw = float(max(0.0, min(1.0, raw)))

    # --- 6. Hard floors ---
    final_score, hard_floored = _apply_hard_floors(raw, is_dq, hp_score)

    logger.debug(
        "[%s] final=%.4f  sem=%.3f  beh=%.3f  dq=%.3f(hard=%s)  hp=%.3f  floored=%s",
        candidate_id, final_score, sem_sim, beh_score,
        dq_penalty, is_dq, hp_score, hard_floored,
    )

    return {
        "candidate_id":           candidate_id,
        "final_score":            round(final_score, 6),
        "semantic_sim":           round(sem_sim, 6),
        "behavioral_score":       round(beh_score, 6),
        "disqualifier_penalty":   round(dq_penalty, 6),
        "is_disqualified":        is_dq,
        "disqualifier_reasons":   dq_reasons,
        "honeypot_score":         round(hp_score, 6),
        "honeypot_checks":        hp_checks,
        "hard_floored":           hard_floored,
    }


# ---------------------------------------------------------------------------
# Batch scoring  (vectorised cosine, serial signal scoring)
# ---------------------------------------------------------------------------

def rank_candidates_batch(
    candidates: list[dict[str, Any]],
    jd_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Score and rank a batch of candidates.

    Cosine similarity is computed in a single vectorised dot product for
    speed. Behavioral, disqualifier, and honeypot scoring remain serial
    (they are Python-pure and fast; no GPU needed).

    Parameters
    ----------
    candidates : list[dict]
        Ordered list of candidate records. Must match the row order of
        candidate_embeddings.
    jd_embedding : np.ndarray
        Shape (384,) — from embed_jd().
    candidate_embeddings : np.ndarray
        Shape (N, 384) — from embed_candidates_batch(), rows align with
        candidates list.

    Returns
    -------
    list[dict]
        Result dicts (same schema as rank_candidate()), sorted descending
        by final_score.

    Raises
    ------
    ValueError
        If the number of candidates and embedding rows don't match.
    """
    n = len(candidates)
    if n == 0:
        return []

    if candidate_embeddings.shape[0] != n:
        raise ValueError(
            f"candidates list has {n} items but candidate_embeddings has "
            f"{candidate_embeddings.shape[0]} rows."
        )

    # Vectorised cosine similarity: dot product (embeddings are L2-normalised)
    jd_norm = jd_embedding / (np.linalg.norm(jd_embedding) + 1e-12)
    cand_norms = candidate_embeddings / (
        np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-12
    )
    # Shape: (N,)
    sem_sims: np.ndarray = np.clip(cand_norms @ jd_norm, 0.0, 1.0)

    results: list[dict[str, Any]] = []

    for i, candidate in enumerate(candidates):
        candidate_id = candidate.get("candidate_id", f"IDX_{i}")
        sem_sim = float(sem_sims[i])

        # Behavioral
        signals = candidate.get("redrob_signals", {})
        beh_score = score_behavior(signals)

        # Disqualifiers
        is_dq, dq_penalty, dq_reasons = check_disqualifiers(candidate)

        # Honeypot
        hp_score, hp_checks = detect_honeypot(candidate)

        # Raw fusion
        raw = float(max(0.0, min(1.0,
            sem_sim     * _W_SEMANTIC
            + beh_score * _W_BEHAVIORAL
            - dq_penalty * _W_DQ_PENALTY
            - hp_score   * _W_HONEYPOT
        )))

        # Hard floors
        final_score, hard_floored = _apply_hard_floors(raw, is_dq, hp_score)

        results.append({
            "candidate_id":           candidate_id,
            "final_score":            round(final_score, 6),
            "semantic_sim":           round(sem_sim, 6),
            "behavioral_score":       round(beh_score, 6),
            "disqualifier_penalty":   round(dq_penalty, 6),
            "is_disqualified":        is_dq,
            "disqualifier_reasons":   dq_reasons,
            "honeypot_score":         round(hp_score, 6),
            "honeypot_checks":        hp_checks,
            "hard_floored":           hard_floored,
        })

    # Sort descending by final_score
    results.sort(key=lambda r: r["final_score"], reverse=True)

    logger.info(
        "Ranked %d candidates. Top score=%.4f | Bottom score=%.4f | "
        "Hard-floored=%d | Disqualified=%d | Honeypots=%d",
        n,
        results[0]["final_score"] if results else 0.0,
        results[-1]["final_score"] if results else 0.0,
        sum(1 for r in results if r["hard_floored"]),
        sum(1 for r in results if r["is_disqualified"]),
        sum(1 for r in results if r["honeypot_score"] > _HONEYPOT_HARD_FLOOR_THRESHOLD),
    )

    return results
