"""
src/embeddings.py
-----------------
Embedding utilities for the RedRob candidate ranker.

Model: all-MiniLM-L6-v2  (384-dim, CPU-friendly, ~22 MB)
All public functions are pure / side-effect-free so they are easy to test.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model singleton – loaded once per process
# ---------------------------------------------------------------------------
_MODEL_NAME = "paraphrase-MiniLM-L3-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return (and lazily initialise) the shared SentenceTransformer model."""
    global _model
    if _model is None:
        logger.info("Loading sentence-transformer model: %s", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Text construction
# ---------------------------------------------------------------------------

def build_candidate_text(candidate: dict) -> str:
    """
    Combine a candidate's most semantically rich text fields into a single
    narrative string that the encoder can embed.

    TOKEN BUDGET DESIGN (model max = 256 tokens):
    Text is ordered highest-signal-first so that when the model truncates
    at 256 tokens, it retains the most important content:
      1. Skills (explicit keywords, ~30-50 tokens) — always fits
      2. profile.headline (~15 tokens) — always fits
      3. profile.summary (~80 tokens) — fits in most cases
      4. career_history[].title only (no long descriptions) — compact
      5. current_title + current_company for role context

    Career descriptions are intentionally excluded from the embedding text
    — they are long (200-400 tokens each) and push skills/headline out of
    the 256-token window. The disqualifier and behavioral signals capture
    career-level signals separately.

    Parameters
    ----------
    candidate : dict
        A single candidate object conforming to candidate_schema.json.

    Returns
    -------
    str
        A whitespace-normalised narrative string ready for encoding.
    """
    parts: list[str] = []

    profile = candidate.get("profile", {})

    # 1. Skill names — highest JD-match signal, shortest tokens (always in window)
    skill_names = [
        s["name"].strip()
        for s in candidate.get("skills", [])
        if s.get("name", "").strip()
    ]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))

    # 2. Current role context (very compact)
    current_title   = (profile.get("current_title") or "").strip()
    current_company = (profile.get("current_company") or "").strip()
    if current_title and current_company:
        parts.append(f"{current_title} at {current_company}")
    elif current_title:
        parts.append(current_title)

    # 3. Headline
    headline = (profile.get("headline") or "").strip()
    if headline:
        parts.append(headline)

    # 4. Professional summary (truncated to 300 chars to stay in budget)
    summary = (profile.get("summary") or "").strip()
    if summary:
        parts.append(summary[:300])

    # 5. Career role titles only (no long descriptions to avoid overflow)
    role_titles = [
        (role.get("title") or "").strip()
        for role in candidate.get("career_history", [])
        if (role.get("title") or "").strip()
    ]
    if role_titles:
        parts.append("Experience: " + "; ".join(role_titles))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# JD embedding
# ---------------------------------------------------------------------------

def embed_jd(jd_parsed: dict) -> np.ndarray:
    """
    Embed the job description using its ideal_profile_signals field,
    supplemented by required_skills and nice_to_have_skills for richer
    semantic coverage.

    Parameters
    ----------
    jd_parsed : dict
        Parsed JD dict (as produced by jd_parser.py / jd_parsed.json).

    Returns
    -------
    np.ndarray
        Shape (384,), dtype float32 – L2-normalised embedding vector.
    """
    model = _get_model()

    parts: List[str] = []

    # Primary signal: the ideal profile description
    ideal = (jd_parsed.get("ideal_profile_signals") or "").strip()
    if ideal:
        parts.append(ideal)

    # Required skills prose (they may be long descriptive strings)
    for skill in jd_parsed.get("required_skills", []):
        text = skill.strip() if isinstance(skill, str) else ""
        if text:
            parts.append(text)

    # Nice-to-have skills (lower weight, but still semantically relevant)
    for skill in jd_parsed.get("nice_to_have_skills", []):
        text = skill.strip() if isinstance(skill, str) else ""
        if text:
            parts.append(text)

    jd_text = " ".join(parts)

    if not jd_text:
        raise ValueError("jd_parsed contains no usable text for embedding.")

    logger.debug("Embedding JD text (%d chars)", len(jd_text))
    embedding: np.ndarray = model.encode(
        jd_text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding.astype(np.float32)


# ---------------------------------------------------------------------------
# Batch candidate embedding
# ---------------------------------------------------------------------------

def embed_candidates_batch(
    candidates: list[dict],
    batch_size: int = 256,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Vectorised batch embedding of a list of candidate dicts.

    All candidates are converted to text first, then passed to
    SentenceTransformer.encode in a single batched call for maximum
    throughput (avoids Python-level loop overhead inside the model).

    Parameters
    ----------
    candidates : list[dict]
        List of candidate objects conforming to candidate_schema.json.
    batch_size : int
        Number of texts to encode per forward pass. 256 is a good
        default for CPU; lower if you hit OOM.
    show_progress : bool
        Whether to display a tqdm progress bar.

    Returns
    -------
    np.ndarray
        Shape (N, 384), dtype float32 – one L2-normalised row per candidate.
        Row order matches the input list order.
    """
    if not candidates:
        return np.empty((0, 384), dtype=np.float32)

    model = _get_model()

    logger.info("Building texts for %d candidates …", len(candidates))
    texts = [build_candidate_text(c) for c in candidates]

    logger.info(
        "Encoding %d texts in batches of %d …", len(texts), batch_size
    )
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)
