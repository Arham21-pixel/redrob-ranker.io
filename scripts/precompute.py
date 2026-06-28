"""
scripts/precompute.py
---------------------
Offline precomputation of candidate embeddings.

Usage
-----
# Test on sample first (fast, ~100 candidates):
    python scripts/precompute.py --sample

# Full 100K run (no time limit, runs offline):
    python scripts/precompute.py

Outputs
-------
data/processed/candidate_embeddings.npy   – float32 array (N, 384)
data/processed/candidate_ids.json         – ordered list of candidate_id strings
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to repo root – script must be run from there)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

CANDIDATES_JSONL = DATA_RAW / "candidates.jsonl"
SAMPLE_CANDIDATES_JSON = DATA_RAW / "sample_candidates.json"

OUT_EMBEDDINGS = DATA_PROCESSED / "candidate_embeddings.npy"
OUT_IDS = DATA_PROCESSED / "candidate_ids.json"

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
# Helpers
# ---------------------------------------------------------------------------

def load_sample(path: Path) -> list[dict]:
    """Load sample_candidates.json (array at root)."""
    logger.info("Loading sample candidates from %s", path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a JSON array in {path}, got {type(data)}")


def load_full_jsonl(path: Path) -> list[dict]:
    """
    Stream-load candidates.jsonl line-by-line to avoid loading 487 MB at once.
    Returns a flat list; adjust if RAM is tight (pass generator to encoder).
    """
    logger.info("Streaming candidates from %s …", path)
    candidates: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", i + 1, exc)
    logger.info("Loaded %d candidates.", len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute candidate embeddings and save to disk."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on sample_candidates.json instead of the full 100K JSONL.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size (default: 256, lower if OOM).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load candidates
    # ------------------------------------------------------------------
    if args.sample:
        candidates = load_sample(SAMPLE_CANDIDATES_JSON)
        logger.info("Sample mode: %d candidates loaded.", len(candidates))
    else:
        candidates = load_full_jsonl(CANDIDATES_JSONL)

    if not candidates:
        logger.error("No candidates found – aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Import here so the model load is reflected in the timing
    # ------------------------------------------------------------------
    # We import after argument parsing so --help is instant
    from src.embeddings import embed_candidates_batch  # noqa: PLC0415

    # ------------------------------------------------------------------
    # 3. Embed
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    embeddings: np.ndarray = embed_candidates_batch(
        candidates,
        batch_size=args.batch_size,
        show_progress=True,
    )

    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 4. Build parallel ID index
    # ------------------------------------------------------------------
    candidate_ids: list[str] = [c["candidate_id"] for c in candidates]

    assert len(candidate_ids) == embeddings.shape[0], (
        f"ID/embedding count mismatch: {len(candidate_ids)} vs {embeddings.shape[0]}"
    )

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Resolve output paths (use _sample suffix in sample mode to avoid
    # accidentally overwriting the full precomputed file)
    if args.sample:
        out_emb = DATA_PROCESSED / "candidate_embeddings_sample.npy"
        out_ids = DATA_PROCESSED / "candidate_ids_sample.json"
    else:
        out_emb = OUT_EMBEDDINGS
        out_ids = OUT_IDS

    logger.info("Saving embeddings -> %s  shape=%s", out_emb, embeddings.shape)
    np.save(out_emb, embeddings)

    logger.info("Saving ID index  -> %s  (%d ids)", out_ids, len(candidate_ids))
    with out_ids.open("w", encoding="utf-8") as fh:
        json.dump(candidate_ids, fh, indent=None)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    n = len(candidate_ids)
    speed = n / elapsed if elapsed > 0 else float("inf")

    print("\n" + "=" * 60)
    print(f"  Candidates embedded : {n:,}")
    print(f"  Embedding shape     : {embeddings.shape}")
    print(f"  Total time          : {elapsed:.1f}s  ({speed:.0f} cands/sec)")
    print(f"  Output embeddings   : {out_emb}")
    print(f"  Output ID index     : {out_ids}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
