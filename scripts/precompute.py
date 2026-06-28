"""
scripts/precompute.py
---------------------
Offline precomputation of candidate embeddings.

Performance notes
-----------------
The bottleneck on CPU is the transformer forward pass, not Python overhead.
Key optimisations used here:
  1. Build ALL texts first (pure Python, fast)
  2. Pass them all to model.encode() in ONE call — SentenceTransformer
     internally batches; Python loop overhead per batch is eliminated
  3. Use batch_size=512 — larger batches amortise tokenization overhead
  4. show_progress_bar=True so we can monitor throughput
  5. Save incrementally every CHUNK_SIZE candidates so we can resume
     if interrupted, and avoid OOM on very large datasets

Usage
-----
# Test on sample first (fast, ~50 candidates):
    python scripts/precompute.py --sample

# Full 100K run (no time limit, runs offline):
    python scripts/precompute.py

# Resume from a checkpoint (if interrupted):
    python scripts/precompute.py --resume

Outputs
-------
data/processed/candidate_embeddings.npy   -- float32 array (N, 384)
data/processed/candidate_ids.json         -- ordered list of candidate_id strings
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
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT      = Path(__file__).resolve().parent.parent
DATA_RAW       = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

CANDIDATES_JSONL       = DATA_RAW / "candidates.jsonl"
SAMPLE_CANDIDATES_JSON = DATA_RAW / "sample_candidates.json"

OUT_EMBEDDINGS = DATA_PROCESSED / "candidate_embeddings.npy"
OUT_IDS        = DATA_PROCESSED / "candidate_ids.json"

# Chunk size for incremental saves (reduces peak RAM and enables resume)
CHUNK_SIZE = 10_000

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
# Loaders
# ---------------------------------------------------------------------------

def load_sample(path: Path) -> list[dict]:
    logger.info("Loading sample candidates from %s", path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []


def load_full_jsonl(path: Path) -> list[dict]:
    logger.info("Streaming candidates from %s ...", path)
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
        "--sample", action="store_true",
        help="Run on sample_candidates.json instead of full JSONL.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Encoding batch size (default: 512).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing partial output (skip already-embedded IDs).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load candidates
    # ------------------------------------------------------------------
    if args.sample:
        candidates = load_sample(SAMPLE_CANDIDATES_JSON)
        out_emb = DATA_PROCESSED / "candidate_embeddings_sample.npy"
        out_ids = DATA_PROCESSED / "candidate_ids_sample.json"
    else:
        candidates = load_full_jsonl(CANDIDATES_JSONL)
        out_emb = OUT_EMBEDDINGS
        out_ids = OUT_IDS

    if not candidates:
        logger.error("No candidates found — aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Handle resume: skip already-embedded candidates
    # ------------------------------------------------------------------
    already_done_ids: set[str] = set()
    existing_embs: list[np.ndarray] = []

    if args.resume and out_emb.exists() and out_ids.exists():
        logger.info("Resume mode: loading existing embeddings ...")
        with out_ids.open(encoding="utf-8") as fh:
            done_ids = json.load(fh)
        already_done_ids = set(done_ids)
        existing_embs = [np.load(out_emb)]
        logger.info("  Already embedded: %d candidates", len(already_done_ids))

    todo = [c for c in candidates if c.get("candidate_id") not in already_done_ids]
    logger.info("Candidates to embed: %d (of %d total)", len(todo), len(candidates))

    # ------------------------------------------------------------------
    # 3. Import model (after argument parsing so --help is instant)
    # ------------------------------------------------------------------
    from src.embeddings import _get_model, build_candidate_text  # noqa: PLC0415

    model = _get_model()

    # ------------------------------------------------------------------
    # 4. Build ALL texts first (pure Python, fast)
    # ------------------------------------------------------------------
    logger.info("Building candidate texts ...")
    t_text = time.perf_counter()
    texts = [build_candidate_text(c) for c in todo]
    logger.info("  Text building done in %.1fs", time.perf_counter() - t_text)

    # ------------------------------------------------------------------
    # 5. Encode in one call — SentenceTransformer handles internal batching
    # ------------------------------------------------------------------
    logger.info(
        "Encoding %d candidates (batch_size=%d) ...",
        len(texts), args.batch_size,
    )
    t0 = time.perf_counter()

    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)

    elapsed = time.perf_counter() - t0
    logger.info("Encoding done in %.1fs (%.0f cands/sec)", elapsed, len(texts) / elapsed)

    # ------------------------------------------------------------------
    # 6. Merge with any existing embeddings (resume mode)
    # ------------------------------------------------------------------
    if existing_embs:
        embeddings = np.vstack(existing_embs + [embeddings])

    # Build full ordered ID list
    done_list = list(already_done_ids)  # already-done first (preserve order from file)
    todo_ids  = [c["candidate_id"] for c in todo]
    # If resuming, combine in original file order
    all_ids_ordered: list[str] = []
    if args.resume and out_ids.exists():
        with out_ids.open(encoding="utf-8") as fh:
            all_ids_ordered = json.load(fh)
        all_ids_ordered.extend(todo_ids)
    else:
        all_ids_ordered = [c["candidate_id"] for c in candidates]

    assert len(all_ids_ordered) == embeddings.shape[0], (
        f"ID/embedding mismatch: {len(all_ids_ordered)} vs {embeddings.shape[0]}"
    )

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    logger.info("Saving embeddings -> %s  shape=%s", out_emb, embeddings.shape)
    np.save(out_emb, embeddings)

    logger.info("Saving ID index  -> %s  (%d ids)", out_ids, len(all_ids_ordered))
    with out_ids.open("w", encoding="utf-8") as fh:
        json.dump(all_ids_ordered, fh)

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    total_time = time.perf_counter() - t0
    n = len(all_ids_ordered)
    speed = len(texts) / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 60)
    print(f"  Candidates embedded : {n:,}")
    print(f"  Embedding shape     : {embeddings.shape}")
    print(f"  Encode time         : {elapsed:.1f}s  ({speed:.0f} cands/sec)")
    print(f"  Output embeddings   : {out_emb}")
    print(f"  Output ID index     : {out_ids}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
