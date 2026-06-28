import streamlit as st
import json
import pandas as pd
import sys
from pathlib import Path
import time
import numpy as np

# Adjust path to find src modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.disqualifiers import check_disqualifiers
from src.honeypot_detector import detect_honeypot
from src.behavioral_scorer import score_behavior
from src.embeddings import build_candidate_text, embed_jd
from src.fusion_ranker import rank_candidates_batch
from src.reasoning_generator import generate_reasoning
from sentence_transformers import SentenceTransformer

# Load JD embedding on startup
@st.cache_resource
def load_jd_and_model():
    jd_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "jd_parsed.json"
    jd = json.loads(jd_path.read_text(encoding="utf-8"))
    
    # We must embed on the fly in sandbox to show full e2e capability
    jd_emb = embed_jd(jd)
    
    # Preload model for candidates
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return jd_emb, model

st.title("Redrob Candidate Ranker — Demo")
st.markdown("""
Upload a JSON file containing a list of candidates (max 100).
The system will run the full ranking pipeline (Disqualifiers, Honeypot Detector, Behavioral Scorer, Semantic Similarity) and return the ranked results.

*Note: The full 100K ranking runs in ~68 seconds on CPU using pre-computed embeddings. This demo runs the end-to-end embedding and scoring live for the uploaded sample.*
""")

uploaded_file = st.file_uploader("Upload candidates JSON", type=["json"])

if uploaded_file is not None:
    try:
        candidates = json.load(uploaded_file)
        if not isinstance(candidates, list):
            st.error("JSON must contain a list of candidate objects.")
            st.stop()
            
        if len(candidates) > 100:
            st.warning(f"File contains {len(candidates)} candidates. Truncating to 100 for this demo.")
            candidates = candidates[:100]
            
        st.write(f"Processing {len(candidates)} candidates...")
        
        with st.spinner("Loading models and JD..."):
            jd_emb, model = load_jd_and_model()
            
        with st.spinner("Running full ranking pipeline..."):
            start_time = time.time()
            
            # Embed candidates
            texts = [build_candidate_text(c) for c in candidates]
            cand_embs = model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
            
            # Rank
            results = rank_candidates_batch(candidates, jd_emb, cand_embs)
            
            # Add reasoning
            cand_map = {c.get("candidate_id", f"IDX_{i}"): c for i, c in enumerate(candidates)}
            
            table_data = []
            for rank, r in enumerate(results, start=1):
                cid = r["candidate_id"]
                cand = cand_map.get(cid, {"candidate_id": cid})
                reasoning = generate_reasoning(cand, r)
                
                table_data.append({
                    "rank": rank,
                    "candidate_id": cid,
                    "score": round(r["final_score"], 4),
                    "reasoning": reasoning,
                    "hard_floored": r["hard_floored"],
                })
                
            elapsed = time.time() - start_time
            
            st.success(f"Ranking completed in {elapsed:.2f} seconds!")
            
            df = pd.DataFrame(table_data)
            
            # Summary stats
            disqualified_count = sum(1 for r in table_data if r["hard_floored"])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Candidates Processed", len(candidates))
            col2.metric("Disqualified/Honeypots", disqualified_count)
            col3.metric("Time Taken (s)", f"{elapsed:.2f}")
            
            st.dataframe(df, use_container_width=True)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
