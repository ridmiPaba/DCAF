"""
app.py  —  DCAF Frontend  (no training happens here)

Run train.py first, then:
    streamlit run app.py
"""

import os
import pickle
import numpy as np
import tensorflow as tf

try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model_logic import (
    DCAFSystem,
    ContentAnchorLayer,
    EMAFusionLayer,
    GlobalBiasLayer,
)

# ── Paths  (MUST match train.py exactly) ──────────────────────────────
SAVE_DIR            = "saved_model"
PATH_KERAS_MODEL    = os.path.join(SAVE_DIR, "dcaf_keras_model.keras")
PATH_STATE          = os.path.join(SAVE_DIR, "dcaf_state.pkl")
PATH_SPLITS         = os.path.join(SAVE_DIR, "splits.pkl")
PATH_EVAL           = os.path.join(SAVE_DIR, "evaluation_results.pkl")
PATH_EMBEDDINGS     = os.path.join(SAVE_DIR, "content_embeddings.npy")

PATH_ABLATION  = os.path.join(SAVE_DIR, "ablation_results.pkl")
REQUIRED_PATHS = [PATH_KERAS_MODEL, PATH_STATE, PATH_SPLITS, PATH_EVAL, PATH_EMBEDDINGS]

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="DCAF Research Dashboard",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="expanded"
)

# =======================
# STYLING
# =======================
st.markdown("""
<style>
.stApp { background-color: #0f0f17; color: #e0e0ee; }
[data-testid="stSidebar"] {
    background-color: #16161f !important;
    border-right: 1px solid #1e1e2e !important;
    min-width: 220px !important; max-width: 220px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > p > strong,
[data-testid="stSidebar"] hr { opacity: 0; height: 0; margin: 0; }
[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] .stRadio > div {
    display: flex !important; flex-direction: column !important;
    gap: 0 !important; padding: 0 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div:first-child { display: none !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    display: block !important; padding: 9px 20px !important;
    font-size: 0.875rem !important; font-weight: 400 !important;
    color: #6868a0 !important; cursor: pointer !important;
    border-radius: 0 !important; background: transparent !important;
    margin: 0 !important; width: 100% !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: #1e1e2e !important; color: #e0e0ee !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
    background: #1e1e2e !important; color: #e0e0ee !important;
    border-left: 2px solid #e0e0ee !important;
    padding-left: 18px !important; font-weight: 500 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div:last-child { padding: 0 !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div:last-child p {
    font-size: 0.875rem !important; color: inherit !important;
    margin: 0 !important; line-height: 1.4 !important;
}
.metric-card {
    background: #1a1a28; padding: 18px; border-radius: 8px;
    border-left: 3px solid #5555cc;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4); margin-bottom: 14px;
}
.metric-card h3 { color: #9999ee; margin-bottom: 8px; font-size: 1em; font-weight: 600; }
.metric-card .value { font-size: 2.2em; font-weight: 700; color: #e0e0ee; }
.metric-card .label { color: #8888aa; font-size: 0.85em; }
h1 { color: #e0e0ee !important; font-size: 1.55em !important; font-weight: 700 !important; margin-bottom: 2px !important; }
h2 { color: #e0e0ee !important; border-bottom: 1px solid #1e1e2e !important;
     padding-bottom: 8px !important; margin-top: 26px !important; font-size: 1.1em !important; }
h3 { color: #e0e0ee !important; font-size: 1em !important; }
.stButton > button {
    background: #1e1e2e !important; color: #c0c0dd !important;
    border: 1px solid #2e2e42 !important; border-radius: 5px !important;
    padding: 8px 18px !important; font-size: 0.875rem !important;
    font-weight: 500 !important; box-shadow: none !important; transform: none !important;
}
.stButton > button:hover { background: #26263a !important; }
.stProgress > div > div { background-color: #5555cc !important; }
.movie-card {
    background: #1a1a28; padding: 16px; border-radius: 8px;
    margin: 6px 0; border: 1px solid #1e1e2e;
}
.movie-title { font-size: 1em; font-weight: 600; color: #e0e0ee; margin-bottom: 4px; }
.movie-meta  { color: #8888aa; font-size: 0.85em; }
.movie-score { font-size: 1.3em; color: #9999ee; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# =======================
# LOAD SAVED MODEL
# cached — runs only once per Streamlit session
# =======================
@st.cache_resource(show_spinner="Loading pre-trained DCAF model …")
def load_artifacts():
    """
    Load everything from saved_model/.
    Returns (dcaf, traindf, testdf, evaluation_results, content_embeddings)
    or raises FileNotFoundError listing what is missing.
    """
    missing = [p for p in REQUIRED_PATHS if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(missing)

    # Keras model — importing model_logic above is enough to register the
    # custom layers via @tf.keras.utils.register_keras_serializable()
    keras_model = tf.keras.models.load_model(
        PATH_KERAS_MODEL,
        custom_objects={
            "ContentAnchorLayer": ContentAnchorLayer,
            "EMAFusionLayer":     EMAFusionLayer,
            "GlobalBiasLayer":    GlobalBiasLayer,
        }
    )

    # Restore DCAFSystem state
    with open(PATH_STATE, "rb") as f:
        state = pickle.load(f)

    dcaf = DCAFSystem(data_path="data")
    for key, val in state.items():
        setattr(dcaf, key, val)
    dcaf.model = keras_model

    # Train / test splits
    with open(PATH_SPLITS, "rb") as f:
        splits = pickle.load(f)

    # Evaluation results
    with open(PATH_EVAL, "rb") as f:
        evaluation_results = pickle.load(f)

    # Content embeddings
    content_embeddings = np.load(PATH_EMBEDDINGS)

    return dcaf, splits["traindf"], splits["testdf"], evaluation_results, content_embeddings


# ── Try to load; show friendly error if train.py hasn't been run ──────
try:
    dcaf, traindf, testdf, evaluation_results, content_embeddings = load_artifacts()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    missing_files = e.args[0]
except Exception as e:
    model_loaded = False
    missing_files = [str(e)]

if not model_loaded:
    st.title("🎬 DCAF Research Dashboard")
    st.error("### ⚠️  Pre-trained model not found")
    st.markdown("""
The `saved_model/` folder is missing or incomplete.
Run the training script **once** from your terminal first:

```
python train.py
```

Then refresh this page. Training takes about **2–4 minutes** on CPU.
""")
    st.markdown("**Missing files:**")
    for p in missing_files:
        st.markdown(f"- `{p}`")
    st.stop()


# =======================
# SIDEBAR
# =======================
st.sidebar.markdown("""
<div style="padding:22px 20px 8px 20px;">
    <div style="font-size:0.9em;font-weight:700;color:#e0e0ee;letter-spacing:0.02em;">DCAF Dashboard</div>
    <div style="font-size:0.68em;color:#2e2e44;margin-top:3px;line-height:1.5;">FYP — IIT × University of Westminster</div>
</div>
<div style="height:1px;background:#1e1e2e;margin:0 0 4px 0;"></div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Evaluation & Metrics", "Live Recommendations",
     "EMA Profile Explorer", "Content Similarity", "Ablation Study"],
    label_visibility="collapsed"
)

st.sidebar.markdown("""
<div style="height:1px;background:#1e1e2e;margin:8px 0;"></div>
<div style="padding:0 20px 4px 20px;">
    <div style="font-size:0.65em;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;color:#252538;margin-bottom:8px;">Status</div>
    <div style="display:flex;justify-content:space-between;font-size:0.78em;">
        <span style="color:#3a3a52;">Model</span>
        <span style="color:#4ade80;">Loaded ✓</span>
    </div>
</div>
<div style="height:1px;background:#1e1e2e;margin:12px 0;"></div>
<div style="padding:0 20px 20px 20px;font-size:0.68em;color:#1e1e30;line-height:1.7;">© 2025 DCAF Research Project</div>
""", unsafe_allow_html=True)


# ==========================================================
# PAGE: OVERVIEW
# ==========================================================
if page == "Overview":
    st.title("Overview")
    st.markdown("### Dynamic Content-Augmented Factorization Machine with Inverse Propensity Weighting")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        #### Research Project Overview

        This system addresses three interconnected challenges in modern recommender systems:

        1. **Zero/Missing Item Information (Item Cold-Start Problem)**
           - New items lack interaction history
           - Solution: Semantic Content Augmentation using SBERT

        2. **Inherited Popularity Bias**
           - Item with lot of interaction which are very popular always tend to get recommended
           - Some movie content features carry historical popularity bias      
           - Solution: Inverse Propensity Weighting (IPW) in loss function

        3. **Dynamic Item Profile Updating**
           - Smooth transition from content-based to interaction-based profiles
           - Solution: Exponential Moving Average (EMA) Fusion
        """)
        st.info("📚 **Dataset:** MovieLens 1M (1M ratings, 6K users, 3.7K movies)")

    with col2:
        st.markdown("#### Research Questions")
        st.markdown("""
        **RQ1:** Can SBERT provide better cold-start profiles?

        **RQ2:** Does IPW mitigate popularity bias?

        **RQ3:** Does EMA ensure stable performance?
        """)
        st.success("✅ Pre-trained Model Loaded")

    st.markdown("---")
    st.markdown("### System Architecture")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("""<div class="metric-card"><h3>📝 Content Augmentation</h3>
        <p><strong>Encoder:</strong> SBERT (all-MiniLM-L6-v2)</p>
        <p><strong>Input:</strong> Title + Genres</p>
        <p><strong>Output:</strong> 384-dim semantic vector</p></div>""", unsafe_allow_html=True)
    with a2:
        st.markdown("""<div class="metric-card"><h3>⚖️ Bias Mitigation</h3>
        <p><strong>Method:</strong> Inverse Propensity Weighting</p>
        <p><strong>Scaling:</strong> Logarithmic (1.0–5.0 range, p99 cap)</p>
        <p><strong>Target:</strong> Fair item exposure</p></div>""", unsafe_allow_html=True)
    with a3:
        st.markdown("""<div class="metric-card"><h3>🔄 Profile Evolution</h3>
        <p><strong>Mechanism:</strong> EMA Fusion</p>
        <p><strong>Formula:</strong> α(t) = k / (k + t)</p>
        <p><strong>Goal:</strong> Stable adaptation</p></div>""", unsafe_allow_html=True)


# ==========================================================
# PAGE: EVALUATION & METRICS
# ==========================================================
elif page == "Evaluation & Metrics":
    st.title("Evaluation & Metrics")

    results = evaluation_results
    k       = results['k']
    has_full = "overall_full" in results and results["overall_full"]

    st.markdown("""
    <div style="background:#1a1a28;border-left:3px solid #5555cc;padding:14px 18px;
                border-radius:0 8px 8px 0;margin-bottom:20px;">
        <div style="font-weight:600;color:#e0e0ee;margin-bottom:6px;">Two Evaluation Protocols</div>
        <div style="color:#8888aa;font-size:0.87em;line-height:1.8;">
            <span style="color:#9999ee;font-weight:600;">Sampled Ranking (1 vs 100):</span>
            Primary within-project protocol. Produces higher absolute values.
            Used for segmentation analysis and within-project comparisons.<br>
            <span style="color:#4ECDC4;font-weight:600;">Full Ranking (1 vs 3,706):</span>
            All catalogue items scored and ranked. Lower absolute values.
            CLCRec (Wei et al., 2021) and CGRC (Kim et al., 2024) use this protocol —
            DCAF full ranking results can be directly compared to those papers.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### Overall Performance (K={k})")
    col_s, col_f = st.columns(2)

    with col_s:
        st.markdown('''<div style="background:#16162a;border:1px solid #2e2e42;border-radius:8px;
                    padding:10px 14px;margin-bottom:10px;">
            <span style="color:#9999ee;font-weight:600;font-size:0.9em;">Sampled Ranking &#8212; 1 vs 100</span><br>
            <span style="color:#555570;font-size:0.78em;">Primary &#183; within-project comparisons only</span>
        </div>''', unsafe_allow_html=True)
        o = results["overall"]
        s1, s2 = st.columns(2)
        s1.markdown(f'<div class="metric-card"><div class="label">NDCG@{k}</div><div class="value">{o["ndcg"]:.4f}</div></div>', unsafe_allow_html=True)
        s2.markdown(f'<div class="metric-card"><div class="label">Hit Rate@{k}</div><div class="value">{o["hit_rate"]:.4f}</div></div>', unsafe_allow_html=True)
        s3, s4 = st.columns(2)
        s3.markdown(f'<div class="metric-card"><div class="label">Novelty@{k}</div><div class="value">{o["novelty"]:.4f}</div><div class="movie-meta">Higher = less popular items recommended</div></div>', unsafe_allow_html=True)
        s4.markdown(f'<div class="metric-card"><div class="label">Item Coverage</div><div class="value">{o["item_coverage"]:.2%}</div><div class="movie-meta">Proportion of catalogue recommended</div></div>', unsafe_allow_html=True)

    with col_f:
        st.markdown('''<div style="background:#0d2420;border:1px solid #1a4a3a;border-radius:8px;
                    padding:10px 14px;margin-bottom:10px;">
            <span style="color:#4ECDC4;font-weight:600;font-size:0.9em;">Full Ranking &#8212; 1 vs 3,706</span><br>
            <span style="color:#1a5a46;font-size:0.78em;">Comparable to CLCRec (Wei et al., 2021) &amp; CGRC (Kim et al., 2024)</span>
        </div>''', unsafe_allow_html=True)
        if has_full:
            of = results["overall_full"]
            f1c, f2c = st.columns(2)
            f1c.markdown(f'<div class="metric-card" style="border-left-color:#4ECDC4"><div class="label">NDCG@{k}</div><div class="value">{of["ndcg"]:.4f}</div></div>', unsafe_allow_html=True)
            f2c.markdown(f'<div class="metric-card" style="border-left-color:#4ECDC4"><div class="label">Hit Rate@{k}</div><div class="value">{of["hit_rate"]:.4f}</div></div>', unsafe_allow_html=True)
            f3c, f4c = st.columns(2)
            f3c.markdown(f'<div class="metric-card" style="border-left-color:#4ECDC4"><div class="label">Novelty@{k}</div><div class="value">{of["novelty"]:.4f}</div><div class="movie-meta">Higher = less popular items recommended</div></div>', unsafe_allow_html=True)
            f4c.markdown(f'<div class="metric-card" style="border-left-color:#4ECDC4"><div class="label">Item Coverage</div><div class="value">{of["item_coverage"]:.2%}</div><div class="movie-meta">Proportion of catalogue recommended</div></div>', unsafe_allow_html=True)
        else:
            st.info("Full ranking results not found. Re-run train.py to generate both protocols.")

    # ── Literature Benchmarks ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Literature Benchmarks (Full Ranking Protocol)")
    st.markdown("""
    <div style="background:#1a1a28;border-left:3px solid #4ECDC4;padding:12px 16px;
                border-radius:0 8px 8px 0;margin-bottom:16px;">
        <div style="color:#8888aa;font-size:0.85em;line-height:1.7;">
            <b style="color:#e0e0ee;">Important:</b> CLCRec and CGRC use full ranking evaluation on MovieLens.
            DCAF full ranking results (1 vs 3,706) are shown for comparison.
            Differences in data splits, cold-item definitions, and preprocessing
            mean these values are <b style="color:#FFD93D;">indicative, not directly comparable</b>.
            CLCRec and CGRC do not report Hit Rate — they report Recall@10 instead.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if has_full:
        of_bench = results["overall_full"]
        sf_bench = results.get("segments_full", {})
        dcaf_overall_ndcg = f'{of_bench["ndcg"]:.4f}'
        dcaf_overall_hr   = f'{of_bench["hit_rate"]:.4f}'
        dcaf_cold_ndcg = f'{sf_bench["Cold"]["ndcg"]:.4f}' if "Cold" in sf_bench else "N/A"
        dcaf_warm_ndcg = f'{sf_bench["Warm"]["ndcg"]:.4f}' if "Warm" in sf_bench else "N/A"
    else:
        dcaf_overall_ndcg = dcaf_overall_hr = dcaf_cold_ndcg = dcaf_warm_ndcg = "N/A"

    bench_df = pd.DataFrame({
        "Model": [
            "DCAF (ours — full ranking)",
            "CLCRec (Wei et al., 2021)",
            "CGRC (Kim et al., 2024)",
        ],
        "Overall NDCG@10": [dcaf_overall_ndcg, "0.1969", "—"],
        "Cold NDCG@10": [dcaf_cold_ndcg, "0.0444", "0.1604"],
        "Warm NDCG@10": [dcaf_warm_ndcg, "0.2392", "—"],
        "Hit Rate@10": [dcaf_overall_hr, "Not reported", "Not reported"],
        "Protocol": [
            "1 vs 3,706 (all items)",
            "Full ranking (all items)",
            "Full ranking (cold items only)",
        ],
    })
    st.dataframe(bench_df, use_container_width=True, hide_index=True)
    st.caption(
        "CLCRec values: Table 2 of Wei et al. (2021), MovieLens, MF backbone. "
        "CGRC values: Table 2 of Kim et al. (2024), ML-1M — evaluates cold items only (70:15:15 item split). "
        "Neither paper reports Hit Rate@10 or Novelty@10."
    )

    st.markdown("---")
    st.markdown("### Segmentation (Cold / Maturing / Warm)")
    st.caption("Cold = 0-5 interactions | Maturing = 6-50 | Warm = 51+")

    seg_col_s, seg_col_f = st.columns(2)

    with seg_col_s:
        st.markdown(f"**Sampled Ranking — 1 vs 100**")
        if 'segments' in results and results['segments']:
            seg_rows = []
            for name, m in results['segments'].items():
                seg_rows.append({
                    'Segment': name,
                    f'NDCG@{k}': m['ndcg'],
                    f'Hit Rate@{k}': m['hit_rate'],
                    f'Novelty@{k}': m['novelty'],
                    f'Coverage@{k}': m['item_coverage'],
                })
            seg_df = pd.DataFrame(seg_rows)
            st.dataframe(seg_df.style.format({
                f'NDCG@{k}':'{:.4f}', f'Hit Rate@{k}':'{:.4f}',
                f'Novelty@{k}':'{:.4f}', f'Coverage@{k}':'{:.2%}'
            }), use_container_width=True)

    with seg_col_f:
        st.markdown(f"**Full Ranking — 1 vs 3,706**")
        if has_full and 'segments_full' in results and results['segments_full']:
            seg_rows_f = []
            for name, m in results['segments_full'].items():
                seg_rows_f.append({
                    'Segment': name,
                    f'NDCG@{k}': m['ndcg'],
                    f'Hit Rate@{k}': m['hit_rate'],
                    f'Novelty@{k}': m['novelty'],
                    f'Coverage@{k}': m['item_coverage'],
                })
            seg_df_f = pd.DataFrame(seg_rows_f)
            st.dataframe(seg_df_f.style.format({
                f'NDCG@{k}':'{:.4f}', f'Hit Rate@{k}':'{:.4f}',
                f'Novelty@{k}':'{:.4f}', f'Coverage@{k}':'{:.2%}'
            }), use_container_width=True)
        elif not has_full:
            st.info("Re-run train.py to generate full ranking results.")

    if 'segments' in results and results['segments']:
        st.markdown("---")
        st.markdown("#### Segmentation Charts — Sampled Ranking (1 vs 100)")
        st.caption("Charts show sampled ranking results only. Full ranking values are in the table above.")
        colors = ['#4ECDC4', '#FF6B6B', '#FFD93D']
        chart_rows = []
        for name, m in results['segments'].items():
            chart_rows.append({'Segment': name, f'NDCG@{k}': m['ndcg'],
                f'Hit Rate@{k}': m['hit_rate'], f'Novelty@{k}': m['novelty'], f'Coverage@{k}': m['item_coverage']})
        chart_df = pd.DataFrame(chart_rows)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.patch.set_facecolor('#0E1117')
        for ax, col_name, ylabel, title in [
            (axes[0,0], f'NDCG@{k}',    'NDCG',         f'NDCG@{k} by Segment (Sampled 1 vs 100)'),
            (axes[0,1], f'Hit Rate@{k}', 'Hit Rate',     f'Hit Rate@{k} by Segment (Sampled 1 vs 100)'),
            (axes[1,0], f'Novelty@{k}',  'Novelty',      f'Novelty@{k} by Segment (Sampled 1 vs 100)'),
            (axes[1,1], f'Coverage@{k}', 'Item Coverage', f'Coverage@{k} by Segment (Sampled 1 vs 100)'),
        ]:
            ax.bar(chart_df['Segment'], chart_df[col_name], color=colors)
            ax.set_ylabel(ylabel, color='white'); ax.set_title(title, color='white')
            ax.grid(axis='y', alpha=0.3); ax.set_facecolor('#1a1a2e'); ax.tick_params(colors='white')
        plt.tight_layout(); st.pyplot(fig)

    st.markdown("---")
    exp1, exp2 = st.columns(2)
    with exp1:
        if 'segments' in results and results['segments']:
            st.download_button("Export Sampled CSV",
                pd.DataFrame(seg_rows).to_csv(index=False), f"sampled_eval_k{k}.csv", "text/csv")
    with exp2:
        if has_full and 'segments_full' in results and results['segments_full']:
            st.download_button("Export Full Ranking CSV",
                pd.DataFrame(seg_rows_f).to_csv(index=False), f"full_ranking_eval_k{k}.csv", "text/csv")

# ==========================================================
# PAGE: LIVE RECOMMENDATIONS
# ==========================================================
elif page == "Live Recommendations":
    st.title("Live Recommendations")
    st.caption("Watched movies are automatically excluded from every recommendation list.")

    if "existing_recs"   not in st.session_state: st.session_state.existing_recs   = None
    if "existing_userid" not in st.session_state: st.session_state.existing_userid = None

    st.markdown("---")
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        user_id = st.selectbox("Select User ID", dcaf.user_encoder.classes_, index=0)
    with c2:
        k_recs = st.slider("Recommendations", 5, 20, 10, step=1)
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        get_btn = st.button("🎬 Get Recommendations", use_container_width=True, type="primary")

    if get_btn:
        with st.spinner("Generating …"):
            recs = dcaf.get_recommendations(user_id, k=k_recs, exclude_seen=True)
            if recs:
                st.session_state.existing_recs   = recs
                st.session_state.existing_userid = user_id
                st.success(f"✅ {len(recs)} recommendations — watched movies excluded")
            else:
                st.error("No recommendations generated.")

    if st.session_state.existing_recs is not None:
        disp_user = st.session_state.existing_userid
        recs      = st.session_state.existing_recs
        st.markdown("---")

        # User info
        try:
            u_idx    = dcaf.user_encoder.transform([disp_user])[0]
            ur       = dcaf.users[dcaf.users["user_idx"] == u_idx].iloc[0]
            gender   = "Male" if ur["GenderCode"]==1 else "Female"
            age_map  = {1:"Under 18",18:"18–24",25:"25–34",35:"35–44",45:"45–49",50:"50–55",56:"56+"}
            occ_map  = {0:"Other",1:"Academic/Educator",2:"Artist",3:"Clerical/Admin",4:"College Student",
                        5:"Customer Service",6:"Doctor/Health Care",7:"Executive/Manager",8:"Farmer",
                        9:"Homemaker",10:"K-12 Student",11:"Lawyer",12:"Programmer",13:"Retired",
                        14:"Sales/Marketing",15:"Scientist",16:"Self-employed",17:"Technician/Engineer",
                        18:"Tradesman",19:"Unemployed",20:"Writer"}
            st.markdown(f"**User {disp_user}** &nbsp;·&nbsp; {gender} &nbsp;·&nbsp; "
                        f"{age_map.get(int(ur['Age']),str(int(ur['Age'])))} &nbsp;·&nbsp; "
                        f"{occ_map.get(int(ur['Occupation']),'Unknown')}")
        except Exception:
            st.markdown(f"**User {disp_user}**")

        # Genre data
        history_movies = []
        try:
            u_idx      = dcaf.user_encoder.transform([disp_user])[0]
            user_train = traindf[traindf["user_idx"]==u_idx].copy()
            user_train = user_train.merge(dcaf.movies[["movie_idx","Title","Genres"]], on="movie_idx", how="left")
            history_movies = user_train.sort_values(["Rating","Timestamp"],ascending=[False,False]).head(10).to_dict("records")
        except Exception:
            pass

        history_gc = {}
        for row in history_movies:
            for g in str(row.get("Genres","")).split("|"):
                g=g.strip()
                if g and g!="nan": history_gc[g]=history_gc.get(g,0)+1

        rec_gc = {}
        for rec in recs:
            for g in str(rec.get("Genres","")).split("|"):
                g=g.strip()
                if g and g!="nan": rec_gc[g]=rec_gc.get(g,0)+1

        history_gs = set(history_gc.keys()); rec_gs = set(rec_gc.keys())
        matching_g = history_gs & rec_gs;    total_g  = history_gs | rec_gs
        matched_r  = sum(1 for r in recs if set(str(r['Genres']).split("|")) & history_gs)
        n_recs     = len(recs)
        match_pct  = (matched_r/n_recs*100) if n_recs>0 else 0
        all_g = sorted(total_g)
        h_vec = np.array([history_gc.get(g,0) for g in all_g], dtype=float)
        r_vec = np.array([rec_gc.get(g,0)     for g in all_g], dtype=float)
        cos_sim = float(np.dot(h_vec,r_vec)/(np.linalg.norm(h_vec)*np.linalg.norm(r_vec)+1e-12))*100

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("### 👤 Watch History (Top Rated)")
            st.caption("Already seen — excluded from recommendations")
            if history_movies:
                for row in history_movies:
                    rating=row.get("Rating",0); stars="⭐"*int(rating)
                    genre_tags="".join(
                        f'<span style="background:{"#1a3a1a" if g.strip() in rec_gs else "#2d2d44"};'
                        f'color:{"#4ECDC4" if g.strip() in rec_gs else "#888"};'
                        f'padding:2px 6px;border-radius:4px;margin:2px;font-size:0.8em;">'
                        f'{g.strip()}{"✓" if g.strip() in rec_gs else ""}</span>'
                        for g in str(row.get("Genres","")).split("|") if g.strip()
                    )
                    st.markdown(f"""
                    <div style="background:#1e1e2e;border-left:3px solid #4ECDC4;
                                padding:10px 14px;border-radius:8px;margin:6px 0;">
                        <div style="font-weight:bold;color:#FAFAFA;font-size:0.95em;">{row['Title']}</div>
                        <div style="margin:4px 0;">{stars}&nbsp;<span style="color:#888;font-size:0.85em;">rated {rating}/5</span></div>
                        <div>{genre_tags}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No history found.")

        with right_col:
            st.markdown("### 🎬 DCAF Recommendations")
            st.caption("Teal border = genre matches history · watched movies excluded")
            for rec in recs:
                match_found=False; genre_tags=""
                for g in str(rec['Genres']).split("|"):
                    g=g.strip()
                    if g in history_gs:
                        genre_tags+=f'<span style="background:#1a3a1a;color:#4ECDC4;padding:2px 6px;border-radius:4px;margin:2px;font-size:0.8em;">{g} ✓</span>'
                        match_found=True
                    else:
                        genre_tags+=f'<span style="background:#2d2d44;color:#888;padding:2px 6px;border-radius:4px;margin:2px;font-size:0.8em;">{g}</span>'
                border="#4ECDC4" if match_found else "#5555cc"
                st.markdown(f"""
                <div style="background:#1e1e2e;border-left:3px solid {border};
                            padding:10px 14px;border-radius:8px;margin:6px 0;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="font-weight:bold;color:#FAFAFA;font-size:0.95em;">#{rec['Rank']} &nbsp;{rec['Title']}</div>
                        <div style="color:#9999ee;font-weight:bold;">{rec['Score']}</div>
                    </div>
                    <div style="margin-top:4px;">{genre_tags}</div>
                    <div style="color:#444;font-size:0.78em;margin-top:3px;">{rec['InteractionCount']} interactions in dataset</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---"); st.markdown("### 📊 Genre Match Analysis")
        if match_pct>=70:   bc,ic,bt="#1a3a1a","✅","Strong match"
        elif match_pct>=40: bc,ic,bt="#3a3a1a","⚠️","Partial match"
        else:               bc,ic,bt="#3a1a1a","❌","Weak match"
        st.markdown(f"""
        <div style="background:{bc};border-radius:10px;padding:16px 20px;margin-bottom:12px;">
            <div style="font-size:1.4em;font-weight:bold;color:#FAFAFA;">{ic} Relevance: {match_pct:.0f}%</div>
            <div style="color:#ccc;margin-top:4px;">{bt} — <strong>{matched_r} of {n_recs}</strong> recommendations share a genre with this user's history</div>
        </div>""", unsafe_allow_html=True)

        sc1,sc2=st.columns(2)
        sc1.markdown(f'<div class="metric-card" style="padding:12px;"><div class="label">Genre Cosine Similarity</div><div class="value" style="font-size:1.6em;">{cos_sim:.1f}%</div></div>', unsafe_allow_html=True)
        sc2.markdown(f'<div class="metric-card" style="padding:12px;"><div class="label">Shared Genres</div><div class="value" style="font-size:1.6em;">{len(matching_g)} / {len(total_g)}</div><div class="movie-meta">{", ".join(sorted(matching_g)) if matching_g else "none"}</div></div>', unsafe_allow_html=True)

        top_g = sorted(total_g, key=lambda g: history_gc.get(g,0)+rec_gc.get(g,0), reverse=True)[:12]
        fig_g,ax=plt.subplots(figsize=(9,4)); fig_g.patch.set_facecolor('#0E1117')
        x=np.arange(len(top_g)); w=0.38
        ax.bar(x-w/2,[history_gc.get(g,0) for g in top_g],w,label="Watch History",  color='#4ECDC4',alpha=0.9)
        ax.bar(x+w/2,[rec_gc.get(g,0)     for g in top_g],w,label="Recommendations",color='#5555cc',alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(top_g,rotation=45,ha='right',color='white',fontsize=9)
        ax.set_ylabel("Count",color='white'); ax.set_title("Genre Frequency: History vs Recommendations",color='white')
        ax.legend(facecolor='#1a1a2e',labelcolor='white'); ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white'); ax.grid(axis='y',alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_g)

        st.markdown("---")
        st.download_button("📥 Export Recommendations as CSV",
                           pd.DataFrame(recs).to_csv(index=False),
                           f"recommendations_user_{disp_user}.csv","text/csv")

# ==========================================================
# PAGE: EMA PROFILE EXPLORER
# ==========================================================
elif page == "EMA Profile Explorer":
    st.title("EMA Profile Explorer")
    st.caption("Visualise how a movie's profile transitions from content-driven to interaction-driven as it accumulates ratings.")

    st.markdown("---")
    movie_titles = dcaf.movies["Title"].tolist()
    movie_idxs   = dcaf.movies["movie_idx"].tolist()
    title_to_idx = dict(zip(movie_titles, movie_idxs))

    selected_title = st.selectbox("Select a Movie", movie_titles, index=0)
    movie_idx = title_to_idx[selected_title]

    train_count = dcaf.countmap.get(movie_idx, 0)
    st.markdown(f"**Training interactions:** {train_count}  ·  "
                f"**Segment:** {'Cold (0–5)' if train_count <= 5 else 'Maturing (6–50)' if train_count <= 50 else 'Warm (51+)'}")

    # EMA alpha curve
    t_values, alphas, similarities = dcaf.get_ema_evolution_data(movie_idx)

    if t_values:
        st.markdown("#### α(t) Decay Curve")
        st.caption("α(t) = k / (k + t)  with k = 5.0  ·  α=1.0 means pure content, α=0.0 means pure interaction")

        fig1, ax1 = plt.subplots(figsize=(10, 4))
        fig1.patch.set_facecolor('#0E1117')
        ax1.plot(t_values, alphas, color='#9999ee', marker='o', linewidth=2, markersize=6)
        ax1.axhline(y=0.5, color='#4ECDC4', linestyle='--', alpha=0.5, label='Balanced (α=0.5)')
        ax1.axhline(y=0.1, color='#FF6B6B', linestyle='--', alpha=0.5, label='Warm threshold (α=0.1)')
        ax1.axvline(x=train_count, color='#FFD93D', linestyle=':', alpha=0.8, label=f'Current t={train_count}')
        ax1.set_xlabel("Interaction count (t)", color='white')
        ax1.set_ylabel("α(t)", color='white')
        ax1.set_title(f"EMA Alpha Decay — {selected_title}", color='white')
        ax1.legend(facecolor='#1a1a2e', labelcolor='white')
        ax1.set_facecolor('#1a1a2e'); ax1.tick_params(colors='white'); ax1.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig1)

        st.markdown("#### Content Similarity Over Time")
        st.caption("How similar the fused profile stays to the original content anchor as interactions grow")

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig2.patch.set_facecolor('#0E1117')
        ax2.plot(t_values, similarities, color='#4ECDC4', marker='s', linewidth=2, markersize=6)
        ax2.axvline(x=train_count, color='#FFD93D', linestyle=':', alpha=0.8, label=f'Current t={train_count}')
        ax2.set_xlabel("Interaction count (t)", color='white')
        ax2.set_ylabel("Cosine similarity to content anchor", color='white')
        ax2.set_title(f"Content Anchor Similarity — {selected_title}", color='white')
        ax2.legend(facecolor='#1a1a2e', labelcolor='white')
        ax2.set_facecolor('#1a1a2e'); ax2.tick_params(colors='white'); ax2.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig2)

    # Profile vectors
    st.markdown("---")
    st.markdown("#### Item Profile Vectors")
    content_vec, interaction_vec, fused_vec = dcaf.get_item_profile_evolution(movie_idx)
    if content_vec is not None:
        import numpy as np
        t = train_count
        k_decay = 5.0
        alpha = k_decay / (k_decay + t)
        pc1, pc2, pc3 = st.columns(3)
        pc1.markdown(f'<div class="metric-card"><div class="label">Content weight α</div><div class="value">{alpha:.3f}</div><div class="movie-meta">From SBERT encoding</div></div>', unsafe_allow_html=True)
        pc2.markdown(f'<div class="metric-card"><div class="label">Interaction weight (1−α)</div><div class="value">{1-alpha:.3f}</div><div class="movie-meta">From BPR training</div></div>', unsafe_allow_html=True)
        pc3.markdown(f'<div class="metric-card"><div class="label">Content norm</div><div class="value">{float(np.linalg.norm(content_vec)):.3f}</div><div class="movie-meta">Content anchor magnitude</div></div>', unsafe_allow_html=True)


# ==========================================================
# PAGE: CONTENT SIMILARITY
# ==========================================================
elif page == "Content Similarity":
    st.title("Content Similarity")
    st.caption("Find movies most similar to a selected movie based on SBERT content embeddings.")

    st.markdown("---")
    movie_titles = dcaf.movies["Title"].tolist()
    movie_idxs   = dcaf.movies["movie_idx"].tolist()
    title_to_idx = dict(zip(movie_titles, movie_idxs))

    selected_title = st.selectbox("Select a Movie", movie_titles, index=0)
    movie_idx = title_to_idx[selected_title]
    k_sim = st.slider("Number of similar movies", 3, 20, 8)

    movie_row = dcaf.movies[dcaf.movies["movie_idx"] == movie_idx].iloc[0]
    train_count = dcaf.countmap.get(movie_idx, 0)

    st.markdown(f"""
    <div style="background:#1e1e2e;border-left:3px solid #9999ee;padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:16px;">
        <div style="font-weight:bold;color:#FAFAFA;font-size:1em;">{movie_row['Title']}</div>
        <div style="color:#888;font-size:0.85em;margin-top:4px;">{movie_row['Genres']} &nbsp;·&nbsp; {train_count} interactions in dataset</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Computing content similarity …"):
        similar = dcaf.get_similar_items_by_content(movie_idx, k=k_sim)

    if similar:
        st.markdown(f"#### Top {k_sim} Most Similar Movies by Content")
        st.caption("Similarity computed from projected SBERT embeddings (title + genres)")

        for item in similar:
            seg = "Cold" if item["InteractionCount"] <= 5 else "Maturing" if item["InteractionCount"] <= 50 else "Warm"
            seg_color = "#4ECDC4" if seg == "Cold" else "#FFD93D" if seg == "Maturing" else "#9999ee"
            genres_html = "".join(
                f'<span style="background:#2d2d44;color:#888;padding:2px 6px;border-radius:4px;margin:2px;font-size:0.8em;">{g.strip()}</span>'
                for g in str(item["Genres"]).split("|") if g.strip()
            )
            st.markdown(f"""
            <div style="background:#1e1e2e;border:1px solid #2e2e42;padding:12px 16px;border-radius:8px;margin:6px 0;
                        display:flex;justify-content:space-between;align-items:flex-start;">
                <div style="flex:1;">
                    <div style="font-weight:bold;color:#FAFAFA;font-size:0.95em;">{item['Title']}</div>
                    <div style="margin-top:4px;">{genres_html}</div>
                    <div style="color:#555;font-size:0.78em;margin-top:4px;">{item['InteractionCount']} interactions
                        · <span style="color:{seg_color};">{seg}</span></div>
                </div>
                <div style="text-align:right;min-width:80px;">
                    <div style="color:#9999ee;font-weight:bold;font-size:1.1em;">{item['Similarity']:.4f}</div>
                    <div style="color:#555;font-size:0.78em;">similarity</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Similarity bar chart
        st.markdown("---")
        fig_s, ax_s = plt.subplots(figsize=(10, 5))
        fig_s.patch.set_facecolor('#0E1117')
        titles_short = [t["Title"][:35] + "…" if len(t["Title"]) > 35 else t["Title"] for t in similar]
        sims = [t["Similarity"] for t in similar]
        colors_seg = ["#4ECDC4" if t["InteractionCount"] <= 5 else "#FFD93D" if t["InteractionCount"] <= 50 else "#9999ee" for t in similar]
        bars = ax_s.barh(titles_short[::-1], sims[::-1], color=colors_seg[::-1])
        ax_s.set_xlabel("Cosine Similarity", color='white')
        ax_s.set_title(f"Content Similarity to: {selected_title[:40]}", color='white')
        ax_s.set_facecolor('#1a1a2e'); ax_s.tick_params(colors='white'); ax_s.grid(axis='x', alpha=0.3)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#4ECDC4', label='Cold (0–5)'),
                           Patch(facecolor='#FFD93D', label='Maturing (6–50)'),
                           Patch(facecolor='#9999ee', label='Warm (51+)')]
        ax_s.legend(handles=legend_elements, facecolor='#1a1a2e', labelcolor='white')
        plt.tight_layout(); st.pyplot(fig_s)
    else:
        st.warning("Could not compute similarity. Ensure the model is loaded correctly.")


# ==========================================================
# PAGE: ABLATION STUDY
# ==========================================================
elif page == "Ablation Study":
    st.title("Ablation Study")
    st.caption("Each component of DCAF is removed one at a time to measure its individual contribution.")

    st.markdown("""
    <div style="background:#1a1a28;border-left:3px solid #5555cc;padding:14px 18px;
                border-radius:0 8px 8px 0;margin-bottom:20px;">
        <div style="font-weight:600;color:#e0e0ee;margin-bottom:6px;">What is an ablation study?</div>
        <div style="color:#8888aa;font-size:0.87em;line-height:1.8;">
            An ablation study proves that each component of a model genuinely contributes to its performance.
            Three variants of DCAF are trained, each with one component removed and compared against the
            full model. If removing a component causes a drop in NDCG or Hit Rate, that component is doing
            real work.<br><br>
            <span style="color:#9999ee;">Full DCAF</span> — all components active (pre-trained model, loaded directly).<br>
            <span style="color:#FF6B6B;">No IPW</span> — Inverse Propensity Weighting removed. All BPR pairs weighted equally.<br>
            <span style="color:#FFD93D;">No EMA</span> — EMA fusion disabled. Alpha fixed at 1.0, pure content anchor always.<br>
            <span style="color:#4ECDC4;">No SBERT</span> — SBERT embeddings replaced with random vectors of the same shape.<br>
            <span style="color:#FF9F43;">MF-BPR</span> — Pure Matrix Factorisation baseline (Rendle et al., 2009). No content, no EMA, no IPW. Standard collaborative filtering lower bound.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not os.path.exists(PATH_ABLATION):
        st.warning("""
        **Ablation results not found.**

        Run the ablation script from your terminal first:
        ```
        python ablation.py
        ```
        This trains 3 model variants (~15 minutes with default 10 epochs).
        Results will appear here automatically once complete.
        """)
        st.stop()

    with open(PATH_ABLATION, "rb") as _f:
        ablation_data = pickle.load(_f)

    abl_results = ablation_data["results"]
    abl_epochs  = ablation_data["epochs"]
    abl_k       = ablation_data["k"]

    st.info(f"Results based on **{abl_epochs} training epochs** per variant · Sampled ranking (1 vs 100) · K={abl_k}"
            + (" · Change EPOCHS=30 in ablation.py for full training." if abl_epochs < 30 else ""))

    # ── Summary metrics cards ──────────────────────────────────────────
    st.markdown("### Overall Performance Comparison")
    variant_colors = {
        "Full DCAF":                    "#9999ee",
        "No IPW":                       "#FF6B6B",
        "No EMA":                       "#FFD93D",
        "No SBERT":                     "#4ECDC4",
        "MF-BPR (Rendle et al., 2009)": "#FF9F43",
    }

    cols = st.columns(len(abl_results))
    for col, (name, res) in zip(cols, abl_results.items()):
        o = res["overall"]
        color = variant_colors.get(name, "#5555cc")
        col.markdown(f"""
        <div style="background:#1a1a28;border-left:4px solid {color};padding:14px;border-radius:0 8px 8px 0;margin-bottom:8px;">
            <div style="font-weight:700;color:{color};font-size:0.9em;margin-bottom:8px;">{name}</div>
            <div style="color:#e0e0ee;font-size:1.6em;font-weight:700;">{o['ndcg']:.4f}</div>
            <div style="color:#888;font-size:0.78em;">NDCG@{abl_k}</div>
            <div style="color:#e0e0ee;font-size:1.2em;font-weight:600;margin-top:6px;">{o['hit_rate']:.4f}</div>
            <div style="color:#888;font-size:0.78em;">Hit Rate@{abl_k}</div>
            <div style="color:#e0e0ee;font-size:1.0em;margin-top:6px;">{o['novelty']:.4f}</div>
            <div style="color:#888;font-size:0.78em;">Novelty@{abl_k}</div>
            <div style="color:#e0e0ee;font-size:1.0em;margin-top:6px;">{o['item_coverage']:.2%}</div>
            <div style="color:#888;font-size:0.78em;">Item Coverage</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Summary table ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Summary Table")

    full_ndcg = abl_results["Full DCAF"]["overall"]["ndcg"]
    table_rows = []
    for name, res in abl_results.items():
        o = res["overall"]
        drop = ((full_ndcg - o["ndcg"]) / full_ndcg * 100) if name != "Full DCAF" else 0.0
        table_rows.append({
            "Variant":        name,
            f"NDCG@{abl_k}":      round(o["ndcg"], 4),
            f"Hit Rate@{abl_k}":  round(o["hit_rate"], 4),
            f"Novelty@{abl_k}":   round(o["novelty"], 4),
            "Item Coverage":  f"{o['item_coverage']:.2%}",
            "NDCG Drop vs Full": f"{drop:+.1f}%" if name != "Full DCAF" else "—",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ── Bar charts ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Visual Comparison")

    names   = list(abl_results.keys())
    colors  = [variant_colors.get(n, "#5555cc") for n in names]
    ndcgs   = [abl_results[n]["overall"]["ndcg"]         for n in names]
    hrs     = [abl_results[n]["overall"]["hit_rate"]      for n in names]
    novelty = [abl_results[n]["overall"]["novelty"]       for n in names]
    cov     = [abl_results[n]["overall"]["item_coverage"] for n in names]

    # Short display labels for chart x-axis to avoid overlap
    short_labels = {
        "Full DCAF":                    "Full DCAF",
        "No IPW":                       "No IPW",
        "No EMA":                       "No EMA",
        "No SBERT":                     "No SBERT",
        "MF-BPR (Rendle et al., 2009)": "MF-BPR baseline",
    }
    chart_names = [short_labels.get(n, n) for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor('#0E1117')

    for ax, vals, ylabel, title in [
        (axes[0,0], ndcgs,   f"NDCG@{abl_k}",     f"NDCG@{abl_k} — Ablation Comparison"),
        (axes[0,1], hrs,     f"Hit Rate@{abl_k}",  f"Hit Rate@{abl_k} — Ablation Comparison"),
        (axes[1,0], novelty, f"Novelty@{abl_k}",   f"Novelty@{abl_k} — Ablation Comparison"),
        (axes[1,1], cov,     "Item Coverage",       "Item Coverage — Ablation Comparison"),
    ]:
        bars = ax.bar(chart_names, vals, color=colors)
        ax.set_ylabel(ylabel, color='white')
        ax.set_title(title, color='white')
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.tick_params(axis='x', rotation=15, labelsize=9)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, vals):
            label = f"{val:.4f}" if ylabel != "Item Coverage" else f"{val:.2%}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    label, ha='center', va='bottom', color='white', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    # ── Segmentation comparison ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Segmentation Comparison (Cold / Maturing / Warm)")
    st.caption("Shows how removing each component affects Cold, Maturing, and Warm items separately.")

    seg_names = ["Cold", "Maturing", "Warm"]
    seg_metric = f"NDCG@{abl_k}"

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.patch.set_facecolor('#0E1117')

    for ax, seg in zip(axes2, seg_names):
        seg_vals = []
        for name in names:
            segs = abl_results[name].get("segments", {})
            val  = segs.get(seg, {}).get("ndcg", 0.0)
            seg_vals.append(val)
        bars = ax.bar(chart_names, seg_vals, color=colors)
        ax.set_title(f"{seg} items — NDCG@{abl_k}", color='white')
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white', axis='both')
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel(f"NDCG@{abl_k}", color='white')
        for bar, val in zip(bars, seg_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha='center', va='bottom', color='white', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig2)

    # ── Export ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.download_button(
        "📥 Export Ablation Results CSV",
        pd.DataFrame(table_rows).to_csv(index=False),
        "ablation_results.csv", "text/csv"
    )